import cv2
import numpy as np
import mss
import time
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
from scipy.interpolate import splprep, splev

# -------------------- Настройка --------------------
SCREEN_REGION = {'left': 100, 'top': 100, 'width': 700, 'height': 700}
MAX_CORNERS = 500
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 7
LK_PARAMS = dict(winSize=(15, 15), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
CAPTURE_INTERVAL = 50  # ms
DIST_THRESHOLD = 20.0  # px для добавления новых точек

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# -------------------- Состояния --------------------
sct = mss.mss()
prev_gray = None
prev_pts = None
world_pos = np.array([0.0, 0.0])
t_prev = time.time()
last_control_point = None

tracking_started = False

class MapTracker:
    def __init__(self):
        self.control_points = []
        self.curve = []

    def add_point(self, point):
        self.control_points.append(tuple(point))
        self.recalculate_curve()

    def recalculate_curve(self):
        cp = np.array(self.control_points)
        if len(cp) < 4:
            self.curve = cp.tolist()  # просто вернуть точки без интерполяции
            return

        try:
            tck, _ = splprep([cp[:, 0], cp[:, 1]], s=0)
            u = np.linspace(0, 1, 200)
            x_i, y_i = splev(u, tck)
            self.curve = list(zip(x_i, y_i))
        except Exception as e:
            print(f"Ошибка при интерполяции кривой: {e}")
            self.curve = cp.tolist()

    def correct_position(self, current_pos):
        if not self.curve:
            return current_pos
        curve = np.array(self.curve)
        dists = np.linalg.norm(curve - current_pos, axis=1)
        nearest = curve[np.argmin(dists)]
        return nearest

map_tracker = MapTracker()

# -------------------- Функции --------------------
def initialize_features(gray):
    gftt = cv2.GFTTDetector_create(maxCorners=MAX_CORNERS, qualityLevel=QUALITY_LEVEL, minDistance=MIN_DISTANCE)
    keypoints = gftt.detect(gray)
    if keypoints:
        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
        return pts
    return None

def preprocess_gray(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    enhanced = CLAHE.apply(blurred)
    return enhanced

def annotate_frame(frame, good_prev, good_next):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    # Отрисовка треков
    for (x0, y0), (x1, y1) in zip(good_prev, good_next):
        draw.line((x0, y0, x1, y1), fill="green", width=1)
        r = 3
        draw.ellipse((x1-r, y1-r, x1+r, y1+r), fill="red")

    # Контрольные точки (синие кружки)
    for pt in map_tracker.control_points:
        r = 4
        draw.ellipse((pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r), outline="blue")

    # Интерполированная кривая (красная линия) + стрелки
    arrow_step = 10  # каждые N точек
    arrow_size = 10
    curve = map_tracker.curve

    if len(curve) >= 2:
        for i in range(len(curve) - 1):
            x1, y1 = curve[i]
            x2, y2 = curve[i+1]
            draw.line((x1, y1, x2, y2), fill="red")

        for i in range(0, len(curve) - 1, arrow_step):
            x1, y1 = curve[i]
            x2, y2 = curve[i+1]
            angle = np.arctan2(y2 - y1, x2 - x1)

            # Конец стрелки
            tip_x = x2
            tip_y = y2

            # Ветки стрелки
            left = (tip_x - arrow_size * np.cos(angle - np.pi / 6),
                    tip_y - arrow_size * np.sin(angle - np.pi / 6))
            right = (tip_x - arrow_size * np.cos(angle + np.pi / 6),
                     tip_y - arrow_size * np.sin(angle + np.pi / 6))

            draw.line((left[0], left[1], tip_x, tip_y), fill="orange", width=2)
            draw.line((right[0], right[1], tip_x, tip_y), fill="orange", width=2)

    return img


def process_frame():
    global prev_gray, prev_pts, world_pos, t_prev, last_control_point

    frame = np.array(sct.grab(SCREEN_REGION))
    gray = preprocess_gray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    if prev_gray is None or prev_pts is None or len(prev_pts) < 10:
        prev_gray = gray
        prev_pts = initialize_features(gray)
        t_prev = time.time()
        return frame, None, None

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **LK_PARAMS)
    if next_pts is None:
        prev_pts = initialize_features(gray)
        return frame, None, None

    mask = status.flatten() == 1
    good_prev = prev_pts[mask].reshape(-1, 2)
    good_next = next_pts[mask].reshape(-1, 2)

    if len(good_prev) >= 5:
        disp = good_next - good_prev
        avg_disp = disp.mean(axis=0)
        dt = time.time() - t_prev
        move = -avg_disp * dt
        world_pos[:] += move

        if map_tracker.control_points:
            corrected = map_tracker.correct_position(world_pos)
            correction_vector = corrected - world_pos
            world_pos[:] += correction_vector * 0.2  # мягкая коррекция

        speed = np.linalg.norm(move) / max(dt, 1e-5)
        direction = np.degrees(np.arctan2(move[1], move[0]))

        print(f"World: {world_pos.round(1)} | Speed: {speed:.2f} px/s | Dir: {direction:.1f}°")

        # Добавить новую точку при прохождении DIST_THRESHOLD
        if tracking_started:
            if last_control_point is None or np.linalg.norm(world_pos - last_control_point) > DIST_THRESHOLD:
                map_tracker.add_point(world_pos.copy())
                last_control_point = world_pos.copy()

        prev_gray[:] = gray
        prev_pts = good_next.reshape(-1, 1, 2)

    return frame, good_prev, good_next

# -------------------- GUI --------------------
class TrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Player Map Tracker")
        self.label = tk.Label(root)
        self.label.pack()

        self.start_button = tk.Button(root, text="Старт", command=self.start_tracking)
        self.start_button.pack()

        self.update_frame()

    def start_tracking(self):
        global tracking_started, world_pos, last_control_point
        map_tracker.control_points.clear()
        world_pos[:] = [SCREEN_REGION['width']//2, SCREEN_REGION['height']//2]
        map_tracker.add_point(world_pos.copy())
        last_control_point = world_pos.copy()
        tracking_started = True
        print("Старт: добавлена первая контрольная точка.")

    def update_frame(self):
        frame, good_prev, good_next = process_frame()
        if frame is not None:
            if good_prev is not None and good_next is not None:
                img = annotate_frame(frame, good_prev, good_next)
            else:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            photo = ImageTk.PhotoImage(img)
            self.label.config(image=photo)
            self.label.image = photo
        self.root.after(CAPTURE_INTERVAL, self.update_frame)

if __name__ == '__main__':
    print("Оптический трекинг игрока с картой корректирующих точек")
    root = tk.Tk()
    app = TrackerApp(root)
    root.mainloop()
