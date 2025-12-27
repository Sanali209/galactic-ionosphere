import cv2
import numpy as np
import mss
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from threading import Thread
import time

# === SegmentMapper класс ===
class SegmentMapper:
    def __init__(self, frame_size=(640, 480), grid_size=(6, 6)):
        self.frame_h, self.frame_w = frame_size[1], frame_size[0]
        self.grid_rows, self.grid_cols = grid_size
        self.seg_h = self.frame_h // self.grid_rows
        self.seg_w = self.frame_w // self.grid_cols
        self.segment_map = []
        self.player_pos = np.array([0.0, 0.0])
        self.detector = cv2.ORB_create(500)

    def extract_segments(self, frame):
        segments = []
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                y1 = i * self.seg_h
                x1 = j * self.seg_w
                seg_img = frame[y1:y1+self.seg_h, x1:x1+self.seg_w]
                kp, des = self.detector.detectAndCompute(seg_img, None)
                segments.append({
                    "img": seg_img,
                    "keypoints": kp,
                    "descriptors": des,
                    "grid_pos": (i, j)
                })
        return segments

    def match_segment(self, new_des, existing_des):
        if new_des is None or existing_des is None:
            return 0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(new_des, existing_des)
        return len([m for m in matches if m.distance < 50])

    def update(self, frame, is_first_frame=False):
        segments = self.extract_segments(frame)

        if is_first_frame:
            self.segment_map.clear()
            for seg in segments:
                seg["world_pos"] = (0, 0)
                self.segment_map.append(seg)
            self.player_pos = np.array([0.0, 0.0])
            return segments, self.draw_map()

        # Пытаемся сопоставить каждый сегмент с уже известной картой
        for seg in segments:
            best_match = 0
            matched_seg = None
            for known in self.segment_map:
                score = self.match_segment(seg["descriptors"], known["descriptors"])
                if score > best_match:
                    best_match = score
                    matched_seg = known

            if best_match > 15:
                # Уже известный сегмент
                seg["world_pos"] = matched_seg["world_pos"]
            else:
                # Новый сегмент — добавляем рядом с игроком
                i, j = seg["grid_pos"]
                rel_x = j - self.grid_cols // 2
                rel_y = i - self.grid_rows // 2
                new_pos = (int(self.player_pos[0] + rel_x), int(self.player_pos[1] + rel_y))

                exists = any(s["world_pos"] == new_pos for s in self.segment_map)
                if not exists:
                    seg["world_pos"] = new_pos
                    self.segment_map.append(seg)

        # Обновляем позицию игрока на основе центра
        self.player_pos = np.array([0.0, 0.0])
        center_idx = (self.grid_rows // 2) * self.grid_cols + (self.grid_cols // 2)
        if center_idx < len(segments):
            self.player_pos = np.array(segments[center_idx].get("world_pos", [0, 0]))

        return segments, self.draw_map()

    def draw_map(self, scale=20, mode='map'):
        if not self.segment_map:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        coords = [seg["world_pos"] for seg in self.segment_map]
        xs, ys = zip(*coords)
        min_x, min_y = min(xs), min(ys)
        max_x, max_y = max(xs), max(ys)

        map_w = (max_x - min_x + 1) * scale
        map_h = (max_y - min_y + 1) * scale
        map_img = np.zeros((map_h, map_w, 3), dtype=np.uint8)

        for seg in self.segment_map:
            x, y = seg["world_pos"]
            px = (x - min_x) * scale
            py = (y - min_y) * scale

            if mode == 'map':
                cv2.rectangle(map_img, (px, py), (px + scale, py + scale), (0, 255, 0), -1)
            elif mode == 'heatmap':
                count = len(seg["keypoints"]) if seg["keypoints"] else 0
                intensity = min(255, count * 5)
                cv2.rectangle(map_img, (px, py), (px + scale, py + scale), (0, intensity, 255 - intensity), -1)
            elif mode == 'keypoints':
                resized = cv2.resize(seg["img"], (scale, scale))
                map_img[py:py+scale, px:px+scale] = resized

        # Положение игрока
        px = (int(self.player_pos[0]) - min_x) * scale
        py = (int(self.player_pos[1]) - min_y) * scale
        cv2.circle(map_img, (px + scale//2, py + scale//2), scale//3, (0, 0, 255), -1)

        return map_img

    def reset(self):
        self.segment_map.clear()
        self.player_pos = np.array([0.0, 0.0])


# === GUI приложение ===
class VisualMappingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visual RPG Mapper")
        self.mapper = SegmentMapper()
        self.first_frame = True
        self.running = False
        self.view_mode = tk.StringVar(value='map')

        # Верхняя панель
        top_frame = tk.Frame(root)
        top_frame.pack()

        self.start_button = tk.Button(top_frame, text="Старт", command=self.toggle_mapping)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = tk.Button(top_frame, text="Сброс карты", command=self.reset_map)
        self.reset_button.pack(side=tk.LEFT, padx=5)

        self.mode_menu = ttk.Combobox(top_frame, textvariable=self.view_mode,
                                      values=['map', 'heatmap', 'keypoints'], width=12, state="readonly")
        self.mode_menu.pack(side=tk.LEFT, padx=5)

        # Области отображения
        self.label_game = tk.Label(root)
        self.label_game.pack(side=tk.LEFT)

        self.label_map = tk.Label(root)
        self.label_map.pack(side=tk.RIGHT)

        # Поток обновления
        self.thread = Thread(target=self.update_loop, daemon=True)
        self.thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def toggle_mapping(self):
        self.running = not self.running
        self.first_frame = self.running
        self.start_button.config(text="Пауза" if self.running else "Старт")

    def reset_map(self):
        self.mapper.reset()
        self.first_frame = True

    def update_loop(self):
        with mss.mss() as sct:
            region = {'top': 100, 'left': 100, 'width': 640, 'height': 360}
            while True:
                if self.running:
                    screen = np.array(sct.grab(region))
                    frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

                    segments, _ = self.mapper.update(frame, self.first_frame)
                    self.first_frame = False

                    # Отображение экрана
                    disp = frame.copy()
                    for seg in segments:
                        for kp in seg["keypoints"]:
                            x, y = kp.pt
                            i, j = seg["grid_pos"]
                            cv2.circle(disp, (int(x + j * self.mapper.seg_w), int(y + i * self.mapper.seg_h)), 2, (255, 0, 0), -1)

                    map_mode = self.view_mode.get()
                    map_vis = self.mapper.draw_map(mode=map_mode)

                    self.update_tk_image(self.label_game, disp)
                    self.update_tk_image(self.label_map, map_vis)

                time.sleep(0.03)

    def update_tk_image(self, widget, img_cv):
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        widget.imgtk = imgtk
        widget.configure(image=imgtk)

    def on_close(self):
        self.root.quit()
        self.root.destroy()


# === Запуск ===
if __name__ == "__main__":
    root = tk.Tk()
    app = VisualMappingApp(root)
    root.mainloop()
