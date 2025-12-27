import cv2
import numpy as np

from SLM.botbox.Environment import EnvironmentEntity


class ZoneMap2(EnvironmentEntity):
    def __init__(self):
        super().__init__()
        self.name = 'ZoneMap'
        self.map_size = (600, 600)
        self.tile_size = 100
        self.obstacle_color = (148, 99, 74)
        self.map_data = np.zeros((self.map_size[1], self.map_size[0], 3), np.uint8)
        self.player_position = [0, 0]
        self.draw_image_buffer = np.zeros((self.map_size[1], self.map_size[0], 3), np.uint8)

    def load_from_minimap(self, minimap_image_path):
        minimap_image = cv2.imread(minimap_image_path)
        minimap_image = cv2.cvtColor(minimap_image, cv2.COLOR_BGR2RGB)
        self.map_data = cv2.resize(minimap_image, (self.map_size[0], self.map_size[1]), interpolation=cv2.INTER_AREA)

def move_to_target(blackboard):
    dist_tolerance = 5
    # Get the current position and target position from the blackboard
    current_position = blackboard.get_value('current_position')
    target_position = blackboard.get_value('target_position')
    if current_position is None or target_position is None:
        return "failure"
    # Calculate the distance to the target
    distance_x = target_position[0] - current_position[0]
    distance_y = target_position[1] - current_position[1]

    # If the distance is small enough, we consider the movement complete
    if abs(distance_x) < dist_tolerance and abs(distance_y) < dist_tolerance:
        return "success"

    move_dir = [0, 0]

    if distance_x > 0:
        move_dir[0] = 1
    elif distance_x < 0:
        move_dir[0] = -1

    if distance_y > 0:
        move_dir[1] = 1
    elif distance_y < 0:
        move_dir[1] = -1
    action = NothingAction("nothing")
    keys = [0, 0]
    #konvert to action
    if move_dir[0] == 1:
        keys[0] = 'right'
    elif move_dir[0] == -1:
        keys[0] = 'left'
    if move_dir[1] == 1:
        keys[1] = 'down'
    elif move_dir[1] == -1:
        keys[1] = 'up'
    #rem 0
    keys = [key for key in keys if key != 0]
    action = KeyDownAction('move', keys)
    blackboard.set_value('action', action)
    print("Moving to", target_position)
    print("Current position", current_position)

    return "running"

class MapGrabber(EntityController):
    def __init__(self):
        super().__init__()
        self.name = 'MapGrabber'
        self.update_interval = 100
        self.paused = True
        self.prew_frame = None
        self.time_from_last_update = 0
        self.last_update_time_stump = 0
        self.flow = None
        self.flow_coefficients = (1, 1)
        self.global_map = None
        self.pos = (0, 0)

    def grab_screen_and_merge(self):
        screen_image = self.parentEntity.env.GetController('screen_capturer').grab_buffer.get('grey')
        if screen_image is None:
            return

        next_gray = screen_image
        if self.prew_frame is None:
            self.prew_frame = next_gray
            return
        try:
            sx, sy = detect_shift(self.prew_frame, next_gray)
        except:
            sx, sy = 0, 0
        self.prew_frame = next_gray

        print(f"Shift x: {sx}, y: {sy}")
        #sx=sx*-1
        #sy=sy*-1

        self.global_map, self.pos = self.stitch_im(self.global_map, next_gray, (sx, sy), self.pos)

        #self.global_map = stitch_with_semitransparent_edges(self.global_map, next_gray)

    def stitch_im(self, canvas, img2, shift, last_paste_pos):
        """Stitches an image onto a canvas with potential positive or negative shift.

        Args:
            canvas (numpy.ndarray): The canvas image.
            img2 (numpy.ndarray): The image to be stitched.
            shift (tuple): (x_shift, y_shift) representing the shift.
            last_paste_pos (tuple): (x_pos, y_pos) representing the last paste position.

        Returns:
            tuple: (updated_canvas, new_last_paste_pos)
        """
        if canvas is None:
            return img2, (0, 0)
        x_shift, y_shift = shift
        x_shift = int(x_shift)
        y_shift = int(y_shift)
        x_pos, y_pos = last_paste_pos

        # Calculate new paste position considering shift
        new_x_pos = x_pos + x_shift
        new_y_pos = y_pos + y_shift

        # Calculate required canvas size

        if new_x_pos == 0 or new_x_pos > 0:
            firs_img_x = 0
            second_img_x = new_x_pos
        else:
            firs_img_x = abs(new_x_pos)
            second_img_x = 0
        if new_y_pos == 0 or new_y_pos > 0:
            firs_img_y = 0
            second_img_y = new_y_pos
        else:
            firs_img_y = abs(new_y_pos)
            second_img_y = 0
        first_rect = (firs_img_y, canvas.shape[0] + firs_img_y, firs_img_x, canvas.shape[1] + firs_img_x)

        second_rect = (second_img_y, img2.shape[0] + second_img_y, second_img_x, img2.shape[1] + second_img_x)
        canvas_height = max(first_rect[1], second_rect[1])
        canvas_width = max(first_rect[3], second_rect[3])

        new_canvas = np.zeros((canvas_height, canvas_width), np.uint8)
        mask_img2 = self.circular_mask(img2.shape[0], img2.shape[1], center=(img2.shape[1] // 2, img2.shape[0] // 2))
        alpha = 0.7

        try:
            new_canvas[first_rect[0]:first_rect[1], first_rect[2]:first_rect[3]] = canvas
        except:
            pass
        try:
            orig_data = new_canvas[second_rect[0]:second_rect[1], second_rect[2]:second_rect[3]]
            orig_data = cv2.addWeighted(orig_data, 1 - alpha, img2, alpha, 0)
            new_canvas[second_rect[0]:second_rect[1], second_rect[2]:second_rect[3]] = orig_data

            #new_canvas[second_rect[0]:second_rect[1], second_rect[2]:second_rect[3]] = img2
        except Exception as e:
            pass

        if new_x_pos < 0:
            new_x_pos = 0
        if new_y_pos < 0:
            new_y_pos = 0

        # Blend in place
        new_last_paste_pos = (new_x_pos, new_y_pos)

        return new_canvas, new_last_paste_pos

    def circular_mask(self, h, w, center=None, radius=None):

        if center is None:
            center = (int(w / 2), int(h / 2))
        if radius is None:
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        mask = dist_from_center <= radius
        # bluring mask
        mask = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 11)
        return mask

    def render(self):
        image = self.global_map
        if image is None:
            return
        # downscale to 800*600
        image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_AREA)
        self.env.renderer.draw_image(image, (0, 0))

    def update(self):
        needUpdate = super().update()
        if not needUpdate or self.paused:
            return
        self.grab_screen_and_merge()
