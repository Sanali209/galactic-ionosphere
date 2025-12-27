import cv2
import numpy as np
import pygame
from PySide6.QtGui import QImage, QPixmap

def get_tile(sheet, x, y, tile_size):
    tile = pygame.Surface((tile_size[0], tile_size[1]))
    tile.blit(sheet, (0, 0), (x * tile_size[0], y * tile_size[1], tile_size[0], tile_size[1]))
    tile.set_colorkey((0, 0, 0))  # Assuming black is the transparent color
    return tile


def get_frame(sheet, frame_x, frame_y, frame_width, frame_height):
    frame = pygame.Surface((frame_width, frame_height))
    frame.blit(sheet, (0, 0), (frame_x, frame_y, frame_width, frame_height))
    frame.set_colorkey((0, 0, 0))  # Assuming black is the transparent color
    return frame

def find_keypoints_and_descriptors(image):
    # Использование SIFT для нахождения ключевых точек и дескрипторов
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_descriptors(des1, des2):
    # Сопоставление дескрипторов с использованием FLANN-based matcher
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Применение ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches

def find_translation(kp1, kp2, matches):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    translation = np.mean(dst_pts - src_pts, axis=0)
    return translation

def detect_shift(img1, img2):
    """Detects shift between two images using SIFT feature matching.

    Args:
        img1 (numpy.ndarray): First image.
        img2 (numpy.ndarray): Second image.

    Returns:
        tuple: Estimated shift (x_shift, y_shift) or None if not enough matches.
    """

    sift = cv2.SIFT_create()  # Use SIFT for more robust features
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Create a FLANN-based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except:
        return 0, 0

    # Apply Lowe's ratio test for better matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # Adjust ratio for better filtering
            good_matches.append(m)

    min_matches = 10  # Adjust for robustness
    if len(good_matches) > min_matches:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Calculate average shift using the median
        x_shifts = [dst_pts[i][0][0] - src_pts[i][0][0] for i in range(len(good_matches))]
        y_shifts = [dst_pts[i][0][1] - src_pts[i][0][1] for i in range(len(good_matches))]
        x_shift = np.median(x_shifts)  # Use median for robustness
        y_shift = np.median(y_shifts)  # Use median for robustness

        # invert results
        x_shift = int(x_shift*-1)
        y_shift = int(y_shift*-1)

        return (x_shift, y_shift)
    else:
        return None

def stitch_with_semitransparent_edges(img1, img2, alpha=0.5):
    """Stitches two images with semitransparent edges using detected shift.

    Args:
        img1 (numpy.ndarray): First image.
        img2 (numpy.ndarray): Second image.
        alpha (float): Transparency value for blending (0.0 - 1.0).

    Returns:
        numpy.ndarray: Stitched image.
    """
    if img1 is None:
        return img2
    shift = detect_shift(img1, img2)

    if shift is not None:
        x_shift, y_shift = shift
        x_shift = int(x_shift)
        y_shift = int(y_shift)
        if x_shift == 0 and y_shift == 0:
            return img1
        if x_shift >100 or y_shift > 100:
            return img1


        #Paste the first image
        if x_shift==0 or x_shift>0:
            firs_img_x = 0
            second_img_x = x_shift
        else:
            firs_img_x = abs(x_shift)
            second_img_x = 0
        if y_shift==0 or y_shift>0:
            firs_img_y = 0
            second_img_y = y_shift

        else:
            firs_img_y = abs(y_shift)
            second_img_y = 0
        first_rect = (firs_img_y,img1.shape[0] + firs_img_y, firs_img_x,img1.shape[1] + firs_img_x)
        # Create a blank canvas with enough space for both images
        stitched_width = img1.shape[1]+ abs(x_shift)
        stitched_height = img1.shape[0] + abs(y_shift)
        stitched_image = np.zeros((stitched_height, stitched_width), np.uint8)

        try:
            stitched_image[first_rect[0]:first_rect[1], first_rect[2]:first_rect[3]] = img1
        except:
            pass
        # Paste the second image
        second_rect = (second_img_y, img2.shape[0] + second_img_y, second_img_x, img2.shape[1] + second_img_x)
        try:
            stitched_image[second_rect[0]:second_rect[1], second_rect[2]:second_rect[3]] = img2
        except:
            pass


        return stitched_image
    else:
        return None


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask

def pil2pixmap(image):
    # Convert PIL image to RGB format (if not already in that format)
    image = image.convert("RGB")

    # Convert PIL image to raw data
    data = image.tobytes("raw", "RGB")

    # Create QImage from raw data
    qimage = QImage(data, image.width, image.height, QImage.Format_RGB888)

    # Convert QImage to QPixmap
    return QPixmap.fromImage(qimage)

def cv2pixmap(cv_img):
    height, width, channel = cv_img.shape
    bytesPerLine = 3 * width
    qImg = QImage(cv_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg)
