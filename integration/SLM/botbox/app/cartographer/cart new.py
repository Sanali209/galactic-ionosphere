import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFilter

def find_shift(img1, img2):
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)
    # adjast second image size to first image
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    try:
        (cc, warp_matrix) = cv2.findTransformECC(img1, img2, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
    except:
        return None


def create_circular_mask_with_gradient(mask_size=(200, 200), gradient_width=50):
    mask_width, mask_height = mask_size

    # Create an empty (black) image
    mask = Image.new('L', (mask_width, mask_height), 0)

    # Create a draw object
    draw = ImageDraw.Draw(mask)

    # Calculate the radius and center
    radius = min(mask_width, mask_height) // 2
    center = (mask_width // 2, mask_height // 2)

    # Draw the solid circle in the center
    draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=255)

    # Create the gradient
    gradient = Image.new('L', (gradient_width, 1), color=0xFF)
    for x in range(gradient_width):
        gradient.putpixel((x, 0), int(255 * (1 - x / gradient_width)))

    alpha = Image.new('L', (mask_width, mask_height), 255)
    alpha.paste(gradient.resize((2 * radius, 2 * radius)), (center[0] - radius, center[1] - radius))
    alpha = alpha.filter(ImageFilter.GaussianBlur(10))

    # Combine the mask and gradient
    mask = Image.composite(mask, alpha, mask)

    return mask


def create_image_with_circular_mask(shift, canvas_size, mask_size, gradient_width):
    # Create a blank image with the desired canvas size
    canvas = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

    # Create the circular mask with gradient
    circular_mask = create_circular_mask_with_gradient(mask_size, gradient_width)

    # Calculate the position to paste the mask in the center of the canvas
    top_left = ((canvas_size[0] - mask_size) // 2 + shift[0], (canvas_size[1] - mask_size) // 2 + shift[1])

    # Convert the mask to RGBA
    circular_mask_rgba = Image.merge('RGBA', (circular_mask, circular_mask, circular_mask, circular_mask))

    # Paste the circular mask into the center of the canvas
    canvas.paste(circular_mask_rgba, top_left, circular_mask_rgba)

    return canvas


def create_circular_mask_with_gradient(mask_size, gradient_width=50):
    mask_width, mask_height = mask_size

    # Create an empty (black) image
    mask = Image.new('L', (mask_width, mask_height), 0)

    # Create a draw object
    draw = ImageDraw.Draw(mask)

    # Calculate the radius and center
    radius = min(mask_width, mask_height) // 2
    center = (mask_width // 2, mask_height // 2)

    # Draw the solid circle in the center
    draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=255)

    # Create the gradient
    gradient = Image.new('L', (gradient_width, 1), color=0xFF)
    for x in range(gradient_width):
        gradient.putpixel((x, 0), int(255 * (1 - x / gradient_width)))

    alpha = Image.new('L', (mask_width, mask_height), 255)
    alpha.paste(gradient.resize((2 * radius, 2 * radius)), (center[0] - radius, center[1] - radius))
    alpha = alpha.filter(ImageFilter.GaussianBlur(10))

    # Combine the mask and gradient
    mask = Image.composite(mask, alpha, mask)

    return mask


def calculate_canvas_size(image_size, shift):
    # Calculate the required canvas size to fit both images with the given shift
    canvas_width = max(image_size[0], image_size[0] + abs(shift[0]))
    canvas_height = max(image_size[1], image_size[1] + abs(shift[1]))

    # Calculate the offset to ensure the shifted image fits within the canvas
    offset_x = max(0, shift[0])
    offset_y = max(0, shift[1])

    return (canvas_width, canvas_height), (offset_x, offset_y)


def blend_images(image1, image2, shift):
    # Calculate the canvas size and offsets
    canvas_size, offset = calculate_canvas_size(image1.size, shift)

    # Create a blank canvas
    canvas = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
    canvas2 = Image.new('RGBA', canvas_size, (0, 0, 0, 0))


    # Calculate positions for the images
    pos_image1 = (0,0)
    pos_image2 = ( shift[0], shift[1])

    # Paste the first image onto the canvas
    canvas.paste(image1, pos_image1)

    mask = create_circular_mask_with_gradient(image2.size, 50)
    # Paste the second image onto the canvas with the shift
    canvas.paste(image2, pos_image2, mask)




    return canvas
def main(folder_path):
    images = []

    image1 = Image.open(r"/SLM/botbox/app/cartographer/scr/Screenshot_5.png").convert('RGBA')
    image2 = Image.open(r"/SLM/botbox/app/cartographer/scr/Screenshot_6.png").convert('RGBA')
    shift = (200, 200)

    res = blend_images(image1, image2,  shift)
    cv2_res = cv2.cvtColor(np.array(res), cv2.COLOR_RGBA2BGRA)
    cv2.imshow('Result', cv2_res)
    cv2.waitKey(0)

    # Чтение изображений из папки
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)

    if not images:
        print("Нет изображений для обработки.")
        return

    # Начальное изображение
    result_img = images[0]
    mask = create_circular_mask(images[0].shape[0], images[0].shape[1])

    # Инициализируем координаты смещения
    prev_shift_x, prev_shift_y = 0, 0

    for i in range(1, len(images)):
        img1_gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

        shift = find_shift(img1_gray, img2_gray)

        mask = create_circular_mask(images[i].shape[0], images[i].shape[1])
        result_img, prev_shift_x, prev_shift_y = merge_images(result_img, images[i], shift, mask, prev_shift_x,
                                                              prev_shift_y)

    # Сохранение итогового изображения
    cv2.imwrite('result.jpg', result_img)
    cv2.imshow('Result', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Укажите путь к папке с изображениями
folder_path = r'/SLM/botbox/app/cartographer/scr'
main(folder_path)
