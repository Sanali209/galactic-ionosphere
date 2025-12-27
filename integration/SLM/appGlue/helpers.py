import zlib
from io import BytesIO
import base64
from PIL import Image, ImageFile



ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_colab_xmlrpc_ngrock_url():
    file_path = "G:\Мой диск/ngrok_url.txt"

    try:
        with open(file_path, "r") as f:
            ngrok_url = f.read().strip()
        print(f"ngrok URL: {ngrok_url}")
    except FileNotFoundError:
        print("File not found. Make sure Google Drive is synced and the file exists.")
        return None
    # string like NgrokTunnel: "https://d7e8-34-133-90-31.ngrok-free.app" -> "http://localhost:8000"
    # need to do like "https://d7e8-34-133-90-31.ngrok-free.app/RPC2"
    ngrok_url = ngrok_url.replace('NgrokTunnel: "', "")
    ngrok_url = ngrok_url.replace('" -> "http://localhost:8000"', "/RPC2")
    return ngrok_url


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def base64_to_image(base64_str):
    import io
    image = Image.open(io.BytesIO(base64.b64decode(base64_str)))
    return image


def image_to_zipped_base64(image_path):
    from SLM.files_data_cache.pool import PILPool
    image = PILPool.get_pil_image(image_path)
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes = image_bytes.getvalue()
    compressed_image = zlib.compress(image_bytes)
    compressed_image_base64 = base64.b64encode(compressed_image).decode('utf-8')
    return compressed_image_base64


class ImageHelper:
    @staticmethod
    def create_thumbnail_pil(path, width, height) -> Image:
        from SLM.files_data_cache.pool import PILPool
        image = PILPool.get_pil_image(path)
        image.thumbnail((width, height))
        return image

    @staticmethod
    def content_md5(img: Image):
        import hashlib
        md5 = str(hashlib.md5(img.tobytes()).hexdigest())
        return md5

    @staticmethod
    def image_load_pil(path):
        from SLM.files_data_cache.pool import PILPool
        """Load image from path using PIL save it to memory and close the file"""
        img = PILPool.get_pil_image(path).copy()
        return img

    @staticmethod
    def image_pil_convert_to_jpg_format(img: Image):
        if img.format != "JPEG":
            img = img.convert("RGB")
        return img

    @staticmethod
    def save_image_pil(thumbnail, thumbpath):
        thumbnail.save(thumbpath)
