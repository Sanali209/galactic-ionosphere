import base64
import xmlrpc.client



def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
promt = "detail describe of image"
base64_image = image_to_base64(r"E:\rawimagedb\repository\safe repo\asorted images\3\37019507112_f2d61af76a_b.jpg")
# Create an XML-RPC client
with xmlrpc.client.ServerProxy("https://9d84-34-142-246-65.ngrok-free.app//RPC2") as proxy:
    res = proxy.caption_image(base64_image,promt)
    print(res)
