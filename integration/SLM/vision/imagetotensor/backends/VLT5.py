from transformers import VLT5Processor, VLT5ForConditionalGeneration
from PIL import Image

# Инициализация модели и процессора
processor = VLT5Processor.from_pretrained("VLT5/vlt5-small")
model = VLT5ForConditionalGeneration.from_pretrained("VLT5/vlt5-small")

# Загрузка изображения и вопроса
image = Image.open("test_image.jpg").convert("RGB")
question = "What is the object in the image?"

# Преобразование данных
inputs = processor(images=image, text=question, return_tensors="pt", padding=True)

# Генерация ответа
outputs = model.generate(**inputs)
answer = processor.decode(outputs[0], skip_special_tokens=True)
print(f"Ответ: {answer}")