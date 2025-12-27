import torch
from transformers import UniterModel, UniterTokenizer
from PIL import Image
import torchvision.transforms as transforms

# Загрузка модели и токенизатора
model = UniterModel.from_pretrained("ChenRocks/UNITER-base")
tokenizer = UniterTokenizer.from_pretrained("ChenRocks/UNITER-base")

# Предобработка текста
text = "What is the color of the car?"
inputs = tokenizer(text, return_tensors="pt")

# Предобработка изображения
image = Image.open("test_image.jpg").convert("RGB")
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
image_tensor = transform(image).unsqueeze(0)

# Объединение данных
outputs = model(input_ids=inputs["input_ids"], pixel_values=image_tensor)

# Использование результатов
answer_logits = outputs.logits
print(f"Ответ: {torch.argmax(answer_logits).item()}")
