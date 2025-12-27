from transformers import OFATokenizer, OFAModel
import torch
from PIL import Image
import numpy as np


class OFAInference:
    def __init__(self, model_name: str = "ofa-base", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Инициализация OFA модели и токенизатора.

        :param model_name: Название модели (например, 'ofa-base' или 'ofa-large').
        :param device: Устройство для выполнения вычислений ('cuda' или 'cpu').
        """
        self.device = device
        self.tokenizer = OFATokenizer.from_pretrained(model_name)
        self.model = OFAModel.from_pretrained(model_name).to(self.device)

    def preprocess_image(self, image_path: str):
        """
        Предобработка изображения для модели.

        :param image_path: Путь к изображению.
        :return: Тензор изображения.
        """
        image = Image.open(image_path).convert("RGB")
        transform = self.model.image_processor
        return transform(image, return_tensors="pt").to(self.device)

    def process_text(self, text: str) -> torch.Tensor:
        """
        Токенизация текста для модели.

        :param text: Входной текст.
        :return: Токены текста.
        """
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def multimodal_inference(self, text: str, image_path: str) -> str:
        """
        Выполнение мультимодального вывода.

        :param text: Текстовая подсказка.
        :param image_path: Путь к изображению.
        :return: Результат вывода модели.
        """
        image = self.preprocess_image(image_path)
        text_tokens = self.process_text(text)
        outputs = self.model.generate(inputs=text_tokens, images=image)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def text_inference(self, text: str) -> str:
        """
        Выполнение вывода на основе текста.

        :param text: Входной текст.
        :return: Результат генерации текста.
        """
        text_tokens = self.process_text(text)
        outputs = self.model.generate(inputs=text_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def image_captioning(self, image_path: str) -> str:
        """
        Генерация описания для изображения.

        :param image_path: Путь к изображению.
        :return: Сгенерированное описание.
        """
        image = self.preprocess_image(image_path)
        inputs = self.process_text("<image> What does this image describe?")
        outputs = self.model.generate(inputs=inputs, images=image)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def question_answering(self, question: str, image_path: str) -> str:
        """
        Ответ на вопросы по изображению.

        :param question: Вопрос.
        :param image_path: Путь к изображению.
        :return: Ответ на вопрос.
        """
        image = self.preprocess_image(image_path)
        inputs = self.process_text(f"<image> {question}")
        outputs = self.model.generate(inputs=inputs, images=image)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def image_classification(self, image_path: str, classes: list) -> str:
        """
        Классификация изображения.

        :param image_path: Путь к изображению.
        :param classes: Список классов для классификации.
        :return: Класс с наивысшей вероятностью.
        """
        image = self.preprocess_image(image_path)
        class_text = " ".join([f'"{cls}"' for cls in classes])
        inputs = self.process_text(f"<image> Which class does this image belong to? {class_text}")
        outputs = self.model.generate(inputs=inputs, images=image)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_image(self, text_prompt: str, output_path: str):
        """
        Генерация изображения на основе текстового описания.

        :param text_prompt: Описание изображения.
        :param output_path: Путь для сохранения сгенерированного изображения.
        :return: Путь к сгенерированному изображению.
        """
        inputs = self.process_text(text_prompt)
        outputs = self.model.generate(inputs=inputs)
        image_array = outputs[0].cpu().detach().numpy()
        image = Image.fromarray((image_array * 255).astype(np.uint8))
        image.save(output_path)
        return output_path

if __name__ == "__main__":
    import os

    # Инициализируем объект класса
    ofa = OFAInference(model_name="ofa-base")

    # Тест текстовой генерации
    def test_text_inference():
        print("Тест: Текстовая генерация")
        result = ofa.text_inference("Translate English to French: Hello, how are you?")
        print(f"Результат: {result}")
        assert "Bonjour" in result, "Ошибка в текстовой генерации"

    # Тест описания изображения
    def test_image_captioning():
        print("\nТест: Описание изображения")
        image_path = r"E:\rawimagedb\repository\safe repo\asorted images\3\37019507112_f2d61af76a_b.jpg"
        assert os.path.exists(image_path), f"Файл {image_path} не найден"
        result = ofa.image_captioning(image_path)
        print(f"Описание изображения: {result}")
        assert len(result) > 0, "Описание изображения пустое"

    # Тест мультимодального вывода
    def test_multimodal_inference():
        print("\nТест: Мультимодальный вывод")
        image_path = r"E:\rawimagedb\repository\safe repo\asorted images\3\37019507112_f2d61af76a_b.jpg"
        assert os.path.exists(image_path), f"Файл {image_path} не найден"
        result = ofa.multimodal_inference("What is in this image?", image_path)
        print(f"Результат мультимодального вывода: {result}")
        assert len(result) > 0, "Мультимодальный вывод пустой"

    # Тест классификации изображения
    def test_image_classification():
        print("\nТест: Классификация изображения")
        image_path = r"E:\rawimagedb\repository\safe repo\asorted images\3\37019507112_f2d61af76a_b.jpg"
        classes = ["dog", "cat", "bird", "car"]
        assert os.path.exists(image_path), f"Файл {image_path} не найден"
        result = ofa.image_classification(image_path, classes)
        print(f"Классификация: {result}")
        assert result in classes, "Классификация вернула некорректный результат"

    # Тест генерации изображения
    def test_generate_image():
        print("\nТест: Генерация изображения")
        text_prompt = "A beautiful sunset over a mountain range"
        output_path = "generated_image.png"
        result_path = ofa.generate_image(text_prompt, output_path)
        print(f"Сгенерированное изображение сохранено по пути: {result_path}")
        assert os.path.exists(result_path), "Сгенерированное изображение не было сохранено"

    # Запуск тестов
    try:
        test_text_inference()
        test_image_captioning()
        test_multimodal_inference()
        test_image_classification()
        test_generate_image()
        print("\nВсе тесты прошли успешно!")
    except AssertionError as e:
        print(f"\nОшибка в тесте: {e}")
