import os
import torch
import random
import hashlib # Added hashlib import
import diskcache # Added diskcache import

from ray.air import ScalingConfig
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
from huggingface_hub import HfApi,  login, create_repo
from loguru import logger # Added loguru import
import ray # Moved import to top level
from ray import tune # Moved import to top level
from ray.tune.schedulers import ASHAScheduler # Moved import to top level
import torch.multiprocessing as mp # Moved import to top level
from functools import partial # Moved import to top level
from ray.tune import RunConfig, CheckpointConfig, Checkpoint  # Moved import to top level
from ray.air import session # Keep session import

import tempfile # Added tempfile import
import shutil # Added shutil import
from SLM.appGlue.core import Allocator # Moved import to top level
from SLM.files_data_cache.thumbnail import ImageThumbCache # Moved import to top level
from SLM.files_db.annotation_tool.annotation import AnnotationRecord, AnnotationJob # Moved import to top level

# --- 2. Кастомный Dataset ---
class ImageClassificationDataset(Dataset):
    def __init__(self, annotation_records, feature_extractor, label2id, cache_dir='./feature_cache'):
        self.records = annotation_records
        self.feature_extractor = feature_extractor
        self.label2id = label2id
        self.cache = diskcache.Cache(cache_dir) # Initialize diskcache
        logger.debug(f"Dataset initialized with {len(self.records)} records. Cache dir: {cache_dir}") # Debug log

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        file_path = record[0]
        label_str = record[1]
        label_id = self.label2id[label_str]

        # Generate cache key based on file path
        cache_key = hashlib.md5(file_path.encode()).hexdigest()

        # Try to get from cache
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            # logger.debug(f"Cache hit for item {idx} (key: {cache_key})")
            # Ensure label is correct tensor type upon retrieval
            cached_data["labels"] = torch.tensor(cached_data["labels"], dtype=torch.long)
            return cached_data

        # logger.debug(f"Cache miss for item {idx} (key: {cache_key}). Processing...")
        # If not in cache, process the image
        try:
            # Динамическая загрузка изображения
            image = Image.open(file_path).convert("RGB")
        except FileNotFoundError:
            logger.error(f"Ошибка: Не найден файл изображения: {file_path}") # Changed print to logger.error
            # Return next item on error to avoid infinite loops if many files are missing
            return self.__getitem__((idx + 1) % len(self))
        except Exception as e:
            logger.error(f"Ошибка при открытии {file_path}: {repr(e)}") # Changed print to logger.error
            return self.__getitem__((idx + 1) % len(self))

        try:
             inputs = self.feature_extractor(images=image, return_tensors="pt")
             # Detach tensor and move to CPU before caching to avoid GPU memory issues in cache
             pixel_values = inputs['pixel_values'].squeeze(0).cpu()
        except Exception as e:
             logger.error(f"Ошибка препроцессинга {file_path}: {repr(e)}") # Changed print to logger.error
             return self.__getitem__((idx + 1) % len(self))

        result = {"pixel_values": pixel_values, "labels": label_id} # Store raw label_id

        # Store the processed data in cache
        try:
            self.cache.set(cache_key, result)
            # logger.debug(f"Stored item {idx} (key: {cache_key}) in cache.")
        except Exception as e:
            logger.warning(f"Failed to cache item {idx} (key: {cache_key}): {repr(e)}")

        # Ensure the returned label is a tensor
        result["labels"] = torch.tensor(label_id, dtype=torch.long)
        return result

# --- 3. Функция оценки ---
def evaluate_model(model, dataloader, criterion, device):
    model.eval() # Переводим модель в режим оценки
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    logger.info("Starting evaluation...") # Added log
    with torch.no_grad(): # Отключаем вычисление градиентов
        for i, batch in enumerate(dataloader): # Итерируем напрямую по dataloader
            # logger.debug(f"Evaluating batch {i}") # Can be verbose
            # Move tensors to the correct device inside the loop
            inputs = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(pixel_values=inputs)
            logits = outputs.logits
            loss = criterion(logits, labels)

            total_loss += loss.item() * inputs.size(0) # Умножаем на размер батча

            preds = torch.argmax(logits, dim=-1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

    if total_samples == 0:
        logger.warning("Evaluation dataloader was empty.") # Added log
        return 0.0, 0.0

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    logger.info(f"Evaluation finished. Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}") # Added log
    return avg_loss, accuracy

# --- 4. Функция обучения (модифицированная для Ray Tune) ---
def train_model_tune(model, train_loader, val_loader, optimizer, criterion, device, num_epochs):
    logger.info(f"Starting train_model_tune for {num_epochs} epochs.") # Added log
    for epoch in range(num_epochs):
        model.train() # Переводим модель в режим обучения
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False) # leave=False для Tune

        for batch_idx, batch in enumerate(progress_bar):
            # logger.debug(f"Epoch {epoch+1}, Batch {batch_idx}") # Can be very verbose
            # Move tensors to the correct device inside the loop
            inputs = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=inputs)
            logits = outputs.logits
            loss = criterion(logits, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Обнаружен NaN/Inf loss на эпохе {epoch+1}, batch {batch_idx}. Прерывание trial.") # Changed print to logger.warning
                session.report({"val_loss": float('inf'), "val_acc": 0.0}) # Use session.report with dict
                return # Завершаем обучение для этого trial

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

            current_acc = correct_predictions / total_samples if total_samples > 0 else 0
            progress_bar.set_postfix(loss=loss.item(), acc=current_acc)

        epoch_train_loss = running_loss / total_samples if total_samples > 0 else 0
        epoch_train_acc = correct_predictions / total_samples if total_samples > 0 else 0
        logger.info(f"Epoch {epoch+1} Training finished. Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f}") # Added log

        # Валидация после каждой эпохи
        logger.info(f"Epoch {epoch+1} starting validation...") # Added log
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        logger.info(f"Epoch {epoch+1} Validation finished. Loss: {val_loss:.4f}, Acc: {val_acc:.4f}") # Added log

        # --- Интеграция с Ray Tune ---
        metrics = {
            "val_loss": val_loss,
            "val_acc": val_acc,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "epoch": epoch + 1,
        }

        # Создаем чекпоинт и передаем его в session.report
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            model_checkpoint_path = os.path.join(temp_checkpoint_dir, "model")
            logger.info(f"Saving checkpoint for epoch {epoch+1} to temporary dir {model_checkpoint_path}") # Added log
            model.save_pretrained(model_checkpoint_path)
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            logger.info(f"Checkpoint created from {temp_checkpoint_dir}") # Added log

            logger.info(f"Reporting metrics and checkpoint for epoch {epoch+1} to Ray AIR Session...") # Added log
            session.report(metrics, checkpoint=checkpoint)
            logger.info(f"Metrics and checkpoint reported.") # Added log

    logger.info("train_model_tune finished all epochs.") # Added log

# --- 5. Функция инференса (остается без изменений) ---
def predict_image(model, feature_extractor, image_path, device, id2label):
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        logger.error(f"Ошибка: Не найден файл для инференса: {image_path}")
        return None, None
    except Exception as e:
        logger.error(f"Ошибка при открытии файла {image_path}: {e}")
        return None, None

    model.eval() # Режим оценки
    model.to(device)

    # Препроцессинг
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits

    # Получаем вероятности и предсказанный класс
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class_id = torch.argmax(probabilities, dim=-1).item()
    predicted_label = id2label[predicted_class_id]
    confidence = probabilities[0, predicted_class_id].item() # Вероятность предсказанного класса

    return predicted_label, confidence

# --- Новая функция-обертка для Ray Tune (принимает ObjectRefs) ---
def train_tune_wrapper(config, train_data_ref, val_data_ref, label2id_ref, id2label_ref, num_labels_ref):
    """
    Обертка для запуска обучения одного trial в Ray Tune.
    Использует предварительно загруженные данные из Ray Object Store.
    """
    logger.info(f"--- Starting Trial with config: {config} ---") # Added logging

    # --- Константы и начальная настройка ---
    MODEL_NAME = "google/vit-base-patch16-224"
    RANDOM_SEED = 42
    NUM_EPOCHS = 5 # Количество эпох для каждого trial
    CACHE_DIR = os.path.join(tempfile.gettempdir(), f"ray_tune_cache_{os.getpid()}") # Unique cache per trial process

    # Установка seed для воспроизводимости внутри trial
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # --- Устройство ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Trial using device: {device}") # Added logging

    # --- Загрузка Feature Extractor ---
    try:
        logger.info("Loading feature extractor...") # Added logging
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        logger.info("Feature extractor loaded.") # Added logging
    except Exception as e:
        logger.error(f"Ошибка при загрузке Feature Extractor в trial: {repr(e)}")
        session.report({"val_acc": 0.0}) # Use session.report with dict
        return

    # --- Получение данных из Ray Object Store ---
    try:
        logger.info("Retrieving data from Ray Object Store...") # Added logging
        train_annotations = ray.get(train_data_ref)
        val_annotations = ray.get(val_data_ref)
        label2id = ray.get(label2id_ref)
        id2label = ray.get(id2label_ref)
        num_labels = ray.get(num_labels_ref)
        logger.info(f"Data retrieved. Train size: {len(train_annotations)}, Val size: {len(val_annotations)}, Num labels: {num_labels}") # Added logging
    except Exception as e:
        logger.error(f"Ошибка при получении данных из Ray Object Store: {repr(e)}")
        session.report({"val_acc": 0.0}) # Use session.report with dict
        return

    # --- Создание Dataset ---
    try:
        logger.info(f"Creating datasets with cache dir: {CACHE_DIR}...") # Added logging
        # Pass the unique cache directory to the dataset
        train_dataset = ImageClassificationDataset(train_annotations, feature_extractor, label2id, cache_dir=CACHE_DIR)
        val_dataset = ImageClassificationDataset(val_annotations, feature_extractor, label2id, cache_dir=CACHE_DIR)
        logger.info("Datasets created.") # Added logging
    except Exception as e:
        logger.error(f"Ошибка при создании Dataset: {repr(e)}")
        session.report({"val_acc": 0.0}) # Use session.report with dict
        return

    # --- Создание DataLoader ---
    batch_size = config["batch_size"]
    # Determine num_workers based on available CPUs, leave some for other processes
    num_workers = max(0, os.cpu_count() // (ray.cluster_resources().get("GPU", 1) if torch.cuda.is_available() else 1) - 2)
    logger.info(f"Creating DataLoaders with batch_size={batch_size}, num_workers={num_workers}...") # Added logging
    try:
        # Use persistent_workers=True if num_workers > 0
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
        logger.info("DataLoaders created.") # Added logging
    except Exception as e:
        logger.error(f"Ошибка при создании DataLoader: {repr(e)}")
        session.report({"val_acc": 0.0}) # Use session.report with dict
        return

    # --- Загрузка Модели ---
    try:
        logger.info("Loading model...") # Added logging
        model = AutoModelForImageClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True
        )
        model.to(device)
        logger.info("Model loaded and moved to device.") # Added logging
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели в trial: {repr(e)}")
        session.report({"val_acc": 0.0}) # Use session.report with dict
        return

    # --- Настройка Оптимизатора и Функции Потерь ---
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    optimizer_type = config["optimizer_type"]
    logger.info(f"Setting up optimizer ({optimizer_type}) with lr={lr}, weight_decay={weight_decay}...") # Added logging
    try:
        if optimizer_type == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            logger.error(f"Неизвестный тип оптимизатора: {optimizer_type}") # Changed print to logger.error
            session.report({"val_acc": 0.0}) # Use session.report with dict
            return
        criterion = nn.CrossEntropyLoss()
        logger.info("Optimizer and criterion set up.") # Added logging
    except Exception as e:
        logger.error(f"Ошибка при настройке оптимизатора/критерия: {repr(e)}")
        session.report({"val_acc": 0.0}) # Use session.report with dict
        return

    # --- Запуск Обучения для trial ---
    logger.info(f"Starting training loop for {NUM_EPOCHS} epochs...") # Added logging
    try:
        train_model_tune(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_epochs=NUM_EPOCHS
        )
        logger.info("Training loop finished successfully for this trial.") # Added logging
    except Exception as e:
        logger.error(f"Исключение во время выполнения train_model_tune: {repr(e)}")
        # Попытка сообщить об ошибке в Tune, если возможно
        try:
            session.report({"val_acc": 0.0}) # Use session.report with dict
        except Exception as report_e:
            logger.error(f"Не удалось сообщить об ошибке в Ray AIR Session: {repr(report_e)}")
        # Не пробрасываем исключение дальше, чтобы Ray мог обработать завершение trial
        return # Завершаем wrapper
    finally:
        # Clean up cache directory for this trial
        logger.info(f"Cleaning up cache directory: {CACHE_DIR}")
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        # Clear dataset cache object if needed (though it should go out of scope)
        del train_dataset.cache
        del val_dataset.cache


# --- 6. Основной блок (модифицированный для Ray Tune) ---
if __name__ == "__main__":
    # --- Константы ---
    TEST_SPLIT_SIZE = 0.2
    RANDOM_SEED = 42
    NUM_EPOCHS_TUNE = 5 # Макс. эпох для каждого trial (для ASHA max_t)
    NUM_SAMPLES = 1     # Reduced samples for debugging
    HUB_MODEL_ID = "sanali209/nsfwfilter" # ID репозитория на Hugging Face Hub
    HF_TOKEN = os.environ.get("HF_TOKEN", None) # Get token from environment variable
    SAVE_DIR_BEST_MODEL = "best_tuned_model" # Директория для сохранения лучшей модели
    MAIN_CACHE_DIR = "./feature_cache_main" # Main cache dir for pre-loading if needed

    # --- Инициализация Allocator (ОДИН РАЗ перед Ray) ---
    try:
        logger.info("Initializing Allocator (main block)...") # Added logging
        config_alloc = Allocator.config
        config_alloc.fileDataManager.path = r"D:\data\ImageDataManager"
        config_alloc.mongoConfig.database_name = "files_db"
        Allocator.init_modules()
        logger.info("Allocator initialized (main block).") # Added logging
    except Exception as e:
        logger.error(f"Fatal error initializing Allocator: {repr(e)}")
        exit(1) # Exit if Allocator fails, as data loading will fail

    # --- Предварительная загрузка и подготовка данных (ОДИН РАЗ) ---
    try:
        logger.info("Pre-loading and preparing data (main block)...")
        my_job: AnnotationJob = AnnotationJob.get_by_name("NSFWFilter")
        all_annotations = my_job.get_all_annotated()
        if not all_annotations:
            raise ValueError("Не удалось загрузить аннотации.")
        logger.info(f"Loaded {len(all_annotations)} annotations.")
        all_annotations = all_annotations[:2200]
        all_labels = sorted(my_job.choices)
        label2id = {label: i for i, label in enumerate(all_labels)}
        id2label = {i: label for label, i in label2id.items()}
        num_labels = len(all_labels)
        logger.info(f"Labels: {all_labels}")

        records = []
        allowed_ext = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        logger.info("Processing records...")
        for fr in tqdm(all_annotations, desc="Checking records"):
            fr: AnnotationRecord
            ext = os.path.splitext(fr.file.full_path)[1]
            if ext.lower() not in allowed_ext:
                continue
            # Use original path for cache key consistency, thumb path for loading
            original_path = fr.file.full_path
            thumb_path = ImageThumbCache.instance().get_thumb(original_path)
            if not os.path.exists(thumb_path):
                logger.warning(f"Thumb not found for {original_path}, skipping record.")
                continue
            label = fr.value
            records.append((thumb_path, label)) # Store thumb path for loading, label
        logger.info(f"Processed {len(records)} valid records.")

        logger.info("Splitting data...")
        train_annotations, val_annotations = train_test_split(
            records,
            test_size=TEST_SPLIT_SIZE,
            random_state=RANDOM_SEED,
            stratify=[rec[1] for rec in records] # Stratify based on labels
        )
        logger.info(f"Data split. Train size: {len(train_annotations)}, Val size: {len(val_annotations)}")

    except Exception as e:
        logger.error(f"Fatal error during data pre-loading: {repr(e)}")
        exit(1)

    # --- Инициализация Ray ---
    if not ray.is_initialized():
        logger.info("Initializing Ray...") # Added logging
        # Указываем _temp_dir, чтобы избежать потенциальных проблем с путями по умолчанию на Windows
        # Use a more robust temp dir path
        ray_temp_dir = os.path.join(tempfile.gettempdir(), "ray_temp")
        os.makedirs(ray_temp_dir, exist_ok=True)
        logger.info(f"Using Ray temp dir: {ray_temp_dir}")
        ray.init(
            ignore_reinit_error=True,
            _temp_dir=ray_temp_dir # Specify explicit temp dir
        )
        logger.info("Ray initialized.") # Added logging

    # --- Помещение данных в Ray Object Store ---
    logger.info("Putting pre-loaded data into Ray Object Store...")
    try:
        train_data_ref = ray.put(train_annotations)
        val_data_ref = ray.put(val_annotations)
        label2id_ref = ray.put(label2id)
        id2label_ref = ray.put(id2label)
        num_labels_ref = ray.put(num_labels)
        logger.info("Data put into Ray Object Store successfully.")
    except Exception as e:
        logger.error(f"Fatal error putting data into Ray Object Store: {repr(e)}")
        ray.shutdown()
        exit(1)

    # --- Пространство поиска гиперпараметров ---
    search_space = {
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([16]), # Reduced max batch size for testing
        "optimizer_type": tune.choice(["AdamW", "Adam"]),
        "weight_decay": tune.uniform(0.0, 0.01)
    }

    # --- Планировщик ASHA ---
    scheduler = ASHAScheduler(
        metric="val_acc",       # Метрика для оптимизации
        mode="max",             # Цель: максимизировать метрику
        max_t=NUM_EPOCHS_TUNE,  # Макс. кол-во эпох (или шагов) на trial
        grace_period=1,         # Мин. кол-во эпох перед возможной остановкой
        reduction_factor=2      # Фактор сокращения кол-ва trials на каждом шаге
    )

    # --- Обертка для передачи данных в train_tune_wrapper ---
    trainable_with_data = partial(
        train_tune_wrapper,
        train_data_ref=train_data_ref,
        val_data_ref=val_data_ref,
        label2id_ref=label2id_ref,
        id2label_ref=id2label_ref,
        num_labels_ref=num_labels_ref
    )

    # --- Запуск Ray Tune ---
    logger.info("\n--- Запуск Ray Tune ---") # Changed print to logger.info
    # Configure computation resources
    # Adjust CPU based on available resources and GPU count
    cpus_per_worker = max(1, os.cpu_count() // (ray.cluster_resources().get("GPU", 1) if torch.cuda.is_available() else 1) - 1)
    gpus_per_worker = 1 if torch.cuda.is_available() else 0
    logger.info(f"Requesting resources per trial: CPU={cpus_per_worker}, GPU={gpus_per_worker}")

    # Define storage path robustly
    storage_path = os.path.abspath("./ray_results")
    logger.info(f"Using storage path: {storage_path}")

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(trainable_with_data),
            resources={"cpu": cpus_per_worker, "gpu": gpus_per_worker} # Указываем ресурсы для каждого trial
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=NUM_SAMPLES, # Количество запусков
            scheduler=scheduler,
        ),
        run_config=RunConfig(
            name="image_classification_tune_cached_v3", # Changed name slightly
            storage_path=storage_path,
            checkpoint_config=CheckpointConfig(
                checkpoint_score_attribute="val_acc",
                checkpoint_score_order="max",
                num_to_keep=1,
            ),
            # Добавляем verbose=1 для более детального вывода от Tune
            verbose=1
        )
    )
    logger.info("Calling tuner.fit()...") # Added logging
    results = None # Initialize results
    try:
        results = tuner.fit()
    except Exception as e:
        logger.error(f"Исключение во время tuner.fit(): {repr(e)}")
        # Попытка получить результаты, если они есть, даже при ошибке
        try:
            results = tuner.get_results()
        except Exception as get_results_e:
            logger.error(f"Не удалось получить результаты после ошибки fit: {repr(get_results_e)}")

    # --- Получение и вывод лучших результатов ---
    logger.info("\n--- Результаты Ray Tune ---") # Changed print to logger.info
    if results:
        try:
            # Check if there are any successful trials before getting best result
            successful_trials = [r for r in results if r.metrics and r.metrics.get("val_acc") is not None]
            if not successful_trials:
                 logger.error("Нет успешных trials для получения лучшего результата.")
                 best_result = None
            else:
                best_result = results.get_best_result(metric="val_acc", mode="max")
        except Exception as e:
            logger.error(f"Ошибка при получении лучшего результата: {repr(e)}")
            best_result = None

        if best_result:
            logger.info(f"Лучший trial завершен с:") # Changed print to logger.info
            logger.info(f"  Validation Accuracy: {best_result.metrics.get('val_acc', 'N/A'):.4f}") # Changed print to logger.info
            logger.info(f"  Validation Loss:     {best_result.metrics.get('val_loss', 'N/A'):.4f}") # Changed print to logger.info
            logger.info(f"  Epoch:               {best_result.metrics.get('epoch', 'N/A')}") # Changed print to logger.info
            logger.info(f"\nЛучшая конфигурация гиперпараметров:") # Changed print to logger.info
            for param, value in best_result.config.items():
                logger.info(f"  {param}: {value}") # Changed print to logger.info

            logger.info(f"\nПуть к логам лучшего trial: {best_result.log_dir}") # Changed print to logger.info
            best_checkpoint = best_result.checkpoint
            if best_checkpoint:
                 # Checkpoint path might be remote or local depending on storage_path
                 logger.info(f"Лучший чекпоинт URI: {best_checkpoint.uri}") # Log URI

                 # --- Сохранение лучшей модели и загрузка на HF ---
                 logger.info(f"\nЗагрузка лучшей модели из чекпоинта: {best_checkpoint.uri}") # Changed print to logger.info
                 try:
                     # Use Checkpoint.to_directory() to get a local path if needed
                     with best_checkpoint.as_directory() as best_checkpoint_dir:
                         logger.info(f"Локальный путь к чекпоинту: {best_checkpoint_dir}")
                         # Assuming the model is saved in a 'model' subdirectory within the checkpoint
                         model_path_in_checkpoint = os.path.join(best_checkpoint_dir, "model")
                         if not os.path.exists(model_path_in_checkpoint):
                              # Fallback if model is directly in checkpoint dir (older Ray versions?)
                              model_path_in_checkpoint = best_checkpoint_dir
                              logger.warning(f"'model' subdirectory not found, trying root: {model_path_in_checkpoint}")

                         if os.path.exists(os.path.join(model_path_in_checkpoint, "config.json")): # Check if it looks like a HF model dir
                             best_model = AutoModelForImageClassification.from_pretrained(model_path_in_checkpoint)
                             # Reload feature extractor to save with the model
                             feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

                             os.makedirs(SAVE_DIR_BEST_MODEL, exist_ok=True)
                             best_model.save_pretrained(SAVE_DIR_BEST_MODEL)
                             feature_extractor.save_pretrained(SAVE_DIR_BEST_MODEL)
                             logger.info(f"Лучшая модель и feature extractor сохранены в: {SAVE_DIR_BEST_MODEL}") # Changed print to logger.info

                             if HUB_MODEL_ID and HF_TOKEN: # Check if token is provided
                                 logger.info(f"\nПопытка загрузить лучшую модель в репозиторий: {HUB_MODEL_ID}") # Changed print to logger.info
                                 try:
                                     login(token=HF_TOKEN)
                                     create_repo(HUB_MODEL_ID, private=False, exist_ok=True)
                                     api = HfApi()
                                     api.upload_folder(
                                         folder_path=SAVE_DIR_BEST_MODEL,
                                         repo_id=HUB_MODEL_ID,
                                         repo_type="model",
                                         commit_message=f"Upload best tuned model (val_acc: {best_result.metrics.get('val_acc', 0):.4f}) from Ray Tune"
                                     )
                                     logger.info(f"Модель успешно загружена в {HUB_MODEL_ID}") # Changed print to logger.info
                                 except Exception as e:
                                     logger.error(f"Ошибка при загрузке модели на Hugging Face Hub: {repr(e)}") # Changed print to logger.error
                                     logger.warning("Убедитесь, что вы вошли в систему (токен HF_TOKEN) и имеете права на запись.") # Changed print to logger.warning
                             elif HUB_MODEL_ID:
                                 logger.warning("HUB_MODEL_ID указан, но HF_TOKEN не найден (проверьте переменную окружения). Пропуск загрузки на HF Hub.")
                             else:
                                 logger.info("HUB_MODEL_ID не указан, пропуск загрузки на HF Hub.")
                         else:
                             logger.error(f"Не удалось найти файлы модели в чекпоинте: {model_path_in_checkpoint}")

                 except Exception as e:
                     logger.error(f"Ошибка при загрузке/сохранении лучшей модели из чекпоинта: {repr(e)}") # Changed print to logger.error

            else:
                logger.warning("Не найден чекпоинт для лучшего trial.") # Changed print to logger.warning
        else:
            logger.error("Не удалось найти лучший результат (get_best_result вернул None или произошла ошибка). Возможно, все trials завершились с ошибкой.") # Changed print to logger.error
            # Log all trial results for debugging
            if results:
                try:
                    all_results_df = results.get_dataframe()
                    logger.info("Все результаты trials:\n" + all_results_df.to_string())
                except Exception as df_err:
                    logger.error(f"Не удалось получить DataFrame результатов: {repr(df_err)}")


    else:
         logger.error("Объект results не был получен (возможно, tuner.fit() вызвал исключение).")

    # --- Завершение Ray ---
    if ray.is_initialized():
        ray.shutdown()
        logger.info("\nRay shutdown.") # Changed print to logger.info
    else:
        logger.info("\nRay не был инициализирован или уже остановлен.")
