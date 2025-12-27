import os
import sys
import torch
import random
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, get_scheduler
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
from huggingface_hub import HfApi, login, create_repo
from torchvision import transforms
from typing import List, Optional

from pydantic import BaseModel, Field

from loguru import logger

# Assuming SLM imports are correct and available in the environment
from SLM.files_data_cache.thumbnail import ImageThumbCache

# --- Pydantic Configuration Models (Unchanged) ---
class GeneralParamsConfig(BaseModel):
    model_name: str = "google/vit-base-patch16-224"
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 5e-5
    save_dir_template: str = "training_output_{model_slug}_regression_v1" # MODIFIED: Changed template name
    test_split_size: float = 0.2
    random_seed: int = 42
    hub_model_id: Optional[str] = None
    num_warmup_steps_ratio: float = 0.1
    log_file: str = "training_runs/training_regression_v1.log" # MODIFIED: Changed log file name
    hf_token: Optional[str] = None


class DataHandlingConfig(BaseModel):
    # MODIFIED: job_name is still relevant, but other fields are less so for regression.
    job_name: str = "rating"
    # MODIFIED: These fields are less relevant for regression but kept for config compatibility.
    excluded_labels: List[str] = []
    no_augment_labels: List[str] = []
    allowed_extensions: List[str] = Field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"])


class AugmentationConfig(BaseModel):
    use_random_resized_crop: bool = True
    random_resized_crop_input_size_aware: bool = True
    random_resized_crop_size: int = 224
    random_resized_crop_scale_min: float = 0.8
    random_resized_crop_scale_max: float = 1.0
    use_random_horizontal_flip: bool = True
    use_color_jitter: bool = True
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.1
    use_random_rotation: bool = True
    random_rotation_degrees: int = 15


class TrainingConfig(BaseModel):
    general_params: GeneralParamsConfig = Field(default_factory=GeneralParamsConfig)
    data_handling: DataHandlingConfig = Field(default_factory=DataHandlingConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)


# --- 2. Кастомный Dataset for Regression ---
# MODIFIED: Renamed and adapted for regression
class ImageRegressionDataset(Dataset):
    def __init__(self, annotation_records, feature_extractor, transform=None):
        self.records = annotation_records
        self.feature_extractor = feature_extractor
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image_path, rating = record[0], record[1]

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            logger.warning(f"Image file not found: {image_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))
        except Exception as e:
            logger.warning(f"Error opening {image_path}: {e}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))

        # MODIFIED: Augmentation is applied to all training images
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                logger.warning(f"Error applying transform to {image_path}: {e}. Using original image.")
                image = Image.open(image_path).convert("RGB")

        try:
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
        except Exception as e:
            logger.warning(f"Preprocessing error for {image_path}: {e}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))

        # MODIFIED: The label is now a float tensor for regression
        return {"pixel_values": pixel_values, "labels": torch.tensor(rating, dtype=torch.float)}


# --- 3. Функция оценки for Regression ---
# MODIFIED: Calculates Loss and Mean Absolute Error (MAE) instead of accuracy.
def evaluate_model_regression(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
        for batch in progress_bar:
            inputs = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(pixel_values=inputs)
            # MODIFIED: Squeeze the output to match the shape of the labels [batch_size]
            predictions = outputs.logits.squeeze(-1)
            loss = criterion(predictions, labels)

            # MODIFIED: Calculate Mean Absolute Error
            mae = torch.abs(predictions - labels).sum().item()

            total_loss += loss.item() * inputs.size(0)
            total_mae += mae
            total_samples += labels.size(0)
            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    avg_mae = total_mae / total_samples if total_samples > 0 else float('inf')
    return avg_loss, avg_mae
class BoundedRegressionModel(nn.Module):
    def __init__(self, base_model, min_val=1.0, max_val=10.0):
        super().__init__()
        self.base = base_model
        self.min_val = min_val
        self.max_val = max_val
        self.sigmoid = nn.Sigmoid()

    def forward(self, pixel_values):
        outputs = self.base(pixel_values=pixel_values)
        raw = outputs.logits
        bounded = self.sigmoid(raw) * (self.max_val - self.min_val) + self.min_val
        outputs.logits = bounded
        return outputs

# --- 4. Функция обучения ---
# MODIFIED: Adapted for regression, logs MAE instead of Accuracy.
def train_model(model, train_loader, val_loader, optimizer, lr_scheduler, criterion, device,
                num_epochs, save_dir, model_name_slug, feature_extractor_instance,
                hub_repo_id=None, config: TrainingConfig = None):
    best_val_loss = float('inf')
    start_epoch = 0
    best_model_path = os.path.join(save_dir, f"best_model_{model_name_slug}")
    checkpoint_path = os.path.join(save_dir, f"latest_checkpoint_{model_name_slug}.pth")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(best_model_path, exist_ok=True)

    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            logger.info(f"Checkpoint loaded. Resuming from epoch {start_epoch + 1}. Best val loss: {best_val_loss:.4f}")
        except Exception as e:
            logger.exception(f"Error loading checkpoint: {e}. Training from scratch.")
            start_epoch = 0
            best_val_loss = float('inf')
    else:
        logger.info(f"Checkpoint not found at {checkpoint_path}. Training from scratch.")

    logger.info(f"Starting training from epoch {start_epoch + 1} up to {num_epochs} epochs.")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        total_samples_epoch = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]", leave=True,
                            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
        for batch in progress_bar:
            inputs = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=inputs)
            predictions = outputs.logits.squeeze(-1) # MODIFIED: Squeeze output
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            running_mae += torch.abs(predictions.detach() - labels).sum().item()
            total_samples_epoch += labels.size(0)
            current_lr = lr_scheduler.get_last_lr()[0]
            progress_bar.set_postfix(loss=loss.item(),
                                     mae=running_mae / total_samples_epoch if total_samples_epoch > 0 else 0, # MODIFIED: Show MAE
                                     lr=f"{current_lr:.2e}")

        epoch_train_loss = running_loss / total_samples_epoch if total_samples_epoch > 0 else float('inf')
        epoch_train_mae = running_mae / total_samples_epoch if total_samples_epoch > 0 else float('inf')
        val_loss, val_mae = evaluate_model_regression(model, val_loader, criterion, device) # MODIFIED: Use regression evaluator

        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}:")
        logger.info(f"  Train Loss: {epoch_train_loss:.4f} | Train MAE: {epoch_train_mae:.4f}") # MODIFIED: Log MAE
        logger.info(f"  Val Loss:   {val_loss:.4f} | Val MAE:   {val_mae:.4f}") # MODIFIED: Log MAE

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"  New best model found! Val Loss: {val_loss:.4f}. Saving to {best_model_path}...")
            model.base.save_pretrained(best_model_path)
            feature_extractor_instance.save_pretrained(best_model_path)
        else:
            logger.info(f"  Validation loss did not improve ({val_loss:.4f} >= {best_val_loss:.4f}).")

        logger.info(f"Saving checkpoint for epoch {epoch + 1} to {checkpoint_path}...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }, checkpoint_path)

    logger.info(f"\nTraining completed. Best Validation Loss: {best_val_loss:.4f}")
    if hub_repo_id and os.path.exists(os.path.join(best_model_path, "config.json")):
        # ... (Upload logic is mostly the same, just changed commit message)
        logger.info(f"\nAttempting to upload best model to repository: {hub_repo_id}")
        try:
            create_repo(hub_repo_id, private=False, exist_ok=True)
            api = HfApi()
            api.upload_folder(
                folder_path=best_model_path,
                repo_id=hub_repo_id,
                repo_type="model",
                commit_message=f"Upload best regression model (val_loss: {best_val_loss:.4f})" # MODIFIED
            )
            logger.info(f"Model successfully uploaded to {hub_repo_id}")
        except Exception as e:
            logger.exception(f"Error uploading model to Hugging Face Hub: {e}")

    return best_model_path


# --- 5. Функция инференса for Regression ---
# MODIFIED: Predicts a rating (float) instead of a class label.
def predict_rating(model, feature_extractor_instance, image_path, device):
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        logger.error(f"Image file not found for inference: {image_path}")
        return None
    except Exception as e:
        logger.exception(f"Error opening image {image_path} for inference: {e}")
        return None

    model.eval()
    model.to(device)

    inputs = feature_extractor_instance(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        # MODIFIED: The output is the predicted rating.
        predicted_rating = outputs.logits.item()

    return predicted_rating


# --- Helper to load config (Unchanged) ---
def load_training_config(config_path="training_config_v5.json") -> TrainingConfig:
    # ... (this function remains the same)
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        config = TrainingConfig(**config_data)
        logger.info(f"Successfully loaded training configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_path} not found. Using default configuration.")
        return TrainingConfig()
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {config_path}. Using default configuration.")
        return TrainingConfig()
    except Exception as e:
        logger.exception(
            f"Error loading or validating configuration from {config_path}: {e}. Using default configuration.")
        return TrainingConfig()

# --- 6. Основной блок ---
if __name__ == "__main__":
    # MODIFIED: Changed default config path to reflect regression task
    cfg = load_training_config("rating_regr.json")

    # ... (Logging and HF login setup are the same)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    log_file_path = cfg.general_params.log_file
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logger.add(log_file_path, rotation="10 MB", level="DEBUG", encoding="utf-8")
    logger.info("Logging initialized.")
    logger.info(f"Using configuration: {cfg.model_dump_json(indent=2)}")
    if cfg.general_params.hf_token:
        try:
            login(token=cfg.general_params.hf_token)
            logger.info("Successfully logged into Hugging Face Hub.")
        except Exception as e:
            logger.warning(f"Error logging into Hugging Face Hub: {e}. Upload to Hub may be unavailable.")
    else:
        logger.info("No Hugging Face token provided in config. Skipping login.")

    # ... (SLM setup is the same)
    from SLM.appGlue.core import Allocator
    from SLM.files_db.annotation_tool.annotation import AnnotationRecord, AnnotationJob
    import torch.multiprocessing as mp

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print("torch.multiprocessing.set_start_method('spawn') уже вызван или не может быть изменен.")

    config = Allocator.config
    config.fileDataManager.path = r"D:\data\ImageDataManager"
    config.mongoConfig.database_name = "files_db"
    Allocator.init_modules()

    # ... (General params setup is the same)
    MODEL_NAME = cfg.general_params.model_name
    NUM_EPOCHS = cfg.general_params.num_epochs
    BATCH_SIZE = cfg.general_params.batch_size
    LEARNING_RATE = cfg.general_params.learning_rate
    MODEL_NAME_SLUG = MODEL_NAME.replace("/", "_")
    SAVE_DIR = cfg.general_params.save_dir_template.format(model_slug=MODEL_NAME_SLUG)
    TEST_SPLIT_SIZE = cfg.general_params.test_split_size
    RANDOM_SEED = cfg.general_params.random_seed
    HUB_MODEL_ID = cfg.general_params.hub_model_id
    NUM_WARMUP_STEPS_RATIO = cfg.general_params.num_warmup_steps_ratio
    ANNOTATION_JOB_NAME = cfg.data_handling.job_name
    ALLOWED_EXTENSIONS = cfg.data_handling.allowed_extensions

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ... (Feature extractor and augmentations setup are the same)
    logger.info(f"Using device: {device}")
    logger.info("\n--- Loading Feature Extractor ---")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    logger.info("\n--- Configuring Data Augmentations ---")
    # ... (augmentation logic is identical)
    train_transforms_list = []
    aug_cfg = cfg.augmentation
    # ... crop size logic ...
    crop_size_to_use = aug_cfg.random_resized_crop_size
    # (The rest of the augmentation setup code is unchanged)
    if aug_cfg.use_random_resized_crop:
        train_transforms_list.append(transforms.RandomResizedCrop(crop_size_to_use, scale=(aug_cfg.random_resized_crop_scale_min, aug_cfg.random_resized_crop_scale_max)))
    if aug_cfg.use_random_horizontal_flip:
        train_transforms_list.append(transforms.RandomHorizontalFlip())
    if aug_cfg.use_color_jitter:
        train_transforms_list.append(transforms.ColorJitter(brightness=aug_cfg.color_jitter_brightness, contrast=aug_cfg.color_jitter_contrast, saturation=aug_cfg.color_jitter_saturation, hue=aug_cfg.color_jitter_hue))
    if aug_cfg.use_random_rotation:
        train_transforms_list.append(transforms.RandomRotation(aug_cfg.random_rotation_degrees))
    data_transforms_train = transforms.Compose(train_transforms_list) if train_transforms_list else None


    logger.info("\n--- Preparing Data for Regression ---")
    try:
        my_job: AnnotationJob = AnnotationJob.get_by_name(ANNOTATION_JOB_NAME)
        if not my_job:
            logger.error(f"Annotation job '{ANNOTATION_JOB_NAME}' not found. Exiting.")
            sys.exit(1)
        # MODIFIED: Assumes the job contains the relevant annotations. We will extract 'avg_rating'.
        all_raw_annotations = AnnotationRecord.find({"parent_id": my_job.id, "manual":True})
    except Exception as e:
        logger.exception(f"Failed to get annotation job '{ANNOTATION_JOB_NAME}'. Error: {e}")
        sys.exit(1)

    # MODIFIED: Data processing loop for regression targets
    initial_records = []
    logger.info("Processing annotations and extracting 'avg_rating'...")

    for fr_ann in tqdm(all_raw_annotations.copy(), desc="Processing raw annotations"):
        same_val = fr_ann.get_field_val("equal_list",[])
        rating = fr_ann.value
        if  len(same_val) > 0:
            for item in same_val:
                item = AnnotationRecord.get_by_id(item)
                item.value = rating
                all_raw_annotations.append(item)

    for fr_ann in tqdm(all_raw_annotations, desc="Processing raw annotations"):
        fr_ann: AnnotationRecord
        if fr_ann.file is None or fr_ann.file.full_path is None:
            continue
        ext = os.path.splitext(fr_ann.file.full_path)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue

        rating = float(fr_ann.value)
        user_voted = fr_ann.get_field_val("manual", None)

        if rating is not None and user_voted is not None:
            try:
                # Ensure the rating is a valid float
                rating_float = max(1.0, min(10.0, rating)) # Ensure rating is within [1, 10]
                thumb_path = ImageThumbCache.instance().get_thumb(fr_ann.file.full_path, "medium")
                initial_records.append((thumb_path, rating_float))
            except (ValueError, TypeError):
                logger.warning(f"Could not convert avg_rating '{rating}' to float for file {fr_ann.file.full_path}. Skipping.")
        else:
            pass
            #logger.warning(f"File {fr_ann.file.full_path} is missing the 'avg_rating' field. Skipping.")

    if not initial_records:
        logger.error("No valid records with 'avg_rating' found. Check your data source.")
        sys.exit(1)

    logger.info(f"Loaded {len(initial_records)} records with valid ratings.")
    ratings = [r[1] for r in initial_records]
    logger.info(f"Min: {min(ratings)}, Max: {max(ratings)}, Mean: {sum(ratings) / len(ratings)}")
    # MODIFIED: Removed all classification-specific logic (label mapping, filtering, etc.)
    # MODIFIED: Simple train-test split without stratification.
    train_annotations, val_annotations = train_test_split(
        initial_records,
        test_size=TEST_SPLIT_SIZE,
        random_state=RANDOM_SEED,
    )

    logger.info(f"\nTraining set size: {len(train_annotations)}")
    logger.info(f"Validation set size: {len(val_annotations)}")

    if not train_annotations or not val_annotations:
        logger.error("One of the datasets is empty after split. Exiting.")
        sys.exit(1)

    logger.info("\n--- Creating Regression Dataset and DataLoader ---")
    # MODIFIED: Use the new ImageRegressionDataset
    train_dataset = ImageRegressionDataset(
        train_annotations, feature_extractor,
        transform=data_transforms_train
    )
    val_dataset = ImageRegressionDataset(
        val_annotations, feature_extractor # No augmentation for validation
    )

    # MODIFIED: Removed WeightedRandomSampler, as it's for classification.
    # Use standard shuffling for the training loader.
    num_workers = 0
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=device.type == 'cuda'
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == 'cuda'
    )
    logger.info("Dataset and DataLoader created.")

    logger.info("\n--- Loading Model for Regression ---")
    try:
        # MODIFIED: Set num_labels=1 for regression.
        # Removed label2id and id2label as they are not needed.
        base_model = AutoModelForImageClassification.from_pretrained(
            MODEL_NAME, num_labels=1,
            ignore_mismatched_sizes=True
        )
        model = BoundedRegressionModel(base_model)
        model.to(device)
        logger.info("Model loaded with a regression head (1 output) and moved to device.")
    except Exception as e:
        logger.exception(f"Error loading model: {e}")
        sys.exit(1)

    logger.info("\n--- Setting up Training ---")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # MODIFIED: Use MSELoss for regression.
    criterion = nn.MSELoss()

    num_training_steps = NUM_EPOCHS * len(train_loader)
    num_warmup_steps = int(NUM_WARMUP_STEPS_RATIO * num_training_steps)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer,
        num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    logger.info(f"Optimizer AdamW and MSELoss criterion set up.")

    logger.info("\n--- Starting Training ---")
    best_model_directory = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, lr_scheduler=lr_scheduler, criterion=criterion, device=device,
        num_epochs=NUM_EPOCHS, save_dir=SAVE_DIR, model_name_slug=MODEL_NAME_SLUG,
        feature_extractor_instance=feature_extractor,
        hub_repo_id=HUB_MODEL_ID, config=cfg
    )
    rating_min, rating_max = 1.0, 10.0
    rating_range = rating_max - rating_min
    # MODIFIED: Inference example adapted for regression
    if os.path.exists(best_model_directory) and os.listdir(best_model_directory):
        logger.info("\n--- Loading best model for inference example ---")
        try:
            base_loaded = AutoModelForImageClassification.from_pretrained(best_model_directory, num_labels=1, ignore_mismatched_sizes=True)
            loaded_model = BoundedRegressionModel(base_loaded, min_val=rating_min, max_val=rating_max)
            loaded_feature_extractor = AutoFeatureExtractor.from_pretrained(best_model_directory)
            loaded_model.to(device)
            logger.info("Best model and feature extractor loaded successfully.")

            if val_annotations:
                example_idx = random.randint(0, len(val_annotations) - 1)
                example_image_path = val_annotations[example_idx][0]
                true_rating = val_annotations[example_idx][1]
                logger.info(f"\n--- Inference example on: {example_image_path} (True rating: {true_rating:.2f}) ---")
                predicted_rating = predict_rating(
                    loaded_model, loaded_feature_extractor, example_image_path, device
                )
                if predicted_rating is not None:
                    logger.info(f"Predicted rating: {predicted_rating:.2f}")
                else:
                    logger.warning("Prediction failed for the example image.")
            else:
                logger.info("No images in validation set for inference example.")
        except Exception as e:
            logger.exception(f"Error during best model loading or inference example: {e}")
    else:
        logger.warning("\nBest model was not saved or directory is empty. Inference example skipped.")

    logger.info("\nScript finished.")
    # Вычисление и вывод процентной ошибки модели на валидации
    rating_min, rating_max = 1.0, 10.0
    rating_range = rating_max - rating_min

    val_loss, val_mae = evaluate_model_regression(
        loaded_model, val_loader, nn.MSELoss(), device
    )
    error_percentage = val_mae / rating_range  # В диапазоне 0-1

    logger.info(f"Средняя абсолютная ошибка (MAE): {val_mae:.4f}")
    logger.info(f"Процент ошибки (0-1): {error_percentage:.4f}")
    
    # Save model metadata including calculated sigma for use in main app
    from datetime import datetime
    min_sigma = 0.1
    calculated_sigma = max(min_sigma,float(val_mae))
    
    model_metadata = {
        "val_mae": float(val_mae),
        "error_percentage": float(error_percentage),
        "calculated_sigma": float(calculated_sigma),
        "rating_range": rating_range,
        "rating_min": rating_min,
        "rating_max": rating_max,
        "trained_date": datetime.now().isoformat(),
        "model_path": best_model_directory,
        "model_name": MODEL_NAME,
        "num_epochs": NUM_EPOCHS
    }
    
    metadata_file = os.path.join(best_model_directory, "model_metadata.json")
    try:
        with open(metadata_file, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        logger.info(f"Model metadata saved to {metadata_file}")
        logger.info(f"Calculated sigma for new items: {calculated_sigma:.4f}")
    except Exception as e:
        logger.error(f"Error saving model metadata: {e}")
