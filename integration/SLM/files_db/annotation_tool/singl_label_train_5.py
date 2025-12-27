import os
import sys  # Added for Loguru
import torch
import random
import json  # Added for config loading
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, get_scheduler
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
from huggingface_hub import HfApi, login, create_repo
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler
from collections import Counter
from typing import List, Dict, Optional, Tuple  # Added for Pydantic

from pydantic import BaseModel, Field, validator  # Added for Pydantic

from loguru import logger  # Added for Loguru

# Assuming SLM imports are correct and available in the environment
from SLM.files_data_cache.thumbnail import ImageThumbCache
from SLM.appGlue.core import Allocator
from SLM.files_db.annotation_tool.annotation import AnnotationRecord, AnnotationJob


# --- Pydantic Configuration Models ---
class GeneralParamsConfig(BaseModel):
    model_name: str = "google/vit-base-patch16-224"
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 5e-5
    save_dir_template: str = "training_output_{model_slug}_v5"
    test_split_size: float = 0.2
    random_seed: int = 42
    hub_model_id: Optional[str] = None
    num_warmup_steps_ratio: float = 0.1
    log_file: str = "training_runs/training_v5.log"
    hf_token: Optional[str] = None


class DataHandlingConfig(BaseModel):
    job_name: str = "rating"
    excluded_labels: List[str] = []
    no_augment_labels: List[str] = []
    allowed_extensions: List[str] = Field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"])


class AugmentationConfig(BaseModel):
    use_random_resized_crop: bool = True
    random_resized_crop_input_size_aware: bool = True
    random_resized_crop_size: int = 224  # Fallback if not input_size_aware or extractor.size is not dict
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


# --- 2. Кастомный Dataset ---
class ImageClassificationDataset(Dataset):
    def __init__(self, annotation_records, feature_extractor, label2id, transform=None, no_augment_labels=None):
        self.records = annotation_records
        self.feature_extractor = feature_extractor
        self.label2id = label2id
        self.transform = transform
        self.no_augment_labels = set(no_augment_labels) if no_augment_labels else set()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image_path, label_str = record[0], record[1]

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            logger.warning(f"Image file not found: {image_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))  # Simple way to get another item
        except Exception as e:
            logger.warning(f"Error opening {image_path}: {e}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform and label_str not in self.no_augment_labels:
            try:
                image = self.transform(image)
            except Exception as e:
                logger.warning(f"Error applying transform to {image_path}: {e}. Using original image.")
                # Fallback to original image if transform fails
                image = Image.open(image_path).convert("RGB")

        try:
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
        except Exception as e:
            logger.warning(f"Preprocessing error for {image_path}: {e}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))

        label_id = self.label2id[label_str]
        return {"pixel_values": pixel_values, "labels": torch.tensor(label_id, dtype=torch.long)}


# --- 3. Функция оценки ---
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
        for batch in progress_bar:
            inputs = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(pixel_values=inputs)
            logits = outputs.logits
            loss = criterion(logits, labels)

            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    return avg_loss, accuracy


# --- 4. Функция обучения ---
def train_model(model, train_loader, val_loader, optimizer, lr_scheduler, criterion, device,
                num_epochs, save_dir, model_name_slug, feature_extractor_instance,
                id2label, label2id, hub_repo_id=None, config: TrainingConfig = None):  # Added config
    best_val_loss = float('inf')  # Changed from acc to loss
    start_epoch = 0
    # Use model_name_slug for directory to avoid issues with special characters
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
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))  # Load best_val_loss
            if checkpoint.get('id2label') != id2label or checkpoint.get('label2id') != label2id:
                logger.warning("Label mappings in checkpoint differ from current. This may cause issues.")
            logger.info(f"Checkpoint loaded. Resuming from epoch {start_epoch + 1}. Best val loss: {best_val_loss:.4f}")
        except Exception as e:
            logger.exception(f"Error loading checkpoint: {e}. Training from scratch.")
            start_epoch = 0
            best_val_loss = float('inf')
    else:
        logger.info(f"Checkpoint not found at {checkpoint_path}. Training from scratch.")

    logger.info(f"Starting training from epoch {start_epoch + 1} up to {num_epochs} epochs.")
    logger.info(f"Best model will be saved to: {best_model_path}")
    if hub_repo_id:
        logger.info(f"Best model will be uploaded to Hugging Face Hub repository: {hub_repo_id}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions_epoch = 0  # Renamed to avoid conflict
        total_samples_epoch = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]", leave=True,
                            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
        for batch in progress_bar:
            inputs = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=inputs)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct_predictions_epoch += (preds == labels).sum().item()
            total_samples_epoch += labels.size(0)
            current_lr = lr_scheduler.get_last_lr()[0]
            progress_bar.set_postfix(loss=loss.item(),
                                     acc=correct_predictions_epoch / total_samples_epoch if total_samples_epoch > 0 else 0,
                                     lr=f"{current_lr:.2e}")

        epoch_train_loss = running_loss / total_samples_epoch if total_samples_epoch > 0 else float('inf')
        epoch_train_acc = correct_predictions_epoch / total_samples_epoch if total_samples_epoch > 0 else 0
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}:")
        logger.info(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        logger.info(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_loss < best_val_loss:  # Changed condition to loss
            best_val_loss = val_loss
            logger.info(f"  New best model found! Val Loss: {val_loss:.4f}. Saving to {best_model_path}...")
            model.save_pretrained(best_model_path)
            feature_extractor_instance.save_pretrained(best_model_path)

        else:
            logger.info(f"  Validation loss did not improve ({val_loss:.4f} >= {best_val_loss:.4f}).")

        logger.info(f"Saving checkpoint for epoch {epoch + 1} to {checkpoint_path}...")
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'best_val_loss': best_val_loss,  # Save best_val_loss
                'id2label': id2label,
                'label2id': label2id
            }, checkpoint_path)
            logger.info("  Checkpoint saved.")
        except Exception as e:
            logger.exception(f"  Error saving checkpoint: {e}")

    if start_epoch < num_epochs:
        logger.info("\nTraining completed.")
        logger.info(f"Best Validation Loss: {best_val_loss:.4f}")
    else:
        logger.info("\nTraining not performed (start_epoch >= num_epochs).")
        logger.info(f"Existing best Validation Loss from checkpoint: {best_val_loss:.4f}")

    logger.info(f"Best model saved in: {best_model_path}")

    if hub_repo_id and os.path.exists(
            os.path.join(best_model_path, "config.json")):  # HF model config, not training_config
        logger.info(f"\nAttempting to upload best model to repository: {hub_repo_id}")
        try:
            create_repo(hub_repo_id, private=False, exist_ok=True)
            api = HfApi()
            api.upload_folder(
                folder_path=best_model_path,
                repo_id=hub_repo_id,
                repo_type="model",
                commit_message=f"Upload best model after training (val_loss: {best_val_loss:.4f})"  # Changed to loss
            )
            logger.info(f"Model successfully uploaded to {hub_repo_id}")
        except Exception as e:
            logger.exception(f"Error uploading model to Hugging Face Hub: {e}")
            logger.warning("Ensure you are logged in and have write permissions.")
    elif hub_repo_id:
        logger.warning(f"\nSkipping upload to Hugging Face Hub: best model was not saved or config.json missing.")

    return best_model_path


# --- 5. Функция инференса ---
def predict_image(model, feature_extractor_instance, image_path, device, id2label):
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        logger.error(f"Image file not found for inference: {image_path}")
        return None, None
    except Exception as e:
        logger.exception(f"Error opening image {image_path} for inference: {e}")
        return None, None

    model.eval()
    model.to(device)

    inputs = feature_extractor_instance(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits

    probabilities = torch.softmax(logits, dim=-1)
    predicted_class_id = torch.argmax(probabilities, dim=-1).item()
    predicted_label = id2label[predicted_class_id]
    confidence = probabilities[0, predicted_class_id].item()

    return predicted_label, confidence


# --- Helper to load config ---
def load_training_config(config_path="training_config_v5.json") -> TrainingConfig:
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        config = TrainingConfig(**config_data)
        logger.info(f"Successfully loaded training configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_path} not found. Using default configuration.")
        return TrainingConfig()  # Return default config
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {config_path}. Using default configuration.")
        return TrainingConfig()
    except Exception as e:  # Pydantic validation errors etc.
        logger.exception(
            f"Error loading or validating configuration from {config_path}: {e}. Using default configuration.")
        return TrainingConfig()


# --- 6. Основной блок ---
if __name__ == "__main__":
    # Load configuration
    cfg = load_training_config("nsfwfilter.json")  # Default path

    # Setup Loguru
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    log_file_path = cfg.general_params.log_file
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # Ensure log directory exists
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


    MODEL_NAME = cfg.general_params.model_name
    NUM_EPOCHS = cfg.general_params.num_epochs
    BATCH_SIZE = cfg.general_params.batch_size
    LEARNING_RATE = cfg.general_params.learning_rate
    # Create a slug from model name for directory naming (replace / with _)
    MODEL_NAME_SLUG = MODEL_NAME.replace("/", "_")
    SAVE_DIR = cfg.general_params.save_dir_template.format(model_slug=MODEL_NAME_SLUG)
    TEST_SPLIT_SIZE = cfg.general_params.test_split_size
    RANDOM_SEED = cfg.general_params.random_seed
    HUB_MODEL_ID = cfg.general_params.hub_model_id
    NUM_WARMUP_STEPS_RATIO = cfg.general_params.num_warmup_steps_ratio

    EXCLUDED_LABELS = cfg.data_handling.excluded_labels
    NO_AUGMENT_LABELS = cfg.data_handling.no_augment_labels
    ANNOTATION_JOB_NAME = cfg.data_handling.job_name
    ALLOWED_EXTENSIONS = cfg.data_handling.allowed_extensions

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("\n--- Loading Feature Extractor ---")
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        logger.info("Feature Extractor loaded successfully.")
    except Exception as e:
        logger.exception(f"Error loading Feature Extractor: {e}")
        sys.exit(1)

    # --- Define data augmentations based on config ---
    logger.info("\n--- Configuring Data Augmentations ---")
    train_transforms_list = []
    aug_cfg = cfg.augmentation

    # Determine crop size
    crop_size_to_use = aug_cfg.random_resized_crop_size
    if aug_cfg.random_resized_crop_input_size_aware:
        try:
            # HuggingFace feature extractors store size differently
            # Common patterns: feature_extractor.size (int), feature_extractor.size['height'] (dict)
            # For ViT, it's often feature_extractor.size which is an int for square images,
            # or a dict like {'height': 224, 'width': 224} or {'shortest_edge': 224}
            if isinstance(feature_extractor.size, dict):
                # Try to get 'height' or 'shortest_edge'
                if 'height' in feature_extractor.size:
                    crop_size_to_use = feature_extractor.size['height']
                elif 'shortest_edge' in feature_extractor.size:
                    crop_size_to_use = feature_extractor.size['shortest_edge']
                else:  # Fallback if dict keys are unexpected
                    logger.warning(
                        f"Feature extractor size is a dict but 'height' or 'shortest_edge' not found: {feature_extractor.size}. Using configured crop_size: {aug_cfg.random_resized_crop_size}")
                    crop_size_to_use = aug_cfg.random_resized_crop_size
            elif isinstance(feature_extractor.size, int):  # If size is just an int
                crop_size_to_use = feature_extractor.size
            else:
                logger.warning(
                    f"Unexpected feature_extractor.size format: {type(feature_extractor.size)}. Using configured crop_size: {aug_cfg.random_resized_crop_size}")
                crop_size_to_use = aug_cfg.random_resized_crop_size
            logger.info(f"Using crop size for RandomResizedCrop: {crop_size_to_use}")
        except AttributeError:
            logger.warning(
                f"Feature extractor does not have a 'size' attribute. Using configured crop_size: {aug_cfg.random_resized_crop_size}")
            crop_size_to_use = aug_cfg.random_resized_crop_size
        except Exception as e:
            logger.warning(
                f"Error determining crop size from feature extractor: {e}. Using configured crop_size: {aug_cfg.random_resized_crop_size}")
            crop_size_to_use = aug_cfg.random_resized_crop_size

    if aug_cfg.use_random_resized_crop:
        train_transforms_list.append(transforms.RandomResizedCrop(
            crop_size_to_use,
            scale=(aug_cfg.random_resized_crop_scale_min, aug_cfg.random_resized_crop_scale_max)
        ))
        logger.info(
            f"Enabled RandomResizedCrop: size={crop_size_to_use}, scale=({aug_cfg.random_resized_crop_scale_min}, {aug_cfg.random_resized_crop_scale_max})")
    if aug_cfg.use_random_horizontal_flip:
        train_transforms_list.append(transforms.RandomHorizontalFlip())
        logger.info("Enabled RandomHorizontalFlip")
    if aug_cfg.use_color_jitter:
        train_transforms_list.append(transforms.ColorJitter(
            brightness=aug_cfg.color_jitter_brightness,
            contrast=aug_cfg.color_jitter_contrast,
            saturation=aug_cfg.color_jitter_saturation,
            hue=aug_cfg.color_jitter_hue
        ))
        logger.info(
            f"Enabled ColorJitter: brightness={aug_cfg.color_jitter_brightness}, contrast={aug_cfg.color_jitter_contrast}, saturation={aug_cfg.color_jitter_saturation}, hue={aug_cfg.color_jitter_hue}")
    if aug_cfg.use_random_rotation:
        train_transforms_list.append(transforms.RandomRotation(aug_cfg.random_rotation_degrees))
        logger.info(f"Enabled RandomRotation: degrees={aug_cfg.random_rotation_degrees}")

    if not train_transforms_list:
        logger.info("No augmentations enabled for training dataset.")
        data_transforms_train = None
    else:
        data_transforms_train = transforms.Compose(train_transforms_list)

    logger.info("\n--- Preparing Data ---")
    try:
        my_job: AnnotationJob = AnnotationJob.get_by_name(ANNOTATION_JOB_NAME)
        if not my_job:
            logger.error(f"Annotation job '{ANNOTATION_JOB_NAME}' not found. Exiting.")
            sys.exit(1)
        all_raw_annotations = my_job.get_all_annotated()
    except Exception as e:  # Catch issues with SLM/Mongo connection if Allocator not set up
        logger.exception(
            f"Failed to get annotation job '{ANNOTATION_JOB_NAME}' or its annotations. Ensure SLM Allocator is correctly configured. Error: {e}")
        sys.exit(1)

    if not all_raw_annotations:
        logger.error("Failed to load annotations or no annotations found. Exiting.")
        sys.exit(1)

    initial_records = []
    logger.info("Processing annotations and checking paths...")
    for fr_ann in tqdm(all_raw_annotations, desc="Processing raw annotations",
                       bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
        fr_ann: AnnotationRecord
        if fr_ann.file is None or fr_ann.file.full_path is None:
            logger.warning(f"No file path for annotation '{fr_ann.value}' (ID: {fr_ann._id}). Skipping.")
            continue
        ext = os.path.splitext(fr_ann.file.full_path)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            logger.warning(f"Unsupported file format: {fr_ann.file.full_path}. Skipping.")
            continue
        thumb_path = ImageThumbCache.instance().get_thumb(fr_ann.file.full_path, "medium")
        initial_records.append((thumb_path, fr_ann.value))

    if not initial_records:
        logger.error("No valid records after initial processing. Check data and allowed extensions.")
        sys.exit(1)

    logger.info("\n--- Label distribution before filtering ---")
    label_counts_before_filter = Counter(rec[1] for rec in initial_records)
    for label, count in label_counts_before_filter.items():
        logger.info(f"  Label '{label}': {count} records")

    if EXCLUDED_LABELS:
        logger.info(f"\nExcluding labels: {EXCLUDED_LABELS}")
        filtered_records = [rec for rec in initial_records if rec[1] not in EXCLUDED_LABELS]
        logger.info(f"  Records remaining after exclusion: {len(filtered_records)}")
    else:
        filtered_records = initial_records

    if not filtered_records:
        logger.error("No records after label filtering. Check EXCLUDED_LABELS.")
        sys.exit(1)

    current_labels_in_data = sorted(list(set(rec[1] for rec in filtered_records)))
    if not current_labels_in_data:
        logger.error("No unique labels after filtering. Cannot continue.")
        sys.exit(1)

    label2id = {label: i for i, label in enumerate(current_labels_in_data)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(current_labels_in_data)

    logger.info(f"\nFound {num_labels} active unique labels: {current_labels_in_data}")
    logger.info(f"New Label to ID mapping: {label2id}")
    logger.info("\n--- Label distribution after filtering ---")
    label_counts_after_filter = Counter(rec[1] for rec in filtered_records)
    for label, count in label_counts_after_filter.items():
        logger.info(f"  Label '{label}': {count} records")

    try:
        # Stratify only if there's more than one sample per class for all classes in the split
        can_stratify = num_labels > 1 and len(filtered_records) > 1 and \
                       all(count > 1 for count in Counter(rec[1] for rec in filtered_records).values())

        train_annotations, val_annotations = train_test_split(
            filtered_records,
            test_size=TEST_SPLIT_SIZE,
            random_state=RANDOM_SEED,
            stratify=[rec[1] for rec in filtered_records] if can_stratify else None
        )
        if not can_stratify:
            logger.warning("Could not stratify (e.g. some classes have only 1 sample). Using random split.")

    except ValueError as e:
        logger.warning(f"Error during stratified split: {e}. Attempting regular split.")
        train_annotations, val_annotations = train_test_split(
            filtered_records, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED
        )

    logger.info(f"\nTraining set size: {len(train_annotations)}")
    logger.info(f"Validation set size: {len(val_annotations)}")

    if not train_annotations or not val_annotations:
        logger.error("One of the datasets is empty after split. Exiting.")
        sys.exit(1)

    logger.info("\n--- Creating Dataset and DataLoader ---")
    train_dataset = ImageClassificationDataset(
        train_annotations, feature_extractor, label2id,
        transform=data_transforms_train,
        no_augment_labels=set(NO_AUGMENT_LABELS)
    )
    val_dataset = ImageClassificationDataset(
        val_annotations, feature_extractor, label2id  # No augmentation for validation
    )

    logger.info("\n--- Setting up WeightedRandomSampler for class balancing ---")
    train_labels_for_sampler = [rec[1] for rec in train_annotations]
    class_counts_sampler = Counter(train_labels_for_sampler)
    logger.info("Class counts in training set for sampler:")
    for label, count in class_counts_sampler.items():
        logger.info(f"  Label '{label}': {count}")

    sampler = None
    if not class_counts_sampler:
        logger.warning("No classes in training set for WeightedRandomSampler.")
    elif len(class_counts_sampler) == 1:
        logger.info("Only one class in training set. WeightedRandomSampler not needed.")
    else:
        class_weights = {label: 1.0 / count for label, count in class_counts_sampler.items()}
        sample_weights = [class_weights[label] for label in train_labels_for_sampler]
        try:
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            logger.info("WeightedRandomSampler created.")
        except Exception as e:
            logger.exception(f"Error creating WeightedRandomSampler: {e}. Proceeding without sampler.")
            sampler = None

    num_workers = 0  # Keep as 0 for simplicity, can be configured
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=num_workers, pin_memory=device.type == 'cuda',
        shuffle=sampler is None  # Shuffle only if sampler is not used
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == 'cuda'
    )
    logger.info("Dataset and DataLoader created.")

    logger.info("\n--- Loading Model ---")
    try:
        model = AutoModelForImageClassification.from_pretrained(
            MODEL_NAME, num_labels=num_labels, label2id=label2id, id2label=id2label,
            ignore_mismatched_sizes=True  # Useful if fine-tuning a model with a different head
        )
        model.to(device)
        logger.info("Model loaded successfully and moved to device.")
    except Exception as e:
        logger.exception(f"Error loading model: {e}")
        sys.exit(1)

    logger.info("\n--- Setting up Training ---")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    num_training_steps = NUM_EPOCHS * len(train_loader)
    num_warmup_steps = int(NUM_WARMUP_STEPS_RATIO * num_training_steps)

    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer,
        num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    logger.info(f"Optimizer AdamW and CrossEntropyLoss criterion set up.")
    logger.info(f"LR Scheduler: linear, Warmup steps: {num_warmup_steps}, Total training steps: {num_training_steps}")

    logger.info("\n--- Starting Training ---")
    best_model_directory = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, lr_scheduler=lr_scheduler, criterion=criterion, device=device,
        num_epochs=NUM_EPOCHS, save_dir=SAVE_DIR, model_name_slug=MODEL_NAME_SLUG,
        feature_extractor_instance=feature_extractor,
        id2label=id2label, label2id=label2id, hub_repo_id=HUB_MODEL_ID, config=cfg
    )

    if os.path.exists(best_model_directory) and os.listdir(best_model_directory):
        logger.info("\n--- Loading best model for inference example ---")
        try:
            loaded_model = AutoModelForImageClassification.from_pretrained(best_model_directory)
            loaded_feature_extractor = AutoFeatureExtractor.from_pretrained(best_model_directory)
            # Potentially load training_config.json here if needed for inference context
            # with open(os.path.join(best_model_directory, "training_config.json"), 'r') as f:
            #     loaded_training_cfg = json.load(f)
            # logger.info(f"Loaded training config from best model dir: {loaded_training_cfg}")

            loaded_model.to(device)
            logger.info("Best model and feature extractor loaded successfully.")

            if val_annotations:
                # Select a random image for more varied examples if run multiple times
                example_idx = random.randint(0, len(val_annotations) - 1)
                example_image_path = val_annotations[example_idx][0]
                example_true_label = val_annotations[example_idx][1]
                logger.info(f"\n--- Inference example on: {example_image_path} (True label: {example_true_label}) ---")
                predicted_label, confidence = predict_image(
                    loaded_model, loaded_feature_extractor, example_image_path, device, id2label
                )
                if predicted_label is not None:
                    logger.info(f"Predicted label: {predicted_label}")
                    logger.info(f"Confidence: {confidence:.4f}")
                else:
                    logger.warning("Prediction failed for the example image.")
            else:
                logger.info("No images in validation set for inference example.")
        except Exception as e:
            logger.exception(f"Error during best model loading or inference example: {e}")
    else:
        logger.warning("\nBest model was not saved or directory is empty. Inference example skipped.")

    logger.info("\nScript finished.")
