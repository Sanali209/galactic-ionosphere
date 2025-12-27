"""
Model Sigma Configuration Manager

Manages LLM model sigma configuration for TrueSkill rating initialization.
The sigma value represents the uncertainty of the rating and is calculated
from the model's training error percentage.
"""
import json
import os
from typing import Optional
from loguru import logger
import torch.nn as nn
from constants import MODEL_SIGMA


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


class ModelSigmaConfig:
    """Manages LLM model sigma configuration for new item initialization"""

    CONFIG_FILE = "model_sigma_config.json"

    HF_MODEL = "sanali209/rating1_10"  # HuggingFace model ID

    def __init__(self):
        self.calculated_sigma: float = MODEL_SIGMA  # Default from constants
        self.use_llm_sigma: bool = False
        self.model_error_percentage: float = 0.0
        self.use_predictions: bool = False
        self.model = None
        self.feature_extractor = None
        self.device = None
        self.load_config()

        # Auto-load model if predictions are enabled
        if self.use_predictions:
            logger.info("Auto-loading model (predictions enabled)")
            self.load_model_for_inference()

    def load_config(self):
        """Load config from JSON file"""
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    self.calculated_sigma = data.get("calculated_sigma", MODEL_SIGMA)
                    self.use_llm_sigma = data.get("use_llm_sigma", False)
                    self.model_error_percentage = data.get("model_error_percentage", 0.0)
                    self.use_predictions = data.get("use_predictions", False)
                logger.info(
                    f"Loaded model config: sigma={self.calculated_sigma:.4f}, use_sigma={self.use_llm_sigma}, use_predictions={self.use_predictions}")
            except Exception as e:
                logger.error(f"Error loading model sigma config: {e}")

    def load_from_model_metadata(self, model_directory: str) -> bool:
        """
        Load sigma from trained model metadata
        
        Args:
            model_directory: Path to the model directory containing model_metadata.json
            
        Returns:
            True if loaded successfully, False otherwise
        """
        metadata_file = os.path.join(model_directory, "model_metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    self.calculated_sigma = data.get("calculated_sigma", MODEL_SIGMA)
                    self.model_error_percentage = data.get("error_percentage", 0.0)
                    self.model_path = model_directory
                logger.info(f"Loaded sigma from model metadata: {self.calculated_sigma:.4f}")
                logger.info(f"Model error percentage: {self.model_error_percentage:.4f}")
                return True
            except Exception as e:
                logger.error(f"Error loading model metadata: {e}")
                return False
        else:
            logger.warning(f"Model metadata file not found: {metadata_file}")
            return False

    def save_config(self):
        """Save current config to JSON"""
        data = {
            "calculated_sigma": self.calculated_sigma,
            "use_llm_sigma": self.use_llm_sigma,
            "model_error_percentage": self.model_error_percentage,
            "use_predictions": self.use_predictions
        }
        try:
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(
                f"Saved model config: sigma={self.calculated_sigma:.4f}, use_sigma={self.use_llm_sigma}, use_predictions={self.use_predictions}")
        except Exception as e:
            logger.error(f"Error saving model sigma config: {e}")

    def get_sigma_for_new_items(self) -> float:
        """
        Get the sigma value to use for new items
        
        Returns:
            Sigma value - either from model or default
        """
        if self.use_llm_sigma:
            return self.calculated_sigma
        else:
            return MODEL_SIGMA  # Default from constants

    def set_manual_sigma(self, sigma: float):
        """
        Manually set the sigma value
        
        Args:
            sigma: The sigma value to set
        """
        self.calculated_sigma = sigma
        logger.info(f"Manually set sigma to: {sigma:.4f}")

    def load_model_for_inference(self) -> bool:
        """
        Load model and feature extractor from HuggingFace
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForImageClassification

            logger.info(f"Loading model from HuggingFace: {self.HF_MODEL}")

            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            # Load feature extractor and model
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.HF_MODEL)
            base_loaded = AutoModelForImageClassification.from_pretrained(self.HF_MODEL, num_labels=1,
                                                                          ignore_mismatched_sizes=True)
            self.model = BoundedRegressionModel(base_loaded, min_val=1, max_val=10)
            self.model.to(self.device)
            self.model.eval()

            logger.info("Model loaded successfully for inference")
            return True

        except Exception as e:
            logger.error(f"Error loading model for inference: {e}")
            self.model = None
            self.feature_extractor = None
            self.device = None
            return False

    def predict_rating(self, image_path: str) -> Optional[float]:
        """
        Predict rating for an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Predicted rating (1-10 scale) or None on failure
        """
        if self.model is None or self.feature_extractor is None:
            logger.warning("Model not loaded for inference")
            return None

        try:
            import torch
            from PIL import Image

            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")

            # Put model in eval mode
            self.model.eval()

            with torch.no_grad():
                # Preprocess image
                inputs = self.feature_extractor(images=image, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(self.device)

                # Get prediction
                outputs = self.model(pixel_values=pixel_values)
                predicted_rating = outputs.logits.item()

                logger.debug(
                    f"Predicted rating for {image_path}: raw={predicted_rating:.4f}, scaled={predicted_rating:.2f}")
                return predicted_rating

        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            return None
        except Exception as e:
            logger.error(f"Error predicting rating for {image_path}: {e}")
            return None

    def unload_model(self):
        """Unload model from memory"""
        self.model = None
        self.feature_extractor = None
        self.device = None
        logger.info("Model unloaded from memory")

    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.feature_extractor is not None

    def get_config_summary(self) -> str:
        """
        Get a summary of the current configuration
        
        Returns:
            Human-readable configuration summary
        """
        status = "Enabled" if self.use_llm_sigma else "Disabled"
        model_status = "Loaded" if self.is_model_loaded() else "Not loaded"
        pred_status = "Enabled" if self.use_predictions else "Disabled"

        return (f"Model Sigma Config:\n"
                f"  Sigma Status: {status}\n"
                f"  Sigma Value: {self.calculated_sigma:.4f}\n"
                f"  Error %: {self.model_error_percentage:.4f}\n"
                f"  HuggingFace Model: {self.HF_MODEL}\n"
                f"  Model Status: {model_status}\n"
                f"  Predictions: {pred_status}")


# Global instance
model_sigma_config = ModelSigmaConfig()
