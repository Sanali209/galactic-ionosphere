import torch
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import argparse
import os


# --- 1. The Core Prediction Function ---
def predict_image_rating(model, feature_extractor, image_path, device):
    """
    Loads an image, preprocesses it, and predicts a continuous rating using the model.

    Args:
        model: The loaded PyTorch model.
        feature_extractor: The loaded Hugging Face feature extractor.
        image_path (str): The path to the image file.
        device: The torch device ('cuda' or 'cpu').

    Returns:
        float: The predicted rating, or None if an error occurs.
    """
    try:
        # Open the image and ensure it's in RGB format
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error opening or converting image {image_path}: {e}")
        return None

    # Put the model in evaluation mode
    model.eval()

    # No gradients are needed for inference, which saves memory and computation
    with torch.no_grad():
        # Preprocess the image and create a batch (of size 1)
        inputs = feature_extractor(images=image, return_tensors="pt")

        # Move the preprocessed inputs to the correct device
        pixel_values = inputs['pixel_values'].to(device)

        # Get the model's output
        outputs = model(pixel_values=pixel_values)

        # For our regression model, the output is in 'logits'.
        # We use .item() to get the single float value from the tensor.
        predicted_rating = outputs.logits.item()

    return predicted_rating


# --- 2. Main Execution Block ---
if __name__ == "__main__":



    MODEL_IDENTIFIER = "sanali209/rating1_10"
    IMAGE_TO_PREDICT = r"D:\Screenshot_1.png"

    # Check if the image path is valid
    if not os.path.exists(IMAGE_TO_PREDICT):
        print(f"Fatal: The provided image path does not exist: {IMAGE_TO_PREDICT}")
        exit()

    print("--- Image Rating Prediction ---")
    print(f"Model: {MODEL_IDENTIFIER}")
    print(f"Image: {IMAGE_TO_PREDICT}")

    # Set the device (use GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    try:
        # Load the model and the feature extractor from the Hub or a local path
        print("Loading model and feature extractor...")
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_IDENTIFIER)
        model = AutoModelForImageClassification.from_pretrained(MODEL_IDENTIFIER)
        model.to(device)  # Move the model to the selected device
        print("Model loaded successfully.\n")
    except Exception as e:
        print(f"Fatal: Failed to load the model from '{MODEL_IDENTIFIER}'.")
        print(f"Error: {e}")
        print("Please ensure the model_id is a valid Hugging Face repo or a local path containing model files.")
        exit()

    # Call the prediction function
    rating = predict_image_rating(model, feature_extractor, IMAGE_TO_PREDICT, device)

    # Print the result
    if rating is not None:
        print("--- Prediction Result ---")
        # The rating is from 0.0 to 10.0, so formatting to 2 decimal places is nice
        print(f"Predicted Rating: {rating:.2f} / 10.0")