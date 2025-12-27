
import sys
import torch
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    import torchvision
    print(f"Torchvision version: {torchvision.__version__}")
    from torchvision import transforms
    print("Torchvision transforms imported successfully")
except ImportError as e:
    print(f"Torchvision import failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
