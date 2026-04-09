import torch
from torchvision import transforms
from PIL import Image
import yaml
import time
from src.resnet_model import ResNet18Transfer

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

class ImageClassifier:
    def __init__(self, model_path="checkpoints/best_resnet18_clean.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")

        self.model = ResNet18Transfer(freeze_backbone=False)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config['dataset']['mean'], std=config['dataset']['std'])
        ])

        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        print(f"✅ Model loaded! (Accuracy: {checkpoint.get('accuracy', 'N/A')}%)")

    def predict(self, image_path: str):
        start_time = time.time()

        # Load image with PIL
        image = Image.open(image_path).convert('RGB')

        # Debug: print image size
        print(f"Debug: Original image size: {image.size}")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)

        predicted_class = self.classes[predicted_idx.item()]
        confidence_percent = confidence.item() * 100
        latency_ms = (time.time() - start_time) * 1000

        print(f"Prediction: {predicted_class} | Confidence: {confidence_percent:.2f}% | Latency: {latency_ms:.2f}ms")

        return {
            "class": predicted_class,
            "confidence": round(confidence_percent, 2),
            "latency_ms": round(latency_ms, 2)
        }


if __name__ == "__main__":
    classifier = ImageClassifier()
    print("\n✅ predict.py ready for testing.")
