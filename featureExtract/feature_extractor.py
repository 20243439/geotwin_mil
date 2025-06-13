import os
import torch
import pickle
from tqdm import tqdm
from collections import defaultdict
from torchvision import transforms
from dataset import get_data_loader
from moco.builder import MoCo_ResNet
from torchvision import models


def load_pretrained_model_with_lora(pretrained_path, rank=16):
    model = MoCo_ResNet(base_encoder=models.resnet50, dim=256, mlp_dim=4096, T=0.2)
    state_dict = torch.load(pretrained_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    
    return model


class MocoTrainer:
    def __init__(self, root_dir, map_dir, device, batch_size=64, epochs=10, lr=1e-4, rank=4):
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            print("Running on CPU")

        # Dataloader 설정
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.data_loader = get_data_loader(root_dir, map_dir, batch_size, transform, num_workers=8, pin_memory=True)

        # Load model with LoRA
        pretrained_path = "./featureExtract/r-50-1000ep.pth.tar"
        self.model = load_pretrained_model_with_lora(pretrained_path, rank=rank).to(device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

    def train(self):
        self.model.train()
        accumulation_steps = 16  # Gradient accumulation steps
        for epoch in range(self.epochs):
            running_loss = 0.0
            self.optimizer.zero_grad()

            with tqdm(total=len(self.data_loader), desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch") as pbar:
                for i, (image, map_image) in enumerate(self.data_loader):
                    image, map_image = image.to(self.device), map_image.to(self.device)
                    loss = self.model(image, map_image, m=0.999)
                    
                    # gradient accumulation
                    loss = loss / accumulation_steps
                    loss.backward()

                    running_loss += loss.item() * accumulation_steps

                    # accumulation_steps마다 optimizer update
                    if (i + 1) % accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    pbar.set_postfix(loss=loss.item() * accumulation_steps)
                    pbar.update(1)

                # 마지막에 남은 배치가 accumulation_steps에 미달하는 casedhvjs
                if (i + 1) % accumulation_steps != 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            avg_loss = running_loss / len(self.data_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}], Avg Loss: {avg_loss:.4f}")

    def save_model(self, path="moco_lora_model.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


class MocoFeatureExtractor:
    def __init__(self, model_path, root_dir, device, batch_size=64, output_dir=None):
        self.device = device
        self.batch_size = batch_size
        self.model = load_pretrained_model_with_lora(model_path).to(device)

        # Feature output directory
        self.output_dir = output_dir if output_dir else "./image_geotwin_feature_1024/"
        os.makedirs(self.output_dir, exist_ok=True)

        # Dataloader
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.data_loader = get_data_loader(root_dir, None, batch_size, transform, num_workers=4, pin_memory=False)

    def extract_features(self):
        """
        For each slide, gather all patch features into a single tensor of shape [N, 256].
        """
        self.model.eval()
        slide_features = defaultdict(list)  # {slide_id: [feature_tensor, feature_tensor, ...]}

        with torch.no_grad():
            for images, filenames in tqdm(self.data_loader, desc="Extracting Features"):
                images = images.to(self.device)
                feature_vectors = self.model.base_encoder(images)  # shape: [batch_size, 256, ...]

                # Global average pool or flatten if needed (ResNet output can be 1x1 or similar)
                # If your base_encoder outputs a 1D 256-vector per image, skip the pooling:
                # feature_vectors = torch.flatten(feature_vectors, start_dim=1)
                # Otherwise, adapt as appropriate.

                # Move features to CPU
                feature_vectors = feature_vectors.cpu()

                for filename, feature in zip(filenames, feature_vectors):
                    slide_id = self.get_slide_id(filename)
                    slide_features[slide_id].append(feature)

        return slide_features

    def save_features(self):
        """
        Saves a single .pth file per slide, each containing a [N, 256] tensor of patch features.
        """
        slide_features = self.extract_features()

        for slide_id, feature_list in slide_features.items():
            # Stack list of tensors into one tensor of shape [N, 256]
            slide_tensor = torch.stack(feature_list, dim=0)  # shape: [N, 256, ...] after optional flattening
            slide_path = os.path.join(self.output_dir, f"{slide_id}.pth")

            # Save the stacked tensor
            torch.save(slide_tensor, slide_path)

            print(f"Features saved for slide '{slide_id}' → {slide_path} (shape: {slide_tensor.shape})")

    @staticmethod
    def get_slide_id(filename):
        """
        Extract the slide ID from the filename.
        Adjust this logic to match your filename conventions.
        """
        # Example: slide1_patch001.jpg → slide1
        return filename.split('_')[0]

