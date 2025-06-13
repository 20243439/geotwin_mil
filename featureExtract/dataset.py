import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class PairedDataset(Dataset):
    def __init__(self, root_dir, map_dir, transform=None):
        self.root_dir = root_dir
        self.map_dir = map_dir
        self.transform = transform
        
        # 위성 이미지 파일 리스트
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]
        
        # 지도 이미지 파일과 매칭되는 위성 이미지만 사용
        self.valid_files = []
        for image_file in self.image_files:
            map_path = os.path.join(map_dir, image_file)
            if os.path.exists(map_path):
                self.valid_files.append(image_file)
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        image_filename = self.valid_files[idx]
        
        # 위성 이미지 로드
        image_path = os.path.join(self.root_dir, image_filename)
        image = Image.open(image_path).convert("RGB")
        
        # 지도 이미지 로드
        map_path = os.path.join(self.map_dir, image_filename)
        map_image = Image.open(map_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            map_image = self.transform(map_image)
        
        return image, map_image

class SingleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.root_dir, image_filename)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, image_filename  # 파일명 반환

def get_data_loader(root_dir, map_dir=None, batch_size=32, transform=None, num_workers=8, pin_memory=True):
    if map_dir:  # Training Mode
        dataset = PairedDataset(root_dir, map_dir, transform=transform)
    else:  # Feature Extraction Mode
        dataset = SingleDataset(root_dir, transform=transform)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if map_dir else False,  # Feature Extract 시 순서 유지
        num_workers=num_workers,
        pin_memory=pin_memory
    )