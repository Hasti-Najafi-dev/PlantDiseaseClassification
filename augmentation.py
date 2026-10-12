from pathlib import Path
from torchvision.transforms import  transforms
from PIL import Image
import hashlib
import uuid
class Augmentation:
    def __init__(self, OUT_DIR = 'preProcess/augmentation', size = 256):
        self.OUT_DIR = Path(OUT_DIR)
        self.OUT_DIR.mkdir(parents = True, exist_ok = True)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # ملایم
        ])

    def create_image(self, row):
        image = Image.open(row['path']).convert("RGB")
        new_image = self.transform(image)

        class_dir = self.OUT_DIR / str(row['label'])
        class_dir.mkdir(parents=True, exist_ok=True)

        file_path = class_dir / f"{Path(row['path']).stem}_aug_{uuid.uuid4().hex[:8]}.jpg"
        new_image.save(file_path, quality=95)

        md5 = hashlib.md5(file_path.read_bytes()).hexdigest()
        return {
            'path': str(file_path),
            'label': row['label'],
            'crop': row['crop'],
            'disease': row['disease'],
            'size': (new_image.width, new_image.height),
            'md5': md5,
        }

