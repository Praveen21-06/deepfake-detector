# preprocess_images.py
import os
import random
from pathlib import Path
from PIL import Image
import numpy as np
import json
from tqdm import tqdm

class ImageDatasetProcessor:
    def __init__(self, raw_data_path, processed_data_path, image_size=224):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.image_size = image_size
        self.create_directories()

    def create_directories(self):
        """Create processed dataset directory structure"""
        splits = ['train', 'val', 'test']
        labels = ['real', 'fake']
        for split in splits:
            for label in labels:
                os.makedirs(f"{self.processed_data_path}/{split}/{label}", exist_ok=True)

    def validate_and_resize_image(self, image_path):
        """Validate and resize image to target size"""
        try:
            img = Image.open(image_path).convert('RGB')
            if img.size[0] < 64 or img.size[1] < 64:
                return None
            img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            arr = np.array(img)
            if np.std(arr) < 10:
                return None
            return img
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def process_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        stats = {
            'total_processed': 0,
            'real_images': 0,
            'fake_images': 0,
            'corrupted_images': 0,
            'splits': {'train': 0, 'val': 0, 'test': 0}
        }
        for label in ['real', 'fake']:
            label_path = os.path.join(self.raw_data_path, label)
            if not os.path.exists(label_path):
                alternatives = {
                    'real': ['original', 'authentic', 'genuine', '0'],
                    'fake': ['fake', 'deepfake', 'manipulated', 'synthetic', '1']
                }
                for alt in alternatives[label]:
                    alt_path = os.path.join(self.raw_data_path, alt)
                    if os.path.exists(alt_path):
                        label_path = alt_path
                        break
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.extend(Path(label_path).glob(f"*{ext}"))
                image_files.extend(Path(label_path).glob(f"*{ext.upper()}"))
            print(f"Found {len(image_files)} {label} images")
            random.shuffle(image_files)
            total = len(image_files)
            train_end = int(total * train_ratio)
            val_end = int(total * (train_ratio + val_ratio))
            splits_data = {
                'train': image_files[:train_end],
                'val': image_files[train_end:val_end],
                'test': image_files[val_end:]
            }
            for split, files in splits_data.items():
                count = 0
                print(f"Processing {split} {label} ({len(files)} images)…")
                for img_path in files:
                    img = self.validate_and_resize_image(img_path)
                    if img is not None:
                        out_dir = f"{self.processed_data_path}/{split}/{label}"
                        out_name = f"{label}_{count:05d}.jpg"
                        img.save(os.path.join(out_dir, out_name), quality=95)
                        count += 1
                        stats['total_processed'] += 1
                        stats['splits'][split] += 1
                        if count % 100 == 0:
                            print(f"  {count} {label} images processed for {split}")
                        if count >= 1000:
                            print(f"Limited to {count} {label} images for {split}")
                            break
                    else:
                        stats['corrupted_images'] += 1
                print(f"Completed {count} {label} images for {split}")
            stats[f'{label}_images'] = len(image_files)
        with open(f"{self.processed_data_path}/dataset_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        print("\n" + "="*50)
        print("DATASET PROCESSING COMPLETED")
        print("="*50)
        print(f"Total processed: {stats['total_processed']}")
        print(f"Real images found: {stats['real_images']}")
        print(f"Fake images found: {stats['fake_images']}")
        print(f"Corrupted/skipped: {stats['corrupted_images']}")
        print(f"Train: {stats['splits']['train']}")
        print(f"Val: {stats['splits']['val']}")
        print(f"Test: {stats['splits']['test']}")
        return stats

def main():
    random.seed(42)
    processor = ImageDatasetProcessor(
        raw_data_path="data/raw",
        processed_data_path="data/processed",
        image_size=224
    )
    processor.process_dataset()

if __name__ == "__main__":
    main()
