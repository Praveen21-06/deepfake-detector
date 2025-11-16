# validate_dataset.py
import os

def validate_processed_dataset(processed_data_path):
    results = {'splits': {}, 'total_images': 0, 'balance_check': {}}
    splits = ['train', 'val', 'test']
    labels = ['real', 'fake']
    print("Validating processed dataset…")
    for split in splits:
        results['splits'][split] = {}
        for label in labels:
            path = os.path.join(processed_data_path, split, label)
            if os.path.exists(path):
                files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
                count = len(files)
                results['splits'][split][label] = count
                results['total_images'] += count
                print(f"{split}/{label}: {count} images")
    for split in splits:
        real = results['splits'][split].get('real', 0)
        fake = results['splits'][split].get('fake', 0)
        total = real + fake
        if total:
            rp = (real/total)*100
            fp = (fake/total)*100
            results['balance_check'][split] = {
                'real%': rp, 'fake%': fp, 'balanced': abs(rp-fp)<20
            }
    print("\n" + "="*60)
    print("DATASET VALIDATION REPORT")
    print("="*60)
    print(f"Total images: {results['total_images']}\n")
    for split,data in results['splits'].items():
        real, fake = data.get('real',0), data.get('fake',0)
        print(f"{split.capitalize()}: {real+fake} (Real: {real}, Fake: {fake})")
    print("\nBalance check:")
    for split, bal in results['balance_check'].items():
        status = "✓ BALANCED" if bal['balanced'] else "⚠ IMBALANCED"
        print(f"{split.capitalize()}: Real {bal['real%']:.1f}% | Fake {bal['fake%']:.1f}% | {status}")
    print("="*60)

if __name__ == "__main__":
    validate_processed_dataset("data/processed")
