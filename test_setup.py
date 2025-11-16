# test_setup.py
import torch
import sys
import os

def test_environment():
    print("="*50)
    print("ENVIRONMENT TEST")
    print("="*50)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"CUDA available: {cuda}")
    if cuda:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory/1024**3
        print(f"GPU memory: {mem:.1f} GB")
        try:
            x = torch.randn(50,50).cuda()
            y = torch.randn(50,50).cuda()
            _ = torch.matmul(x,y)
            print("✓ GPU tensor ops OK")
        except Exception as e:
            print(f"⚠ GPU ops failed: {e}")
    print("\nDirectory checks:")
    for d in ["data/raw", "data/processed", "data/processed/train", 
              "data/processed/val", "data/processed/test"]:
        print(f"{d}: {'Exists' if os.path.exists(d) else 'Missing'}")
    print("="*50)

if __name__ == "__main__":
    test_environment()
