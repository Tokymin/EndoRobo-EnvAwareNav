"""
直接从URL下载模型（不通过huggingface_hub）
"""
import os
import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """下载文件"""
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_depth_anything_v2_direct(model_size='vits'):
    """
    直接下载Depth Anything V2模型
    
    Args:
        model_size: 'vits' (Small - 24.8M), 'vitb' (Base - 97.5M), 'vitl' (Large - 335.3M)
    """
    model_urls = {
        'vits': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth',
        'vitb': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth',
        'vitl': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth'
    }
    
    model_file_map = {
        'vits': 'depth_anything_v2_vits.pth',
        'vitb': 'depth_anything_v2_vitb.pth',
        'vitl': 'depth_anything_v2_vitl.pth'
    }
    
    if model_size not in model_urls:
        raise ValueError(f"Model size {model_size} not supported. Choose from: vits, vitb, vitl")
    
    print(f"\n{'='*60}")
    print(f"Downloading Depth Anything V2 {model_size.upper()} model...")
    print(f"URL: {model_urls[model_size]}")
    print(f"{'='*60}\n")
    
    # 设置下载路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.join(script_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    target_path = os.path.join(checkpoints_dir, model_file_map[model_size])
    
    # 检查是否已下载
    if os.path.exists(target_path):
        print(f"Model already exists at: {target_path}")
        print(f"File size: {os.path.getsize(target_path) / 1024 / 1024:.2f} MB")
        response = input("Re-download? (y/n): ")
        if response.lower() != 'y':
            print("Using existing model.")
            return target_path
    
    # 下载模型
    try:
        print(f"\nDownloading to: {target_path}\n")
        download_url(model_urls[model_size], target_path)
        print(f"\n{'='*60}")
        print(f"Model downloaded successfully!")
        print(f"Location: {target_path}")
        print(f"Size: {os.path.getsize(target_path) / 1024 / 1024:.2f} MB")
        print(f"{'='*60}\n")
        return target_path
    except Exception as e:
        print(f"\nError downloading model: {e}")
        print(f"\n{'='*60}")
        print("Manual download instructions:")
        print(f"1. Visit: {model_urls[model_size]}")
        print(f"2. Save the file to: {target_path}")
        print(f"{'='*60}\n")
        raise


if __name__ == "__main__":
    import sys
    
    model_size = 'vits'  # 默认使用Small模型（速度最快）
    if len(sys.argv) > 1:
        model_size = sys.argv[1]
    
    try:
        model_path = download_depth_anything_v2_direct(model_size)
        print("\nDone! You can now use the model for inference.")
        print(f"Test the model by running: python python_models/depth_model_dav2.py")
    except Exception as e:
        print(f"\nDownload failed. Please download manually following the instructions above.")
        sys.exit(1)

