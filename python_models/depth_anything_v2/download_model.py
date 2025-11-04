"""
下载Depth Anything V2模型
"""
import os
import torch
from huggingface_hub import hf_hub_download

def download_depth_anything_v2_model(model_size='vits'):
    """
    下载Depth Anything V2模型
    
    Args:
        model_size: 'vits' (Small - 24.8M), 'vitb' (Base - 97.5M), 'vitl' (Large - 335.3M)
    """
    model_repo_map = {
        'vits': 'depth-anything/Depth-Anything-V2-Small',
        'vitb': 'depth-anything/Depth-Anything-V2-Base',
        'vitl': 'depth-anything/Depth-Anything-V2-Large'
    }
    
    model_file_map = {
        'vits': 'depth_anything_v2_vits.pth',
        'vitb': 'depth_anything_v2_vitb.pth',
        'vitl': 'depth_anything_v2_vitl.pth'
    }
    
    if model_size not in model_repo_map:
        raise ValueError(f"Model size {model_size} not supported. Choose from: vits, vitb, vitl")
    
    print(f"Downloading Depth Anything V2 {model_size.upper()} model...")
    print(f"From: {model_repo_map[model_size]}")
    
    # 设置下载路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.join(script_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # 下载模型
    model_path = hf_hub_download(
        repo_id=model_repo_map[model_size],
        filename=model_file_map[model_size],
        cache_dir=checkpoints_dir
    )
    
    # 复制到标准位置
    target_path = os.path.join(checkpoints_dir, model_file_map[model_size])
    if model_path != target_path:
        import shutil
        shutil.copy(model_path, target_path)
    
    print(f"Model downloaded to: {target_path}")
    return target_path


if __name__ == "__main__":
    import sys
    
    model_size = 'vits'  # 默认使用Small模型（速度最快）
    if len(sys.argv) > 1:
        model_size = sys.argv[1]
    
    download_depth_anything_v2_model(model_size)
    print("\nDone! You can now use the model for inference.")

