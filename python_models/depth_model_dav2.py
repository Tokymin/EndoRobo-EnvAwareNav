"""
Depth Anything V2 深度估计模型
用于实时深度估计
"""
import numpy as np
import cv2
import torch
import sys
import os

# 添加depth_anything_v2路径
sys.path.append(os.path.dirname(__file__))

from depth_anything_v2.dpt import DepthAnythingV2


class DepthModel:
    """Depth Anything V2深度估计模型"""
    
    def __init__(self, model_size='vits', device='cuda'):
        """
        初始化模型
        
        Args:
            model_size: 'vits' (Small), 'vitb' (Base), 'vitl' (Large)
            device: 'cuda', 'cpu', or 'mps'
        """
        self.model_size = model_size
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        
        # 模型配置
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        
        print(f"[DepthModel] Initializing Depth Anything V2 {model_size.upper()} on {self.device}...")
        
        # 创建模型
        self.model = DepthAnythingV2(**self.model_configs[model_size])
        
        # 加载权重
        model_path = self._get_model_path()
        if os.path.exists(model_path):
            print(f"[DepthModel] Loading model from: {model_path}")
            state_dict = torch.load(model_path, map_location='cpu')
            # 使用 strict=False 允许部分权重不匹配（兼容不同版本）
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"[DepthModel] Warning: Missing keys ({len(missing_keys)} keys)")
            if unexpected_keys:
                print(f"[DepthModel] Warning: Unexpected keys ({len(unexpected_keys)} keys)")
        else:
            print(f"[DepthModel] Warning: Model file not found at {model_path}")
            print(f"[DepthModel] Please run: python python_models/depth_anything_v2/download_model.py {model_size}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = self.model.to(self.device).eval()
        print(f"[DepthModel] Model loaded successfully!")
    
    def _get_model_path(self):
        """获取模型路径"""
        script_dir = os.path.dirname(__file__)
        model_filename = f'depth_anything_v2_{self.model_size}.pth'
        return os.path.join(script_dir, 'depth_anything_v2', 'checkpoints', model_filename)
    
    @torch.no_grad()
    def estimate_depth(self, image, input_size=518):
        """
        估计深度图
        
        Args:
            image: BGR格式的numpy数组 (H, W, 3)
            input_size: 输入尺寸，越大越精细但越慢
            
        Returns:
            depth: 深度图 (H, W)，归一化到0-255
        """
        if image is None or image.size == 0:
            return None
        
        # 转换为RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 使用模型推理
        depth = self.model.infer_image(image, input_size=input_size)
        
        return depth
    
    def get_colored_depth(self, depth_map):
        """
        将深度图转换为彩色可视化
        
        Args:
            depth_map: 深度图 (H, W)
            
        Returns:
            colored_depth: 彩色深度图 (H, W, 3)
        """
        # 应用颜色映射
        colored_depth = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
        return colored_depth


def test_model():
    """测试模型"""
    import sys
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 初始化模型
    model = DepthModel(model_size='vits', device='cuda')
    
    # 估计深度
    print("Estimating depth...")
    depth = model.estimate_depth(test_image)
    
    print(f"Depth shape: {depth.shape}")
    print(f"Depth range: [{depth.min()}, {depth.max()}]")
    
    # 可视化
    colored = model.get_colored_depth(depth)
    print(f"Colored depth shape: {colored.shape}")
    
    print("Test passed!")


if __name__ == "__main__":
    test_model()

