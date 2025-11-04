# Depth Anything V2 模型下载说明

由于网络问题，您可以通过以下方式手动下载模型：

## 方法1：直接下载链接

### Small模型 (推荐，最快)
- **大小**: 24.8M
- **下载链接**: https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
- **保存路径**: `python_models/depth_anything_v2/checkpoints/depth_anything_v2_vits.pth`

### Base模型 (中等)
- **大小**: 97.5M  
- **下载链接**: https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth
- **保存路径**: `python_models/depth_anything_v2/checkpoints/depth_anything_v2_vitb.pth`

### Large模型 (最精确但最慢)
- **大小**: 335.3M
- **下载链接**: https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
- **保存路径**: `python_models/depth_anything_v2/checkpoints/depth_anything_v2_vitl.pth`

## 方法2：使用浏览器下载

1. 访问 Hugging Face 页面:
   - Small: https://huggingface.co/depth-anything/Depth-Anything-V2-Small
   - Base: https://huggingface.co/depth-anything/Depth-Anything-V2-Base
   - Large: https://huggingface.co/depth-anything/Depth-Anything-V2-Large

2. 点击 "Files and versions" 标签

3. 下载对应的 .pth 文件

4. 将文件放到: `python_models/depth_anything_v2/checkpoints/` 目录下

## 方法3：使用镜像站（中国大陆用户）

如果无法访问HuggingFace，可以使用镜像站：

```bash
# 使用hf-mirror
HF_ENDPOINT=https://hf-mirror.com python python_models/depth_anything_v2/download_model.py vits
```

## 推荐配置

- **实时应用**: 使用 Small (vits) 模型
- **平衡模式**: 使用 Base (vitb) 模型
- **离线/高精度**: 使用 Large (vitl) 模型

## 下载后验证

下载完成后，确保文件结构如下：

```
python_models/
└── depth_anything_v2/
    └── checkpoints/
        └── depth_anything_v2_vits.pth (或 vitb.pth 或 vitl.pth)
```

然后运行测试：

```bash
python python_models/depth_model_dav2.py
```

