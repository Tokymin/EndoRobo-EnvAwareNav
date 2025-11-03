# Python Models Directory

## 目录说明

这个目录用于存放Python训练好的深度学习模型。

## 文件说明

- `pose_model.py` - 位姿估计模型接口
- `depth_model.py` - 深度估计模型接口
- `requirements.txt` - Python依赖包列表
- `pose_estimation/` - 位姿估计模型权重文件目录（需要创建）
- `depth_estimation/` - 深度估计模型权重文件目录（需要创建）

## 使用说明

### 1. 安装Python依赖

```bash
pip install -r requirements.txt
```

### 2. 准备模型文件

将你训练好的模型权重文件放置在相应目录：

```
python_models/
├── pose_estimation/
│   └── model.pth          # 位姿估计模型权重
├── depth_estimation/
│   └── model.pth          # 深度估计模型权重
```

### 3. 修改模型接口

编辑 `pose_model.py` 和 `depth_model.py`，替换占位符代码为你的实际模型加载和推理代码。

### 4. 测试模型

```bash
python pose_model.py
python depth_model.py
```

## 模型接口规范

### 位姿估计模型 (pose_model.py)

**输入：**
- `image`: NumPy数组，形状 (H, W, 3)，RGB格式，归一化到 [0, 1]
- `previous_image`: （可选）前一帧图像

**输出：**
```python
{
    'translation': [x, y, z],          # 平移向量（米）
    'rotation': [qw, qx, qy, qz],      # 四元数 (w, x, y, z)
    'confidence': float                 # 置信度 [0, 1]
}
```

### 深度估计模型 (depth_model.py)

**输入：**
- `image`: NumPy数组，形状 (H, W, 3)，RGB格式，归一化到 [0, 1]

**输出：**
- `depth_map`: NumPy数组，形状 (H, W)，float32类型，归一化到 [0, 1]

实际深度值计算：
```
actual_depth = depth_map * (max_depth - min_depth) + min_depth
```

## 推荐的深度学习模型

### 深度估计
- **MiDaS** - 单目深度估计
- **DPT** - Dense Prediction Transformer
- **MonoDepth2** - 自监督单目深度估计
- **EndoSLAM** - 专门针对内窥镜的深度估计

### 位姿估计
- **PoseNet** - 相机位姿估计
- **DeepVO** - 基于深度学习的视觉里程计
- **DROID-SLAM** - 深度学习SLAM
- **EndoMapper** - 内窥镜场景的位姿估计

## 医疗内窥镜特定注意事项

1. **光照变化**：肠腔环境光照变化大，模型需要对光照鲁棒
2. **纹理稀疏**：肠腔表面纹理较少，需要加强特征提取
3. **镜面反射**：注意处理内窥镜图像中的高光和反射
4. **形变**：肠腔会发生形变，需要动态重建
5. **尺度模糊**：单目深度估计的尺度需要通过其他方式校准

## 数据集推荐

- **SCARED** - Surgical endoscopic vision challenge
- **Hamlyn** - Hamlyn Centre Laparoscopic dataset
- **EndoVis** - Endoscopic Vision challenges
- **CholecSeg8k** - Cholecystectomy dataset

## 参考文献

1. Chen et al., "Self-supervised Depth Estimation in Laparoscopic Surgery", MICCAI 2020
2. Turan et al., "Deep EndoVO: A Recurrent Convolutional Neural Network for Monocular Endoscopy", IROS 2018
3. Ozyoruk et al., "EndoSLAM dataset and an unsupervised monocular visual odometry and depth estimation approach for endoscopic videos", Medical Image Analysis 2021

