# 快速开始指南

## 5分钟快速上手

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/EndoRobo-EnvAwareNav.git
cd EndoRobo-EnvAwareNav
```

### 2. 安装依赖

#### Windows (vcpkg)

```powershell
# 安装vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# 安装依赖
.\vcpkg install opencv:x64-windows eigen3:x64-windows pcl:x64-windows yaml-cpp:x64-windows

cd ..
```

#### Linux

```bash
sudo apt-get install -y libopencv-dev libeigen3-dev libpcl-dev libyaml-cpp-dev python3-dev
```

### 3. 安装Python依赖

```bash
pip install -r python_models/requirements.txt
```

### 4. 编译项目

#### Windows

```powershell
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg路径]/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

#### Linux

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 5. 配置相机参数

编辑 `config/camera_config.yaml`：

```yaml
camera:
  width: 1920
  height: 1080
  camera_id: 0  # 根据你的相机调整
```

### 6. 运行程序

```bash
# Windows
.\build\bin\Release\endorobo_main.exe

# Linux
./build/bin/endorobo_main
```

## 测试系统

如果没有实际的内窥镜摄像头，可以使用普通USB摄像头或视频文件进行测试。

## 下一步

- 阅读 [详细文档](../README.md)
- 配置 [Python模型](../python_models/README.md)
- 查看 [示例代码](EXAMPLES.md)

## 常见问题

### Q: 相机打不开？
A: 检查 `camera_id` 配置，尝试 0、1、2 等不同值。

### Q: Python模型报错？
A: 确保已安装 PyTorch，检查模型文件路径。

### Q: 编译失败？
A: 确认所有依赖库已安装，CMake版本 >= 3.15。

## 获取帮助

遇到问题？请：
1. 查看 [README.md](../README.md)
2. 提交 [Issue](https://github.com/yourusername/EndoRobo-EnvAwareNav/issues)

