# 开发指南

## 项目结构

### 模块划分

项目按功能模块组织：

1. **Core模块** (`include/core/`, `src/core/`)
   - 配置管理
   - 日志系统
   - 基础工具

2. **Camera模块** (`include/camera/`, `src/camera/`)
   - 相机采集
   - 图像预处理

3. **Python Interface模块** (`include/python_interface/`, `src/python_interface/`)
   - Python包装器
   - 位姿估计接口
   - 深度估计接口

4. **Reconstruction模块** (`include/reconstruction/`, `src/reconstruction/`)
   - 点云构建
   - 肠腔重建
   - 冗余点去除

5. **Utils模块** (`include/utils/`, `src/utils/`)
   - 数学工具
   - 计时器
   - 其他工具函数

## 代码规范

### 命名规范

```cpp
// 类名：大驼峰命名
class CameraCapture { };

// 函数名：小驼峰命名
void processImage();

// 变量名：小写+下划线
int frame_count_;

// 常量：大写+下划线
const int MAX_BUFFER_SIZE = 100;

// 命名空间：小写
namespace endorobo { }
```

### 注释规范

```cpp
/**
 * @brief 简要描述
 * 
 * 详细描述（可选）
 * 
 * @param input 输入参数说明
 * @return 返回值说明
 */
bool processFrame(const cv::Mat& input);
```

### 头文件保护

```cpp
#pragma once  // 推荐使用

// 或者
#ifndef ENDOROBO_MODULE_CLASS_H
#define ENDOROBO_MODULE_CLASS_H
// ...
#endif
```

## 添加新功能

### 1. 添加新的处理模块

创建头文件 `include/module_name/feature_name.h`：

```cpp
#pragma once

namespace endorobo {

class FeatureName {
public:
    FeatureName();
    ~FeatureName();
    
    bool initialize();
    bool process();
    
private:
    // 私有成员
};

} // namespace endorobo
```

创建源文件 `src/module_name/feature_name.cpp`：

```cpp
#include "module_name/feature_name.h"
#include "core/logger.h"

namespace endorobo {

FeatureName::FeatureName() {
}

FeatureName::~FeatureName() {
}

bool FeatureName::initialize() {
    LOG_INFO("Initializing FeatureName...");
    return true;
}

bool FeatureName::process() {
    return true;
}

} // namespace endorobo
```

在 `CMakeLists.txt` 中添加：

```cmake
set(MODULE_SOURCES
    src/module_name/feature_name.cpp
)

# 添加到库中
target_sources(endorobo_core PRIVATE ${MODULE_SOURCES})
```

### 2. 添加Python模型接口

参考 `python_models/pose_model.py` 的模板创建新的模型接口。

关键要求：
- 提供 `init_model()` 函数
- 提供 `predict_xxx()` 函数
- 输入输出使用NumPy数组
- 返回标准格式的字典或数组

### 3. 添加配置选项

在 `config/camera_config.yaml` 中添加配置：

```yaml
new_feature:
  enabled: true
  parameter1: value1
  parameter2: value2
```

在 `include/core/config_manager.h` 中添加结构体：

```cpp
struct NewFeatureConfig {
    bool enabled;
    int parameter1;
    double parameter2;
};
```

在 `src/core/config_manager.cpp` 中解析配置：

```cpp
if (config["new_feature"]) {
    auto nf = config["new_feature"];
    new_feature_config_.enabled = nf["enabled"].as<bool>();
    // ...
}
```

## 测试

### 单元测试

推荐使用 Google Test 框架：

```cpp
#include <gtest/gtest.h>
#include "module_name/feature_name.h"

TEST(FeatureNameTest, BasicTest) {
    endorobo::FeatureName feature;
    EXPECT_TRUE(feature.initialize());
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

### 集成测试

创建测试程序验证各模块协作：

```cpp
// test/integration_test.cpp
int main() {
    // 初始化各模块
    // 运行完整流程
    // 验证结果
}
```

## 调试技巧

### 使用日志

```cpp
LOG_DEBUG("Debug information");
LOG_INFO("Normal information");
LOG_WARNING("Warning message");
LOG_ERROR("Error occurred");
LOG_FATAL("Fatal error");
```

### 使用计时器

```cpp
#include "utils/timer.h"

// 方法1：手动计时
Timer timer;
timer.start("operation");
// ... 操作 ...
double elapsed = timer.stop("operation");
LOG_INFO("Operation took ", elapsed, " ms");

// 方法2：作用域计时（RAII）
{
    ScopedTimer timer("operation");
    // ... 操作 ...
} // 自动打印耗时
```

### Visual Studio调试

1. 设置断点
2. F5 开始调试
3. 使用监视窗口查看变量
4. 使用立即窗口执行表达式

### GDB调试

```bash
gdb ./build/bin/endorobo_main
(gdb) break main.cpp:100
(gdb) run
(gdb) print variable_name
(gdb) continue
```

## 性能优化

### 1. 多线程

```cpp
#include <thread>
#include <future>

// 异步执行
std::future<Result> future = std::async(std::launch::async, []() {
    return doHeavyWork();
});

// 后续获取结果
Result result = future.get();
```

### 2. 内存优化

```cpp
// 预分配内存
cloud->reserve(expected_size);

// 使用智能指针
std::shared_ptr<PointCloud> cloud = std::make_shared<PointCloud>();

// 及时释放不需要的资源
large_data.clear();
```

### 3. GPU加速

在Python模型中启用GPU：

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

## 发布流程

### 1. 版本号

遵循语义化版本 (Semantic Versioning)：

- `MAJOR.MINOR.PATCH`
- 例如：`1.2.3`

### 2. 更新日志

在 `CHANGELOG.md` 中记录更新：

```markdown
## [1.2.0] - 2025-11-03

### Added
- 新功能描述

### Changed
- 修改内容

### Fixed
- 修复的bug
```

### 3. 打包

```bash
# 创建发布包
cd build
cmake --build . --config Release
cpack
```

## 贡献流程

1. Fork 项目
2. 创建特性分支：`git checkout -b feature/my-feature`
3. 提交代码：`git commit -am 'Add my feature'`
4. 推送分支：`git push origin feature/my-feature`
5. 提交 Pull Request

### Pull Request要求

- 描述清楚修改内容
- 通过所有测试
- 遵循代码规范
- 包含必要的文档更新

## 参考资源

- [Modern CMake](https://cliutils.gitlab.io/modern-cmake/)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [PCL Tutorial](https://pcl.readthedocs.io/)

