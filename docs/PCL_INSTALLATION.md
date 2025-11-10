# PCL库安装指南

本文档说明如何在Windows上安装PCL (Point Cloud Library)库以支持3D重建功能。

## 方法1: 使用AllInOne安装包（最简单，推荐）⭐

这是最简单快捷的安装方法，适合大多数用户。

### 步骤1: 下载安装包

从PCL GitHub Releases页面下载：
- **PCL-1.15.0-AllInOne-win64.exe** (336 MB) - 主安装包
- **pcl-1.15.0-pdb-msvc2022-win64.zip** (181 MB) - 调试符号（可选，用于调试）

下载地址：https://github.com/PointCloudLibrary/pcl/releases

### 步骤2: 运行安装程序

1. 双击运行 `PCL-1.15.0-AllInOne-win64.exe`
2. 选择安装路径（建议使用默认路径：`C:\Program Files\PCL 1.15.0`）
3. **重要**：勾选"Add PCL to system PATH"选项（如果安装程序提供此选项）
4. 完成安装

### 步骤3: 配置环境变量

安装完成后，需要手动配置环境变量以确保PCL能被正确找到：

#### 3.1 添加PCL_ROOT环境变量

1. 打开"系统属性" → "高级" → "环境变量"
2. 在"系统变量"中点击"新建"
3. 变量名：`PCL_ROOT`
4. 变量值：`C:\Program Files\PCL 1.15.0`（根据实际安装路径调整）

#### 3.2 添加到PATH环境变量

在"系统变量"中找到`Path`，点击"编辑"，添加以下路径：

```
C:\Program Files\PCL 1.15.0\bin
C:\Program Files\PCL 1.15.0\3rdParty\VTK\bin
C:\Program Files\PCL 1.15.0\3rdParty\FLANN\bin
C:\Program Files\PCL 1.15.0\3rdParty\Qhull\bin
```

**注意**：如果某些3rdParty目录不存在，可以跳过。

#### 3.3 配置OpenNI2（如果需要RGB-D相机支持）

如果安装包包含OpenNI2：
1. 在安装目录的 `3rdParty\OpenNI2` 中找到并运行 `OpenNI-Windows-x64-2.2.msi`
2. 添加环境变量：
   - 变量名：`OPENNI2_REDIST64`
   - 变量值：`C:\Program Files\PCL 1.15.0\3rdParty\OpenNI2\Redist`

### 步骤4: 配置CMakeLists.txt

项目已经配置为自动查找PCL。如果CMake找不到PCL，可以手动指定路径：

在CMakeLists.txt中添加（在find_package之前）：

```cmake
# 如果自动查找失败，手动指定PCL路径
if(NOT PCL_FOUND)
    set(PCL_DIR "C:/Program Files/PCL 1.15.0/lib/cmake/pcl-1.15")
    find_package(PCL 1.9 QUIET COMPONENTS common io filters visualization)
endif()
```

### 步骤5: 验证安装

1. **检查环境变量**：
   ```powershell
   echo $env:PCL_ROOT
   ```

2. **检查文件是否存在**：
   ```powershell
   Test-Path "C:\Program Files\PCL 1.15.0\include\pcl\pcl_config.h"
   ```

3. **重新配置CMake项目**：
   - 删除 `build/` 目录
   - 重新运行CMake配置
   - 应该看到：`-- PCL 1.15.0 found - 3D reconstruction enabled`

### 步骤6: 安装调试符号（可选）

如果需要调试PCL代码：
1. 解压 `pcl-1.15.0-pdb-msvc2022-win64.zip`
2. 将 `.pdb` 文件复制到 `C:\Program Files\PCL 1.15.0\bin` 目录

## 方法2: 使用vcpkg安装

### 步骤1: 确保vcpkg已配置

项目已经包含了vcpkg，位于 `vcpkg/` 目录。

### 步骤2: 安装PCL

在项目根目录执行：

```powershell
.\vcpkg\vcpkg.exe install pcl:x64-windows
```

**注意**: PCL编译需要较长时间（通常30-60分钟），请耐心等待。

### 步骤3: 配置CMake使用vcpkg工具链

在CMake配置时，需要指定vcpkg工具链：

```powershell
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake
```

或者在Visual Studio中：
1. 打开CMakeSettings.json
2. 添加 `"cmakeToolchainFile": "${workspaceFolder}/vcpkg/scripts/buildsystems/vcpkg.cmake"`

### 步骤4: 验证安装

重新配置CMake后，应该看到：
```
-- PCL 1.15.1 found - 3D reconstruction enabled
```

## 方法2: 手动安装PCL（如果vcpkg安装失败）

### 下载预编译版本

1. 访问PCL官方下载页面：https://github.com/PointCloudLibrary/pcl/releases
2. 下载Windows预编译版本（如果有）
3. 解压到指定目录，例如 `C:\PCL`

### 配置环境变量

设置以下环境变量：
- `PCL_ROOT=C:\PCL`
- 将 `C:\PCL\bin` 添加到 `PATH`

### 在CMakeLists.txt中手动指定PCL路径

如果vcpkg安装失败，可以手动指定PCL路径：

```cmake
set(PCL_DIR "C:/PCL/lib/cmake/pcl-1.15")
find_package(PCL REQUIRED)
```

## 方法3: 使用Conda安装（替代方案）

如果vcpkg安装持续失败，可以考虑使用Conda：

```bash
conda install -c conda-forge pcl
```

然后在CMakeLists.txt中配置相应的路径。

## 故障排除

### 问题1: 编译超时或内存不足

PCL编译需要大量内存和时间。建议：
- 关闭其他占用内存的程序
- 使用 `--no-downloads` 选项避免重复下载
- 只编译Release版本（如果只需要Release）

### 问题2: 找不到PCL

确保：
1. vcpkg工具链已正确配置
2. PCL已成功安装（检查 `vcpkg\installed\x64-windows\include\pcl`）
3. CMake缓存已清除：删除 `build/` 目录后重新配置

### 问题3: 链接错误

检查CMakeLists.txt中的PCL组件是否正确：
- `PCL::common`
- `PCL::io`
- `PCL::filters`
- `PCL::visualization`

## 验证安装

安装成功后，可以运行以下命令验证：

```powershell
.\vcpkg\vcpkg.exe list pcl
```

应该看到：
```
pcl:x64-windows                   1.15.1           Point Cloud Library...
```

## 下一步

安装PCL后：
1. 重新配置CMake项目
2. 编译项目
3. 3D重建功能将自动启用

## 参考资源

- PCL官方文档: https://pcl.readthedocs.io/
- vcpkg PCL端口: https://github.com/microsoft/vcpkg/tree/master/ports/pcl
- PCL GitHub: https://github.com/PointCloudLibrary/pcl

