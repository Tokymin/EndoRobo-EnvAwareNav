# PCL AllInOne安装脚本
# 此脚本帮助配置PCL AllInOne安装后的环境变量

param(
    [string]$PCLPath = "C:\Program Files\PCL 1.15.0"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PCL AllInOne 环境配置脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查PCL安装路径
if (-not (Test-Path $PCLPath)) {
    Write-Host "错误: 找不到PCL安装目录: $PCLPath" -ForegroundColor Red
    Write-Host "请确认安装路径，或使用 -PCLPath 参数指定正确路径" -ForegroundColor Yellow
    exit 1
}

Write-Host "检测到PCL安装路径: $PCLPath" -ForegroundColor Green

# 检查关键文件
$pclConfig = Join-Path $PCLPath "include\pcl\pcl_config.h"
if (-not (Test-Path $pclConfig)) {
    Write-Host "警告: 找不到PCL配置文件，可能安装不完整" -ForegroundColor Yellow
}

# 设置用户环境变量（当前会话）
$env:PCL_ROOT = $PCLPath
$env:PATH = "$PCLPath\bin;$env:PATH"

# 检查是否需要添加到系统PATH
$binPath = Join-Path $PCLPath "bin"
if ($env:PATH -notlike "*$binPath*") {
    Write-Host "正在添加PCL到系统PATH..." -ForegroundColor Yellow
    
    try {
        $currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
        if ($currentPath -notlike "*$binPath*") {
            [Environment]::SetEnvironmentVariable("Path", "$currentPath;$binPath", "User")
            Write-Host "已添加到用户PATH" -ForegroundColor Green
        }
    } catch {
        Write-Host "无法自动设置系统PATH，请手动添加: $binPath" -ForegroundColor Yellow
    }
}

# 设置PCL_ROOT环境变量
try {
    [Environment]::SetEnvironmentVariable("PCL_ROOT", $PCLPath, "User")
    Write-Host "已设置PCL_ROOT环境变量" -ForegroundColor Green
} catch {
    Write-Host "无法自动设置PCL_ROOT，请手动设置: PCL_ROOT=$PCLPath" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "配置完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "下一步操作:" -ForegroundColor Yellow
Write-Host "1. 重新打开终端或重启IDE以使环境变量生效" -ForegroundColor White
Write-Host "2. 删除 build/ 目录（如果存在）" -ForegroundColor White
Write-Host "3. 重新配置CMake项目" -ForegroundColor White
Write-Host "4. 应该看到: 'PCL 1.15.0 found - 3D reconstruction enabled'" -ForegroundColor White
Write-Host ""

# 验证CMake能否找到PCL
Write-Host "验证PCL配置..." -ForegroundColor Cyan
$cmakePath = Join-Path $PCLPath "lib\cmake\pcl-1.15"
if (Test-Path $cmakePath) {
    Write-Host "✓ CMake配置文件找到: $cmakePath" -ForegroundColor Green
} else {
    Write-Host "✗ 找不到CMake配置文件: $cmakePath" -ForegroundColor Red
}

$includePath = Join-Path $PCLPath "include\pcl"
if (Test-Path $includePath) {
    Write-Host "✓ 头文件目录找到: $includePath" -ForegroundColor Green
} else {
    Write-Host "✗ 找不到头文件目录: $includePath" -ForegroundColor Red
}

$libPath = Join-Path $PCLPath "lib"
if (Test-Path $libPath) {
    Write-Host "✓ 库文件目录找到: $libPath" -ForegroundColor Green
} else {
    Write-Host "✗ 找不到库文件目录: $libPath" -ForegroundColor Red
}

Write-Host ""


