# ========================================
# EndoRobo Dependencies Setup Script
# ========================================

Write-Host "`n========================================"
Write-Host "EndoRobo Auto Setup"
Write-Host "========================================`n"

$ProjectRoot = "F:\Toky\VSProject\Repos\EndoRobo-EnvAwareNav"
$VcpkgRoot = "$ProjectRoot\vcpkg"

# Check OpenCV
Write-Host "Step 1: Checking dependencies...`n"

Write-Host "  Checking OpenCV..." -NoNewline
$opencvFound = $false
$opencvPath = ""
$pathMatches = $env:PATH -split ';' | Where-Object { $_ -match 'opencv' }
if ($pathMatches) {
    $opencvFound = $true
    $opencvPath = "D:/Program Files/opencv-4.3.0/opencv_cmake/install"
    Write-Host " [OK] Found in PATH"
    Write-Host "    Path: $opencvPath"
} else {
    Write-Host " [NOT FOUND]"
}

Write-Host "  Checking Eigen3..." -NoNewline
$eigen3Found = Test-Path "$VcpkgRoot\installed\x64-windows\include\eigen3"
if ($eigen3Found) {
    Write-Host " [OK] Found in vcpkg"
} else {
    Write-Host " [NOT FOUND]"
}

Write-Host "  Checking yaml-cpp..." -NoNewline
$yamlFound = Test-Path "$VcpkgRoot\installed\x64-windows\include\yaml-cpp"
if ($yamlFound) {
    Write-Host " [OK] Found in vcpkg"
} else {
    Write-Host " [NOT FOUND]"
}

# Install missing dependencies
Write-Host "`nStep 2: Installing missing dependencies...`n"

Set-Location $VcpkgRoot

if (-not $eigen3Found) {
    Write-Host "  Installing eigen3..."
    .\vcpkg install eigen3:x64-windows
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    [OK] eigen3 installed"
        $eigen3Found = $true
    }
}

if (-not $yamlFound) {
    Write-Host "  Installing yaml-cpp..."
    .\vcpkg install yaml-cpp:x64-windows
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    [OK] yaml-cpp installed"
        $yamlFound = $true
    }
}

if ($eigen3Found -and $yamlFound) {
    Write-Host "`n  [OK] All required dependencies ready!"
}

# Configure CMake
Write-Host "`nStep 3: Configuring CMake...`n"

Set-Location $ProjectRoot

if (Test-Path "build") {
    Write-Host "  Cleaning old build directory..."
    Remove-Item -Recurse -Force "build" -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Path "build" -Force | Out-Null
Set-Location "build"

$cmakeCmd = "cmake .. -DCMAKE_TOOLCHAIN_FILE=`"$VcpkgRoot\scripts\buildsystems\vcpkg.cmake`" -DOpenCV_DIR=`"$opencvPath`""

Write-Host "  Running: $cmakeCmd`n"
Invoke-Expression $cmakeCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n  [OK] CMake configuration successful!"
    
    # Build
    Write-Host "`nStep 4: Building project...`n"
    Write-Host "  Building Release configuration..."
    cmake --build . --config Release
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n  [SUCCESS] Build completed!"
        Write-Host "`n  Executable: $ProjectRoot\build\bin\Release\endorobo_main.exe"
    } else {
        Write-Host "`n  [ERROR] Build failed"
    }
} else {
    Write-Host "`n  [ERROR] CMake configuration failed"
}

# Summary
Write-Host "`n========================================"
Write-Host "Setup Summary"
Write-Host "========================================`n"

if ($opencvFound) { Write-Host "  [OK] OpenCV" } else { Write-Host "  [X] OpenCV" }
if ($eigen3Found) { Write-Host "  [OK] Eigen3" } else { Write-Host "  [X] Eigen3" }
if ($yamlFound) { Write-Host "  [OK] yaml-cpp" } else { Write-Host "  [X] yaml-cpp" }
Write-Host "  [-] PCL (skipped - too large)"

Write-Host "`nNext steps:"
Write-Host "  1. Install Python packages: pip install -r python_models/requirements.txt"
Write-Host "  2. Configure camera: edit config/camera_config.yaml"
Write-Host "  3. Add Python models: see python_models/README.md"
Write-Host "  4. Run: .\build\bin\Release\endorobo_main.exe`n"

Set-Location $ProjectRoot
