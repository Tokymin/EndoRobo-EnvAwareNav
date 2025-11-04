$pythonEnv = "C:\Users\DELL\anaconda3\envs\EndoRobo-EnvAwareNav"
$targetDir = ".\build\bin\Release"

Write-Host "Copying Python DLLs to executable directory..." -ForegroundColor Green

# Copy main Python DLL
Copy-Item "$pythonEnv\python310.dll" $targetDir -Force
Write-Host "  Copied python310.dll"

# Copy important DLLs from root
$rootDlls = @("python3.dll", "vcruntime140.dll", "vcruntime140_1.dll")
foreach ($dll in $rootDlls) {
    $dllPath = Join-Path $pythonEnv $dll
    if (Test-Path $dllPath) {
        Copy-Item $dllPath $targetDir -Force
        Write-Host "  Copied $dll"
    }
}

# Copy Library\bin DLLs
$libBinPath = "$pythonEnv\Library\bin"
if (Test-Path $libBinPath) {
    Get-ChildItem "$libBinPath\*.dll" | ForEach-Object {
        Copy-Item $_.FullName $targetDir -Force
        Write-Host "  Copied $($_.Name) from Library\bin"
    }
}

Write-Host "`nDone! All Python DLLs copied." -ForegroundColor Green
Write-Host "Now you can run: .\build\bin\Release\endorobo_main.exe"

