# Monitor depth estimation in real-time

Write-Host "=== Depth Estimation Monitor ===" -ForegroundColor Green
Write-Host "Watching log file for depth estimation activity..." -ForegroundColor Yellow
Write-Host ""

$lastSize = 0

while ($true) {
    Start-Sleep -Milliseconds 500
    
    if (Test-Path "endorobo.log") {
        $currentSize = (Get-Item "endorobo.log").Length
        
        if ($currentSize -ne $lastSize) {
            $lastSize = $currentSize
            
            # Show latest depth-related logs
            $content = Get-Content "endorobo.log" -Tail 50 | Select-String "Depth estimation: frame|First frame|Captured [0-9]+ frames|CUDA|cuda"
            
            if ($content) {
                Clear-Host
                Write-Host "=== Depth Estimation Monitor ===" -ForegroundColor Green
                Write-Host "Latest activity:" -ForegroundColor Cyan
                $content | ForEach-Object { Write-Host $_ }
                Write-Host ""
                Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor DarkGray
            }
        }
    }
}

