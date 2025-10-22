#requires -version 5.0
Param(
    [string]$PythonPath = "python",
    [string]$VenvDir = ".venv"
)

Write-Host "==> Creating virtual environment ($VenvDir)..." -ForegroundColor Cyan
if (-Not (Test-Path $VenvDir)) {
    & $PythonPath -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment. Please check that Python is available."
    }
} else {
    Write-Host "Virtual environment already exists. Skipping creation." -ForegroundColor Yellow
}

$venvActivate = Join-Path $VenvDir "Scripts\Activate.ps1"
if (-Not (Test-Path $venvActivate)) {
    throw "Virtual environment activation script not found: $venvActivate"
}

Write-Host "==> Installing project dependencies..." -ForegroundColor Cyan
& "$VenvDir\Scripts\pip.exe" install --upgrade pip
& "$VenvDir\Scripts\pip.exe" install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed. Please check network connectivity or requirements.txt."
}

Write-Host ""
Write-Host "==> Setup complete! Next steps:" -ForegroundColor Green
Write-Host "    .\$VenvDir\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "    setx DASHSCOPE_API_KEY \"your_dashscope_key\"" -ForegroundColor Green
Write-Host "    setx VECTOR_DATA_DIR \"C:\faiss_data\"  # optional" -ForegroundColor Green
Write-Host ""
Write-Host "Redis defaults to localhost:6379. Update .env if you use a different host/port." -ForegroundColor Green
