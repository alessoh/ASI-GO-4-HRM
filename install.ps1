# ASI-GO-4-HRM Installation Script for PowerShell

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ASI-GO-4-HRM Installation Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
Write-Host ""

# Install PyTorch CPU version
Write-Host "Installing PyTorch (CPU version)..." -ForegroundColor Yellow
pip install torch --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install PyTorch. Trying alternative method..." -ForegroundColor Red
    pip install torch
}
Write-Host ""

# Install numpy with specific version
Write-Host "Installing NumPy..." -ForegroundColor Yellow
pip install "numpy>=1.24.0,<2.0.0"
Write-Host ""

# Install core dependencies
Write-Host "Installing core dependencies..." -ForegroundColor Yellow
pip install python-dotenv colorama tqdm psutil
Write-Host ""

# Install LLM providers
Write-Host "Installing LLM providers..." -ForegroundColor Yellow
pip install openai google-generativeai anthropic
Write-Host ""

# Install code analysis tools
Write-Host "Installing code analysis tools..." -ForegroundColor Yellow
pip install astunparse black autopep8
Write-Host ""

# Install utilities
Write-Host "Installing utilities..." -ForegroundColor Yellow
pip install requests aiohttp tenacity jsonschema scikit-learn
Write-Host ""

Write-Host "========================================" -ForegroundColor Green
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Create a .env file with your API keys" -ForegroundColor White
Write-Host "2. Run: python initialize_hrm.py" -ForegroundColor White
Write-Host "3. Run: python main.py --mode hybrid --goal 'your problem'" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Green