@echo off
echo ========================================
echo ASI-GO-4-HRM Installation Script
echo ========================================
echo.

REM Upgrade pip first
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install PyTorch CPU version specifically
echo Installing PyTorch (CPU version)...
pip install torch --index-url https://download.pytorch.org/whl/cpu
echo.

REM Install numpy with specific version
echo Installing NumPy...
pip install "numpy>=1.24.0,<2.0.0"
echo.

REM Install core dependencies
echo Installing core dependencies...
pip install python-dotenv colorama tqdm psutil
echo.

REM Install LLM providers
echo Installing LLM providers...
pip install openai google-generativeai anthropic
echo.

REM Install code analysis tools
echo Installing code analysis tools...
pip install astunparse black autopep8
echo.

REM Install utilities
echo Installing utilities...
pip install requests aiohttp tenacity jsonschema scikit-learn
echo.

echo.
echo ========================================
echo Installation complete!
echo.
echo Next steps:
echo 1. Create a .env file with your API keys
echo 2. Run: python initialize_hrm.py
echo 3. Run: python main.py --mode hybrid --goal "your problem"
echo ========================================
pause