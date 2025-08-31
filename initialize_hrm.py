"""
Initialize HRM components for ASI-GO-4-HRM
Downloads base models and sets up directories
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
import urllib.request
import zipfile
import argparse
import logging

from hrm_evaluator import HierarchicalReasoningModel, HRMEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary directories"""
    dirs = [
        'models',
        'data',
        'cache',
        'logs',
        'checkpoints'
    ]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        logger.info(f"✓ Created directory: {dir_name}")


def create_default_configs():
    """Create default configuration files"""
    
    # HRM configuration
    hrm_config = {
        "model": {
            "hidden_size": 256,
            "n_cycles": 4,
            "t_steps": 4,
            "dropout": 0.1,
            "input_size": 512
        },
        "evaluation": {
            "confidence_threshold": 0.85,
            "fallback_to_llm": True,
            "cache_evaluations": True,
            "cache_size": 1000
        },
        "training": {
            "batch_size": 8,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "epochs": 10,
            "val_split": 0.2
        }
    }
    
    with open('hrm_config.json', 'w') as f:
        json.dump(hrm_config, f, indent=2)
    logger.info("✓ Created hrm_config.json")
    
    # Evaluation weights configuration
    eval_weights = {
        "correctness": 0.40,
        "efficiency": 0.25,
        "readability": 0.20,
        "generality": 0.15
    }
    
    with open('evaluation_weights.json', 'w') as f:
        json.dump(eval_weights, f, indent=2)
    logger.info("✓ Created evaluation_weights.json")
    
    # Main configuration update for hybrid mode
    main_config = {
        "llm_provider": "openai",
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 2000,
        "max_attempts": 5,
        "evaluation_mode": "hybrid",
        "hrm_enabled": True,
        "hrm_confidence_threshold": 0.85,
        "use_cache": True,
        "verbose": False
    }
    
    # Check if config.json exists and update it
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            existing_config = json.load(f)
        existing_config.update(main_config)
        main_config = existing_config
    
    with open('config.json', 'w') as f:
        json.dump(main_config, f, indent=2)
    logger.info("✓ Updated config.json for hybrid mode")


def initialize_hrm_model(force_new=False):
    """Initialize or download HRM model"""
    model_path = 'models/hrm_evaluator.pth'
    
    if os.path.exists(model_path) and not force_new:
        logger.info(f"✓ Model already exists: {model_path}")
        return
    
    logger.info("Initializing new HRM model...")
    
    # Create new model
    model = HierarchicalReasoningModel(input_size=512, hidden_size=256)
    
    # Initialize weights with good defaults
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    # Save initialized model
    torch.save(model.state_dict(), model_path)
    logger.info(f"✓ Initialized and saved HRM model to {model_path}")


def create_sample_knowledge_base():
    """Create sample knowledge base for training"""
    kb_path = 'knowledge_base.json'
    
    if os.path.exists(kb_path):
        logger.info(f"✓ Knowledge base already exists: {kb_path}")
        return
    
    # Create sample knowledge base with a few examples
    sample_kb = {
        "find_prime_numbers": {
            "solution": """def find_primes(n):
    \"\"\"Find all prime numbers up to n using Sieve of Eratosthenes\"\"\"
    if n < 2:
        return []
    
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    
    return [i for i in range(n + 1) if sieve[i]]""",
            "fitness_score": 0.92,
            "strategy": "Sieve of Eratosthenes algorithm"
        },
        "fibonacci_sequence": {
            "solution": """def fibonacci(n):
    \"\"\"Generate Fibonacci sequence up to n terms\"\"\"
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[-1] + fib[-2])
    
    return fib""",
            "fitness_score": 0.85,
            "strategy": "Iterative approach"
        },
        "binary_search": {
            "solution": """def binary_search(arr, target):
    \"\"\"Binary search implementation\"\"\"
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1""",
            "fitness_score": 0.88,
            "strategy": "Divide and conquer"
        }
    }
    
    with open(kb_path, 'w') as f:
        json.dump(sample_kb, f, indent=2)
    logger.info(f"✓ Created sample knowledge base: {kb_path}")


def run_self_test():
    """Run self-test to verify installation"""
    logger.info("\n🧪 Running self-test...")
    
    try:
        # Test 1: Model loading
        logger.info("Test 1: Loading HRM model...")
        evaluator = HRMEvaluator(model_path="models/hrm_evaluator.pth")
        logger.info("✓ Model loaded successfully")
        
        # Test 2: Simple evaluation
        logger.info("Test 2: Evaluating sample code...")
        test_code = """def add(a, b):
    return a + b"""
        
        result = evaluator.evaluate(test_code, "Add two numbers")
        logger.info(f"✓ Evaluation completed: Overall score = {result.overall:.2f}")
        
        # Test 3: Config loading
        logger.info("Test 3: Loading configurations...")
        with open('hrm_config.json', 'r') as f:
            config = json.load(f)
        logger.info("✓ Configuration loaded successfully")
        
        # Test 4: Knowledge base access
        logger.info("Test 4: Accessing knowledge base...")
        with open('knowledge_base.json', 'r') as f:
            kb = json.load(f)
        logger.info(f"✓ Knowledge base loaded with {len(kb)} examples")
        
        logger.info("\n✅ All tests passed! System is ready.")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Self-test failed: {e}")
        return False


def check_dependencies():
    """Check if all required dependencies are installed"""
    logger.info("\n📦 Checking dependencies...")
    
    required = {
        'torch': '2.0.0',
        'numpy': '1.24.0',
        'tqdm': '4.65.0'
    }
    
    missing = []
    for package, min_version in required.items():
        try:
            module = __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing.append(package)
            logger.error(f"✗ {package} is missing")
    
    if missing:
        logger.error(f"\n❌ Missing dependencies: {', '.join(missing)}")
        logger.info("Please run: pip install -r requirements.txt")
        return False
    
    return True


def display_welcome():
    """Display welcome message"""
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║          ASI-GO-4-HRM Initialization Tool           ║
    ║    Self-Improving AI with Hierarchical Reasoning    ║
    ╚══════════════════════════════════════════════════════╝
    """)


def main():
    parser = argparse.ArgumentParser(description='Initialize ASI-GO-4-HRM')
    parser.add_argument('--force-download', action='store_true',
                      help='Force re-download of models')
    parser.add_argument('--skip-test', action='store_true',
                      help='Skip self-test')
    parser.add_argument('--verbose', action='store_true',
                      help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    display_welcome()
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    logger.info("\n🚀 Starting HRM initialization...")
    
    # Create directories
    logger.info("\n📁 Setting up directories...")
    create_directories()
    
    # Create configurations
    logger.info("\n⚙️ Creating configuration files...")
    create_default_configs()
    
    # Initialize model
    logger.info("\n🧠 Initializing HRM model...")
    initialize_hrm_model(force_new=args.force_download)
    
    # Create sample knowledge base
    logger.info("\n📚 Setting up knowledge base...")
    create_sample_knowledge_base()
    
    # Run self-test
    if not args.skip_test:
        if not run_self_test():
            logger.error("\n❌ Initialization failed!")
            sys.exit(1)
    
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║            ✅ Initialization Complete!               ║
    ║                                                      ║
    ║  Your system is ready to use ASI-GO-4-HRM.         ║
    ║                                                      ║
    ║  Quick start:                                        ║
    ║    python main.py --mode hybrid --goal "problem"    ║
    ║                                                      ║
    ║  Train HRM:                                          ║
    ║    python train_hrm.py --epochs 10                  ║
    ║                                                      ║
    ║  See README.md for detailed instructions.           ║
    ╚══════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()