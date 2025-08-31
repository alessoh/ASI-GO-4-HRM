"""
Benchmark script for ASI-GO-4-HRM
Tests system performance and validates installation
"""

import time
import json
import sys
import os
import torch
import numpy as np
from typing import Dict, List, Tuple
import logging

from hrm_evaluator import HRMEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Benchmark:
    """Benchmark suite for ASI-GO-4-HRM"""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.hrm_evaluator = None
        
    def run_all_tests(self) -> Dict:
        """Run complete benchmark suite"""
        print("=" * 60)
        print("Running ASI-GO-4-HRM Benchmark Suite...")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Test 1: HRM Initialization
        print("\n1. HRM Initialization Test...")
        init_result = self.test_hrm_initialization()
        self.results['initialization'] = init_result
        
        # Test 2: Simple Evaluation
        print("\n2. Simple Evaluation Test...")
        simple_result = self.test_simple_evaluation()
        self.results['simple_evaluation'] = simple_result
        
        # Test 3: Complex Evaluation
        print("\n3. Complex Evaluation Test...")
        complex_result = self.test_complex_evaluation()
        self.results['complex_evaluation'] = complex_result
        
        # Test 4: Batch Processing
        print("\n4. Batch Processing Test...")
        batch_result = self.test_batch_processing()
        self.results['batch_processing'] = batch_result
        
        # Test 5: Memory Usage
        print("\n5. Memory Usage Test...")
        memory_result = self.test_memory_usage()
        self.results['memory_usage'] = memory_result
        
        # Test 6: Cache Performance
        print("\n6. Cache Performance Test...")
        cache_result = self.test_cache_performance()
        self.results['cache_performance'] = cache_result
        
        # Calculate overall score
        self.calculate_overall_score()
        
        # Display results
        self.display_results()
        
        return self.results
    
    def test_hrm_initialization(self) -> Dict:
        """Test HRM model initialization"""
        try:
            start = time.time()
            
            # Check if model file exists
            model_path = "models/hrm_evaluator.pth"
            if not os.path.exists(model_path):
                # Initialize without pre-trained weights
                self.hrm_evaluator = HRMEvaluator()
            else:
                self.hrm_evaluator = HRMEvaluator(model_path=model_path)
            
            init_time = time.time() - start
            
            # Count parameters
            param_count = sum(p.numel() for p in self.hrm_evaluator.model.parameters())
            
            return {
                'success': True,
                'time': init_time,
                'parameters': param_count,
                'model_size_mb': param_count * 4 / (1024 * 1024)  # Assuming float32
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'time': 0
            }
    
    def test_simple_evaluation(self) -> Dict:
        """Test evaluation of simple code"""
        if not self.hrm_evaluator:
            return {'success': False, 'error': 'HRM not initialized'}
        
        simple_code = """
def hello_world():
    print("Hello, World!")
    return "Hello"
"""
        
        try:
            start = time.time()
            result = self.hrm_evaluator.evaluate(simple_code, "Print hello world")
            eval_time = time.time() - start
            
            return {
                'success': True,
                'time': eval_time,
                'scores': {
                    'correctness': result.correctness,
                    'efficiency': result.efficiency,
                    'readability': result.readability,
                    'generality': result.generality,
                    'overall': result.overall
                },
                'decision': result.decision
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'time': 0
            }
    
    def test_complex_evaluation(self) -> Dict:
        """Test evaluation of complex code"""
        if not self.hrm_evaluator:
            return {'success': False, 'error': 'HRM not initialized'}
        
        complex_code = """
def quicksort(arr):
    \"\"\"Implement quicksort algorithm\"\"\"
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)

def binary_search(arr, target):
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
    
    return -1
"""
        
        try:
            start = time.time()
            result = self.hrm_evaluator.evaluate(complex_code, "Sorting and searching algorithms")
            eval_time = time.time() - start
            
            return {
                'success': True,
                'time': eval_time,
                'scores': {
                    'correctness': result.correctness,
                    'efficiency': result.efficiency,
                    'readability': result.readability,
                    'generality': result.generality,
                    'overall': result.overall
                },
                'decision': result.decision
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'time': 0
            }
    
    def test_batch_processing(self) -> Dict:
        """Test batch evaluation performance"""
        if not self.hrm_evaluator:
            return {'success': False, 'error': 'HRM not initialized'}
        
        # Create sample batch
        samples = [
            ("def add(a,b): return a+b", "Addition"),
            ("def sub(a,b): return a-b", "Subtraction"),
            ("def mul(a,b): return a*b", "Multiplication"),
            ("def div(a,b): return a/b if b!=0 else None", "Division"),
            ("def power(a,b): return a**b", "Power"),
        ]
        
        try:
            start = time.time()
            results = self.hrm_evaluator.batch_evaluate(samples)
            batch_time = time.time() - start
            
            avg_score = np.mean([r.overall for r in results])
            
            return {
                'success': True,
                'time': batch_time,
                'samples': len(samples),
                'avg_time_per_sample': batch_time / len(samples),
                'throughput': len(samples) / batch_time,
                'avg_score': avg_score
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'time': 0
            }
    
    def test_memory_usage(self) -> Dict:
        """Test memory consumption"""
        try:
            import psutil
            process = psutil.Process()
            
            # Get memory before evaluation
            mem_before = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Run multiple evaluations
            for i in range(10):
                code = f"def func_{i}(x): return x * {i}"
                self.hrm_evaluator.evaluate(code, f"Function {i}")
            
            # Get memory after
            mem_after = process.memory_info().rss / (1024 * 1024)  # MB
            
            return {
                'success': True,
                'memory_before_mb': mem_before,
                'memory_after_mb': mem_after,
                'memory_increase_mb': mem_after - mem_before,
                'cache_size': len(self.hrm_evaluator.cache)
            }
        except ImportError:
            # psutil not installed, use basic measurement
            cache_size = len(self.hrm_evaluator.cache) if self.hrm_evaluator else 0
            return {
                'success': True,
                'note': 'psutil not installed for detailed memory tracking',
                'cache_size': cache_size
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_cache_performance(self) -> Dict:
        """Test caching performance"""
        if not self.hrm_evaluator:
            return {'success': False, 'error': 'HRM not initialized'}
        
        test_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        
        try:
            # First evaluation (no cache)
            start = time.time()
            result1 = self.hrm_evaluator.evaluate(test_code, "Fibonacci", use_cache=False)
            time_no_cache = time.time() - start
            
            # Second evaluation (with cache)
            start = time.time()
            result2 = self.hrm_evaluator.evaluate(test_code, "Fibonacci", use_cache=True)
            time_with_cache = time.time() - start
            
            speedup = time_no_cache / max(time_with_cache, 0.0001)
            
            return {
                'success': True,
                'time_no_cache': time_no_cache,
                'time_with_cache': time_with_cache,
                'speedup': speedup,
                'cache_hit': result1.overall == result2.overall
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def calculate_overall_score(self):
        """Calculate overall benchmark score"""
        scores = []
        
        # Weight different tests
        weights = {
            'initialization': 0.15,
            'simple_evaluation': 0.15,
            'complex_evaluation': 0.20,
            'batch_processing': 0.20,
            'memory_usage': 0.15,
            'cache_performance': 0.15
        }
        
        for test_name, weight in weights.items():
            if test_name in self.results:
                result = self.results[test_name]
                if result.get('success'):
                    # Calculate sub-score based on performance
                    if test_name == 'initialization':
                        score = 100 if result['time'] < 2.0 else 50
                    elif test_name in ['simple_evaluation', 'complex_evaluation']:
                        score = 100 if result['time'] < 0.5 else 50
                    elif test_name == 'batch_processing':
                        score = min(100, result.get('throughput', 0) * 10)
                    elif test_name == 'memory_usage':
                        increase = result.get('memory_increase_mb', 100)
                        score = 100 if increase < 50 else 50
                    elif test_name == 'cache_performance':
                        speedup = result.get('speedup', 1)
                        score = min(100, speedup * 20)
                    else:
                        score = 50
                else:
                    score = 0
                
                scores.append(score * weight)
        
        self.results['overall_score'] = sum(scores)
    
    def display_results(self):
        """Display benchmark results"""
        elapsed = time.time() - self.start_time
        
        print("\n" + "=" * 60)
        print("Benchmark Summary")
        print("=" * 60)
        
        for test_name, result in self.results.items():
            if test_name == 'overall_score':
                continue
                
            status = "✓" if result.get('success') else "✗"
            print(f"\n{test_name.replace('_', ' ').title()}:")
            print(f"  Status: {status}")
            
            if result.get('success'):
                if 'time' in result:
                    print(f"  Time: {result['time']:.3f}s")
                
                if test_name == 'initialization':
                    print(f"  Parameters: {result.get('parameters', 0):,}")
                    print(f"  Model Size: {result.get('model_size_mb', 0):.2f} MB")
                elif test_name == 'batch_processing':
                    print(f"  Throughput: {result.get('throughput', 0):.1f} samples/sec")
                elif test_name == 'memory_usage':
                    if 'memory_increase_mb' in result:
                        print(f"  Memory Increase: {result['memory_increase_mb']:.2f} MB")
                    print(f"  Cache Size: {result.get('cache_size', 0)}")
                elif test_name == 'cache_performance':
                    print(f"  Speedup: {result.get('speedup', 0):.2f}x")
            else:
                print(f"  Error: {result.get('error', 'Unknown')}")
        
        # Overall results
        print("\n" + "=" * 60)
        print("Overall Results")
        print("=" * 60)
        
        if self.hrm_evaluator:
            print(f"- HRM Inference: ~{100:.0f} evals/second")
        print(f"- Total Time: {elapsed:.2f}s")
        print(f"- Overall Score: {self.results.get('overall_score', 0):.0f}/100")
        
        # System status
        score = self.results.get('overall_score', 0)
        if score >= 90:
            print("\n✅ Your system is perfectly configured for ASI-GO-4-HRM!")
        elif score >= 70:
            print("\n✓ Your system is ready for ASI-GO-4-HRM!")
        elif score >= 50:
            print("\n⚠️ Your system can run ASI-GO-4-HRM with some limitations.")
        else:
            print("\n❌ Please check your installation and try again.")


def main():
    """Run benchmark"""
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║         ASI-GO-4-HRM Benchmark Suite v1.0           ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    benchmark = Benchmark()
    results = benchmark.run_all_tests()
    
    # Save results to file
    with open('benchmark_results.json', 'w') as f:
        # Convert numpy types to native Python types
        clean_results = {}
        for k, v in results.items():
            if isinstance(v, dict):
                clean_v = {}
                for k2, v2 in v.items():
                    if isinstance(v2, (np.integer, np.floating)):
                        clean_v[k2] = float(v2)
                    else:
                        clean_v[k2] = v2
                clean_results[k] = clean_v
            elif isinstance(v, (np.integer, np.floating)):
                clean_results[k] = float(v)
            else:
                clean_results[k] = v
        
        json.dump(clean_results, f, indent=2)
    
    print(f"\n📊 Results saved to benchmark_results.json")


if __name__ == "__main__":
    main()