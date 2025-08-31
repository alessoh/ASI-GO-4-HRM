"""
Enhanced Engineer Module with HRM Integration
Combines HRM evaluation with LLM fallback for robust code assessment
"""

import re
import json
import subprocess
import tempfile
import os
import sys
import time
import logging
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

from engineer import Engineer  # Import original Engineer
from hrm_evaluator import HRMEvaluator, EvaluationResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HybridEvaluationResult:
    """Combined evaluation result from HRM and potentially LLM"""
    hrm_result: EvaluationResult
    llm_result: Optional[Dict[str, Any]] = None
    final_decision: str = ""
    final_score: float = 0.0
    evaluation_time: float = 0.0
    used_llm: bool = False


class EngineerHRM(Engineer):
    """Enhanced Engineer that uses HRM for primary evaluation"""
    
    def __init__(self, llm_interface=None, verbose: bool = False):
        super().__init__(llm_interface, verbose)
        
        # Initialize HRM evaluator
        self.hrm_evaluator = HRMEvaluator(
            model_path="models/hrm_evaluator.pth" if os.path.exists("models/hrm_evaluator.pth") else None
        )
        
        # Statistics tracking
        self.stats = {
            'hrm_evaluations': 0,
            'llm_evaluations': 0,
            'hrm_accepts': 0,
            'llm_accepts': 0,
            'total_time': 0.0,
            'hrm_time': 0.0,
            'llm_time': 0.0
        }
        
        # Mode configuration
        self.evaluation_mode = "hybrid"  # "hybrid", "hrm", "classic"
        
    def set_evaluation_mode(self, mode: str):
        """Set evaluation mode: hybrid, hrm, or classic"""
        if mode not in ["hybrid", "hrm", "classic"]:
            raise ValueError(f"Invalid mode: {mode}")
        self.evaluation_mode = mode
        logger.info(f"Evaluation mode set to: {mode}")
    
    def evaluate_solution(self, code: str, problem_description: str = "", 
                         test_cases: Optional[List[Dict]] = None) -> HybridEvaluationResult:
        """
        Evaluate solution using HRM with optional LLM fallback
        """
        start_time = time.time()
        
        # First, check if code executes without errors
        execution_result = self.test_execution(code)
        if not execution_result['success']:
            # Quick rejection for code that doesn't run
            hrm_result = EvaluationResult(
                correctness=0.0,
                efficiency=0.5,
                readability=0.5,
                generality=0.5,
                overall=0.25,
                confidence=1.0,
                decision='REJECT',
                reasoning={'error': execution_result.get('error', 'Execution failed')}
            )
            return HybridEvaluationResult(
                hrm_result=hrm_result,
                final_decision='REJECT',
                final_score=0.0,
                evaluation_time=time.time() - start_time,
                used_llm=False
            )
        
        # Mode-specific evaluation
        if self.evaluation_mode == "classic":
            return self._evaluate_classic(code, problem_description, execution_result, start_time)
        elif self.evaluation_mode == "hrm":
            return self._evaluate_hrm_only(code, problem_description, execution_result, start_time)
        else:  # hybrid
            return self._evaluate_hybrid(code, problem_description, execution_result, start_time)
    
    def _evaluate_hrm_only(self, code: str, problem_description: str, 
                          execution_result: Dict, start_time: float) -> HybridEvaluationResult:
        """Evaluate using only HRM"""
        hrm_start = time.time()
        
        # Get HRM evaluation
        hrm_result = self.hrm_evaluator.evaluate(code, problem_description)
        
        hrm_time = time.time() - hrm_start
        self.stats['hrm_evaluations'] += 1
        self.stats['hrm_time'] += hrm_time
        
        # Adjust score based on execution
        if execution_result['success']:
            final_score = hrm_result.overall
        else:
            final_score = hrm_result.overall * 0.5
        
        # Make final decision
        if final_score >= 0.75:
            final_decision = 'ACCEPT'
            self.stats['hrm_accepts'] += 1
        else:
            final_decision = 'REJECT'
        
        if self.verbose:
            self._print_hrm_evaluation(hrm_result)
        
        return HybridEvaluationResult(
            hrm_result=hrm_result,
            final_decision=final_decision,
            final_score=final_score,
            evaluation_time=time.time() - start_time,
            used_llm=False
        )
    
    def _evaluate_hybrid(self, code: str, problem_description: str,
                        execution_result: Dict, start_time: float) -> HybridEvaluationResult:
        """Evaluate using HRM with LLM fallback"""
        hrm_start = time.time()
        
        # Get HRM evaluation
        hrm_result = self.hrm_evaluator.evaluate(code, problem_description)
        
        hrm_time = time.time() - hrm_start
        self.stats['hrm_evaluations'] += 1
        self.stats['hrm_time'] += hrm_time
        
        if self.verbose:
            self._print_hrm_evaluation(hrm_result)
        
        # Decide if LLM evaluation is needed
        if hrm_result.decision == 'ACCEPT' and hrm_result.confidence > 0.9:
            # High confidence accept
            final_decision = 'ACCEPT'
            final_score = hrm_result.overall
            used_llm = False
            self.stats['hrm_accepts'] += 1
            
        elif hrm_result.decision == 'REJECT' and hrm_result.confidence > 0.9:
            # High confidence reject
            final_decision = 'REJECT'
            final_score = hrm_result.overall
            used_llm = False
            
        else:
            # Need LLM verification
            if self.verbose:
                print("🔄 HRM uncertain, consulting LLM...")
            
            llm_start = time.time()
            llm_result = self._get_llm_evaluation(code, problem_description)
            llm_time = time.time() - llm_start
            
            self.stats['llm_evaluations'] += 1
            self.stats['llm_time'] += llm_time
            used_llm = True
            
            # Combine evaluations
            if llm_result and llm_result.get('valid'):
                # Weight LLM more heavily for uncertain cases
                final_score = 0.3 * hrm_result.overall + 0.7 * llm_result.get('score', 0.5)
                final_decision = 'ACCEPT' if final_score >= 0.7 else 'REJECT'
                if final_decision == 'ACCEPT':
                    self.stats['llm_accepts'] += 1
            else:
                # Default to HRM decision if LLM fails
                final_decision = 'ACCEPT' if hrm_result.overall >= 0.75 else 'REJECT'
                final_score = hrm_result.overall
            
            return HybridEvaluationResult(
                hrm_result=hrm_result,
                llm_result=llm_result,
                final_decision=final_decision,
                final_score=final_score,
                evaluation_time=time.time() - start_time,
                used_llm=used_llm
            )
        
        return HybridEvaluationResult(
            hrm_result=hrm_result,
            final_decision=final_decision,
            final_score=final_score,
            evaluation_time=time.time() - start_time,
            used_llm=used_llm
        )
    
    def _evaluate_classic(self, code: str, problem_description: str,
                         execution_result: Dict, start_time: float) -> HybridEvaluationResult:
        """Evaluate using only LLM (classic mode)"""
        llm_result = self._get_llm_evaluation(code, problem_description)
        
        self.stats['llm_evaluations'] += 1
        
        # Create dummy HRM result for consistency
        hrm_result = EvaluationResult(
            correctness=0.5,
            efficiency=0.5,
            readability=0.5,
            generality=0.5,
            overall=0.5,
            confidence=0.0,
            decision='NEEDS_LLM',
            reasoning={'mode': 'Classic LLM-only evaluation'}
        )
        
        if llm_result and llm_result.get('valid'):
            final_decision = 'ACCEPT'
            final_score = llm_result.get('score', 0.75)
            self.stats['llm_accepts'] += 1
        else:
            final_decision = 'REJECT'
            final_score = 0.3
        
        return HybridEvaluationResult(
            hrm_result=hrm_result,
            llm_result=llm_result,
            final_decision=final_decision,
            final_score=final_score,
            evaluation_time=time.time() - start_time,
            used_llm=True
        )
    
    def _get_llm_evaluation(self, code: str, problem_description: str) -> Dict[str, Any]:
        """Get evaluation from LLM"""
        if not self.llm_interface:
            return {'valid': False, 'reason': 'No LLM configured'}
        
        try:
            # Use parent class method for LLM evaluation
            result = self.judge_code(code, problem_description)
            
            # Convert to standardized format
            return {
                'valid': result.get('is_valid', False),
                'score': result.get('fitness_score', 0.5),
                'reasoning': result.get('reasoning', ''),
                'improvements': result.get('improvements', [])
            }
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            return {'valid': False, 'reason': str(e)}
    
    def test_execution(self, code: str) -> Dict[str, Any]:
        """Test if code executes without errors"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            os.unlink(temp_file)
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'error': result.stderr if result.returncode != 0 else None
            }
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Execution timeout'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _print_hrm_evaluation(self, result: EvaluationResult):
        """Print HRM evaluation results"""
        print("\n📊 HRM Evaluation Results:")
        print(f"├── Correctness: {result.correctness:.2f}")
        print(f"├── Efficiency:  {result.efficiency:.2f}")
        print(f"├── Readability: {result.readability:.2f}")
        print(f"├── Generality:  {result.generality:.2f}")
        print(f"├── Overall:     {result.overall:.2f}")
        print(f"├── Confidence:  {result.confidence:.2f}")
        print(f"└── Decision:    {result.decision}")
        
        if result.reasoning:
            print("\n💭 Reasoning:")
            for aspect, reason in result.reasoning.items():
                print(f"  • {aspect}: {reason}")
    
    def train_hrm_from_history(self, knowledge_base_path: str = "knowledge_base.json"):
        """Train HRM using historical solutions from knowledge base"""
        try:
            with open(knowledge_base_path, 'r') as f:
                knowledge_base = json.load(f)
            
            training_data = []
            for problem, data in knowledge_base.items():
                if 'solution' in data and 'fitness_score' in data:
                    training_data.append({
                        'code': data['solution'],
                        'score': data['fitness_score'],
                        'problem': problem
                    })
            
            if training_data:
                logger.info(f"Training HRM on {len(training_data)} examples...")
                # This would connect to the actual training logic
                # For now, just update the model based on feedback
                for sample in training_data:
                    self.hrm_evaluator.update_from_feedback(
                        sample['code'],
                        sample['score'],
                        sample['problem']
                    )
                logger.info("HRM training completed")
            else:
                logger.warning("No training data found in knowledge base")
                
        except FileNotFoundError:
            logger.error(f"Knowledge base not found: {knowledge_base_path}")
        except Exception as e:
            logger.error(f"Training failed: {e}")
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get statistics about evaluation performance"""
        total_evals = self.stats['hrm_evaluations'] + self.stats['llm_evaluations']
        if total_evals == 0:
            return {'message': 'No evaluations performed yet'}
        
        hrm_only = self.stats['hrm_evaluations'] - self.stats.get('llm_evaluations', 0)
        
        stats = {
            'total_evaluations': total_evals,
            'hrm_evaluations': self.stats['hrm_evaluations'],
            'llm_evaluations': self.stats['llm_evaluations'],
            'hrm_only_evaluations': max(0, hrm_only),
            'hrm_accept_rate': self.stats['hrm_accepts'] / max(self.stats['hrm_evaluations'], 1),
            'llm_accept_rate': self.stats['llm_accepts'] / max(self.stats['llm_evaluations'], 1),
            'avg_hrm_time': self.stats['hrm_time'] / max(self.stats['hrm_evaluations'], 1),
            'avg_llm_time': self.stats['llm_time'] / max(self.stats['llm_evaluations'], 1),
            'llm_usage_rate': self.stats['llm_evaluations'] / total_evals,
            'cost_reduction': 1.0 - (self.stats['llm_evaluations'] / total_evals)
        }
        
        return stats
    
    def clear_cache(self):
        """Clear HRM evaluation cache"""
        self.hrm_evaluator.cache.clear()
        logger.info("HRM cache cleared")