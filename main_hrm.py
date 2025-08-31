"""
Enhanced Main Module for ASI-GO-4-HRM
Integrates HRM evaluation into the self-improving AI system
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional

from colorama import init, Fore, Style
init(autoreset=True)

# Import original modules
from cognition_base import CognitionBase
from researcher import Researcher
from analyst import Analyst
from llm_interface import LLMInterface

# Import HRM-enhanced engineer
from engineer_hrm import EngineerHRM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ASIGO4HRM:
    """Main orchestrator for ASI-GO-4-HRM system"""
    
    def __init__(self, config_path: str = "config.json", mode: str = "hybrid"):
        self.config = self.load_config(config_path)
        self.mode = mode  # "hybrid", "hrm", or "classic"
        
        # Initialize LLM interface
        self.llm_interface = LLMInterface(self.config)
        
        # Initialize modules
        self.cognition_base = CognitionBase()
        self.researcher = Researcher(self.llm_interface, verbose=self.config.get('verbose', False))
        self.engineer = EngineerHRM(self.llm_interface, verbose=self.config.get('verbose', False))
        self.analyst = Analyst(self.llm_interface, verbose=self.config.get('verbose', False))
        
        # Set evaluation mode
        self.engineer.set_evaluation_mode(mode)
        
        # Statistics
        self.stats = {
            'start_time': None,
            'iterations': 0,
            'llm_calls': 0,
            'hrm_evaluations': 0,
            'solutions_found': 0
        }
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {
                'max_attempts': 5,
                'temperature': 0.7,
                'verbose': False
            }
    
    def solve(self, goal: str, max_attempts: Optional[int] = None) -> Dict[str, Any]:
        """Main solving loop with HRM integration"""
        self.stats['start_time'] = time.time()
        max_attempts = max_attempts or self.config.get('max_attempts', 5)
        
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}🚀 ASI-GO-4-HRM Starting (Mode: {self.mode.upper()})")
        print(f"{Fore.CYAN}{'='*60}")
        print(f"{Fore.WHITE}Goal: {goal}")
        print(f"{Fore.WHITE}Max Attempts: {max_attempts}")
        print(f"{Fore.WHITE}Evaluation Mode: {self.mode}")
        print(f"{Fore.CYAN}{'='*60}\n")
        
        # Check cognition base for existing solution
        existing_solution = self.cognition_base.retrieve(goal)
        if existing_solution and existing_solution.get('solution'):
            print(f"{Fore.GREEN}✓ Found existing solution in knowledge base!")
            if self.config.get('verbose'):
                print(f"Solution: {existing_solution['solution'][:200]}...")
            return existing_solution
        
        best_solution = None
        best_score = 0.0
        
        for attempt in range(max_attempts):
            self.stats['iterations'] += 1
            print(f"\n{Fore.YELLOW}Attempt {attempt + 1}/{max_attempts}")
            print(f"{Fore.YELLOW}{'-'*40}")
            
            # Research phase
            print(f"{Fore.BLUE}🔬 Researcher: Generating solution...")
            proposal = self.researcher.propose_solution(goal)
            
            if not proposal or 'code' not in proposal:
                print(f"{Fore.RED}✗ Failed to generate proposal")
                continue
            
            # Engineering phase with HRM
            print(f"{Fore.BLUE}⚙️ Engineer: Evaluating solution...")
            eval_start = time.time()
            
            eval_result = self.engineer.evaluate_solution(
                proposal['code'],
                goal,
                test_cases=None
            )
            
            eval_time = time.time() - eval_start
            self.stats['hrm_evaluations'] += 1
            
            # Display evaluation results
            self._display_evaluation(eval_result, eval_time)
            
            # Check if solution is accepted
            if eval_result.final_decision == 'ACCEPT':
                print(f"{Fore.GREEN}✓ Solution accepted!")
                
                # Analysis phase
                print(f"{Fore.BLUE}📊 Analyst: Extracting insights...")
                analysis = self.analyst.analyze_result({
                    'code': proposal['code'],
                    'goal': goal,
                    'score': eval_result.final_score,
                    'hrm_scores': {
                        'correctness': eval_result.hrm_result.correctness,
                        'efficiency': eval_result.hrm_result.efficiency,
                        'readability': eval_result.hrm_result.readability,
                        'generality': eval_result.hrm_result.generality
                    }
                })
                
                # Store in cognition base
                self.cognition_base.store(goal, {
                    'solution': proposal['code'],
                    'fitness_score': eval_result.final_score,
                    'hrm_evaluation': eval_result.hrm_result.__dict__,
                    'analysis': analysis,
                    'timestamp': datetime.now().isoformat()
                })
                
                self.stats['solutions_found'] += 1
                
                # Update HRM with feedback
                if self.mode != "classic":
                    self.engineer.hrm_evaluator.update_from_feedback(
                        proposal['code'],
                        eval_result.final_score,
                        goal
                    )
                
                return {
                    'success': True,
                    'solution': proposal['code'],
                    'score': eval_result.final_score,
                    'evaluation': eval_result,
                    'analysis': analysis,
                    'attempts': attempt + 1
                }
            
            # Track best solution even if not accepted
            if eval_result.final_score > best_score:
                best_score = eval_result.final_score
                best_solution = proposal['code']
            
            # Refine for next attempt
            if attempt < max_attempts - 1:
                print(f"{Fore.YELLOW}🔄 Refining approach based on feedback...")
                
                # Use HRM reasoning for refinement hints
                refinement_hints = self._generate_refinement_hints(eval_result)
                self.researcher.refine_approach(refinement_hints)
        
        # Return best solution found
        print(f"\n{Fore.YELLOW}⚠️ Max attempts reached")
        if best_solution:
            print(f"{Fore.YELLOW}Returning best solution (score: {best_score:.2f})")
            return {
                'success': False,
                'solution': best_solution,
                'score': best_score,
                'attempts': max_attempts,
                'partial': True
            }
        
        print(f"{Fore.RED}✗ No valid solution found")
        return {
            'success': False,
            'attempts': max_attempts
        }
    
    def _display_evaluation(self, eval_result, eval_time):
        """Display evaluation results in a formatted way"""
        hrm = eval_result.hrm_result
        
        if self.config.get('verbose') or self.mode != "classic":
            print(f"\n{Fore.CYAN}📊 HRM Evaluation ({eval_time:.2f}s):")
            print(f"  {Fore.WHITE}├── Correctness: {self._score_bar(hrm.correctness)}")
            print(f"  {Fore.WHITE}├── Efficiency:  {self._score_bar(hrm.efficiency)}")
            print(f"  {Fore.WHITE}├── Readability: {self._score_bar(hrm.readability)}")
            print(f"  {Fore.WHITE}├── Generality:  {self._score_bar(hrm.generality)}")
            print(f"  {Fore.WHITE}├── Overall:     {self._score_bar(hrm.overall)}")
            print(f"  {Fore.WHITE}└── Confidence:  {self._score_bar(hrm.confidence)}")
        
        if eval_result.used_llm:
            print(f"\n{Fore.YELLOW}🤖 LLM verification was used")
        
        # Decision coloring
        if eval_result.final_decision == 'ACCEPT':
            color = Fore.GREEN
            symbol = "✓"
        else:
            color = Fore.RED
            symbol = "✗"
        
        print(f"\n{color}{symbol} Decision: {eval_result.final_decision} (Score: {eval_result.final_score:.2f})")
    
    def _score_bar(self, score: float, width: int = 20) -> str:
        """Create a visual bar for scores"""
        filled = int(score * width)
        bar = '█' * filled + '░' * (width - filled)
        
        if score >= 0.8:
            color = Fore.GREEN
        elif score >= 0.6:
            color = Fore.YELLOW
        else:
            color = Fore.RED
        
        return f"{color}{bar}{Style.RESET_ALL} {score:.2f}"
    
    def _generate_refinement_hints(self, eval_result) -> Dict[str, Any]:
        """Generate refinement hints based on HRM evaluation"""
        hints = {}
        hrm = eval_result.hrm_result
        
        if hrm.correctness < 0.7:
            hints['focus'] = "syntax and logic correctness"
        elif hrm.efficiency < 0.7:
            hints['focus'] = "algorithm efficiency and optimization"
        elif hrm.readability < 0.7:
            hints['focus'] = "code clarity and documentation"
        elif hrm.generality < 0.7:
            hints['focus'] = "edge case handling and robustness"
        else:
            hints['focus'] = "minor improvements and polish"
        
        hints['weak_areas'] = []
        if hrm.correctness < 0.8:
            hints['weak_areas'].append('correctness')
        if hrm.efficiency < 0.8:
            hints['weak_areas'].append('efficiency')
        if hrm.readability < 0.8:
            hints['weak_areas'].append('readability')
        if hrm.generality < 0.8:
            hints['weak_areas'].append('generality')
        
        hints['reasoning'] = hrm.reasoning
        
        return hints
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if self.stats['start_time']:
            elapsed = time.time() - self.stats['start_time']
        else:
            elapsed = 0
        
        eng_stats = self.engineer.get_evaluation_stats()
        
        return {
            'execution_time': elapsed,
            'iterations': self.stats['iterations'],
            'solutions_found': self.stats['solutions_found'],
            'evaluation_stats': eng_stats,
            'mode': self.mode,
            'cost_savings': eng_stats.get('cost_reduction', 0) if eng_stats else 0
        }
    
    def train_hrm(self):
        """Train HRM on knowledge base"""
        print(f"\n{Fore.CYAN}🎓 Training HRM on knowledge base...")
        self.engineer.train_hrm_from_history()
        print(f"{Fore.GREEN}✓ Training complete")


def main():
    parser = argparse.ArgumentParser(description='ASI-GO-4-HRM: Self-Improving AI with HRM')
    parser.add_argument('--goal', type=str, required=True,
                      help='Problem to solve')
    parser.add_argument('--mode', type=str, default='hybrid',
                      choices=['hybrid', 'hrm', 'classic'],
                      help='Evaluation mode')
    parser.add_argument('--attempts', type=int, default=5,
                      help='Maximum solution attempts')
    parser.add_argument('--train', action='store_true',
                      help='Train HRM before solving')
    parser.add_argument('--verbose', action='store_true',
                      help='Verbose output')
    parser.add_argument('--save-training-data', action='store_true',
                      help='Save data for HRM training')
    parser.add_argument('--config', type=str, default='config.json',
                      help='Configuration file path')
    
    args = parser.parse_args()
    
    # Update config with command-line arguments
    if args.verbose:
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config = json.load(f)
            config['verbose'] = True
            with open(args.config, 'w') as f:
                json.dump(config, f, indent=2)
    
    # Initialize system
    system = ASIGO4HRM(config_path=args.config, mode=args.mode)
    
    # Train HRM if requested
    if args.train:
        system.train_hrm()
    
    # Solve the problem
    result = system.solve(args.goal, max_attempts=args.attempts)
    
    # Display results
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}📈 Final Results")
    print(f"{Fore.CYAN}{'='*60}")
    
    if result.get('success'):
        print(f"{Fore.GREEN}✅ Solution found successfully!")
        print(f"{Fore.WHITE}Score: {result.get('score', 0):.2f}")
        print(f"{Fore.WHITE}Attempts: {result.get('attempts', 0)}")
        
        if args.verbose:
            print(f"\n{Fore.CYAN}Solution:")
            print(f"{Fore.WHITE}{result.get('solution', 'N/A')}")
    else:
        if result.get('partial'):
            print(f"{Fore.YELLOW}⚠️ Partial solution found")
            print(f"{Fore.WHITE}Best Score: {result.get('score', 0):.2f}")
        else:
            print(f"{Fore.RED}❌ No solution found")
    
    # Display statistics
    stats = system.get_statistics()
    print(f"\n{Fore.CYAN}📊 Statistics:")
    print(f"{Fore.WHITE}├── Execution Time: {stats['execution_time']:.2f}s")
    print(f"{Fore.WHITE}├── Iterations: {stats['iterations']}")
    print(f"{Fore.WHITE}├── Evaluation Mode: {stats['mode']}")
    
    if stats.get('evaluation_stats'):
        e_stats = stats['evaluation_stats']
        if e_stats.get('llm_usage_rate') is not None:
            print(f"{Fore.WHITE}├── LLM Usage Rate: {e_stats['llm_usage_rate']:.1%}")
            print(f"{Fore.WHITE}└── Cost Reduction: {e_stats.get('cost_reduction', 0):.1%}")
    
    # Save training data if requested
    if args.save_training_data and result.get('solution'):
        training_entry = {
            'timestamp': datetime.now().isoformat(),
            'goal': args.goal,
            'solution': result.get('solution'),
            'score': result.get('score', 0),
            'mode': args.mode
        }
        
        training_file = 'training_data.jsonl'
        with open(training_file, 'a') as f:
            f.write(json.dumps(training_entry) + '\n')
        print(f"\n{Fore.GREEN}✓ Training data saved to {training_file}")


if __name__ == "__main__":
    main()