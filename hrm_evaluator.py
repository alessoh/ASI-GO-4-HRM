"""
HRM Evaluator Module for ASI-GO-4-HRM
Implements Hierarchical Reasoning Model for code evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import re
import ast
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Multi-dimensional evaluation result from HRM"""
    correctness: float
    efficiency: float
    readability: float
    generality: float
    overall: float
    confidence: float
    decision: str  # 'ACCEPT', 'REJECT', 'NEEDS_LLM'
    reasoning: Dict[str, str]


class LowLevelModule(nn.Module):
    """Fast processing module for local pattern recognition"""
    
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Pattern recognition layers
        self.pattern_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        ])
        
        # Specialized heads for different aspects
        self.syntax_head = nn.Linear(hidden_size, 64)
        self.pattern_head = nn.Linear(hidden_size, 64)
        self.complexity_head = nn.Linear(hidden_size, 32)
        
    def forward(self, x: torch.Tensor, h_context: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Process input with high-level context"""
        # Combine input with context
        combined = x + 0.5 * h_context
        
        # Process through pattern layers
        for layer in self.pattern_layers:
            if isinstance(layer, nn.Linear):
                combined = layer(combined)
            else:
                combined = layer(combined)
        
        # Extract specific features
        syntax_features = torch.relu(self.syntax_head(combined))
        pattern_features = torch.relu(self.pattern_head(combined))
        complexity_features = torch.relu(self.complexity_head(combined))
        
        features = {
            'syntax': syntax_features,
            'patterns': pattern_features,
            'complexity': complexity_features
        }
        
        return combined, features


class HighLevelModule(nn.Module):
    """Slow processing module for global strategy assessment"""
    
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Strategy assessment layers
        self.strategy_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        ])
        
        # Global assessment heads
        self.overall_head = nn.Linear(hidden_size, 128)
        self.confidence_head = nn.Linear(hidden_size, 1)
        
    def forward(self, h_state: torch.Tensor, l_output: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Update state based on low-level findings"""
        # Integrate low-level output
        combined = h_state + l_output
        
        # Process through strategy layers
        for layer in self.strategy_layers:
            if isinstance(layer, nn.Linear):
                combined = layer(combined)
            else:
                combined = layer(combined)
        
        # Generate overall assessment
        overall_features = self.overall_head(combined)
        confidence = torch.sigmoid(self.confidence_head(combined))
        
        return combined, confidence.item()


class HierarchicalReasoningModel(nn.Module):
    """Main HRM model for code evaluation"""
    
    def __init__(self, input_size: int = 512, hidden_size: int = 256):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input encoding
        self.code_encoder = CodeEncoder(input_size, hidden_size)
        
        # Hierarchical modules
        self.L_module = LowLevelModule(hidden_size)
        self.H_module = HighLevelModule(hidden_size)
        
        # Evaluation heads
        self.correctness_head = nn.Linear(hidden_size, 1)
        self.efficiency_head = nn.Linear(hidden_size, 1)
        self.readability_head = nn.Linear(hidden_size, 1)
        self.generality_head = nn.Linear(hidden_size, 1)
        
        # Learnable initial states
        self.L_init = nn.Parameter(torch.randn(1, hidden_size) * 0.01)
        self.H_init = nn.Parameter(torch.randn(1, hidden_size) * 0.01)
        
    def forward(self, code_features: torch.Tensor, n_cycles: int = 4, t_steps: int = 4) -> Dict[str, float]:
        """Hierarchical reasoning for code evaluation"""
        batch_size = code_features.shape[0]
        
        # Initialize states
        z_L = self.L_init.expand(batch_size, -1)
        z_H = self.H_init.expand(batch_size, -1)
        
        # Store intermediate features for analysis
        all_features = []
        
        # Hierarchical computation
        for n in range(n_cycles):
            cycle_features = []
            
            # Low-level convergence
            for t in range(t_steps):
                z_L, features = self.L_module(code_features, z_H)
                cycle_features.append(features)
            
            # High-level update
            z_H, confidence = self.H_module(z_H, z_L)
            all_features.append((cycle_features, confidence))
        
        # Generate evaluation scores
        scores = {
            'correctness': torch.sigmoid(self.correctness_head(z_H)).item(),
            'efficiency': torch.sigmoid(self.efficiency_head(z_H)).item(),
            'readability': torch.sigmoid(self.readability_head(z_H)).item(),
            'generality': torch.sigmoid(self.generality_head(z_H)).item(),
            'confidence': confidence
        }
        
        return scores, all_features


class CodeEncoder(nn.Module):
    """Encode code into feature vectors for HRM processing"""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Multiple encoding strategies
        self.token_embed = nn.Embedding(10000, 64)  # For token-based encoding
        self.ast_embed = nn.Linear(100, 64)  # For AST features
        self.metric_embed = nn.Linear(20, 64)  # For code metrics
        
        # Fusion layer
        self.fusion = nn.Linear(192, hidden_size)
        
    def forward(self, code_str: str) -> torch.Tensor:
        """Convert code string to feature tensor"""
        # Extract different representations
        token_features = self.encode_tokens(code_str)
        ast_features = self.encode_ast(code_str)
        metric_features = self.encode_metrics(code_str)
        
        # Combine features
        combined = torch.cat([token_features, ast_features, metric_features], dim=-1)
        encoded = self.fusion(combined)
        
        return encoded
    
    def encode_tokens(self, code: str) -> torch.Tensor:
        """Simple token-based encoding"""
        # Tokenize code (simplified)
        tokens = re.findall(r'\w+|[^\w\s]', code)[:100]  # Max 100 tokens
        token_ids = [hash(token) % 10000 for token in tokens]
        
        # Pad to fixed length
        token_ids = token_ids[:100] + [0] * (100 - len(token_ids))
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        
        # Embed and average
        embedded = self.token_embed(token_tensor)
        return embedded.mean(dim=0, keepdim=True)
    
    def encode_ast(self, code: str) -> torch.Tensor:
        """Extract AST-based features"""
        features = torch.zeros(1, 100)
        
        try:
            tree = ast.parse(code)
            
            # Count different node types
            node_counts = {}
            for node in ast.walk(tree):
                node_type = type(node).__name__
                node_counts[node_type] = node_counts.get(node_type, 0) + 1
            
            # Convert to feature vector
            for i, (node_type, count) in enumerate(node_counts.items()):
                if i < 100:
                    features[0, i] = min(count / 10.0, 1.0)  # Normalize
                    
        except:
            # If parsing fails, use zeros
            pass
            
        return self.ast_embed(features)
    
    def encode_metrics(self, code: str) -> torch.Tensor:
        """Extract code quality metrics"""
        metrics = []
        
        # Line count
        lines = code.split('\n')
        metrics.append(min(len(lines) / 100.0, 1.0))
        
        # Average line length
        avg_length = np.mean([len(line) for line in lines]) if lines else 0
        metrics.append(min(avg_length / 80.0, 1.0))
        
        # Function count
        function_count = len(re.findall(r'def \w+', code))
        metrics.append(min(function_count / 10.0, 1.0))
        
        # Class count
        class_count = len(re.findall(r'class \w+', code))
        metrics.append(min(class_count / 5.0, 1.0))
        
        # Loop complexity
        loop_count = len(re.findall(r'(for |while )', code))
        metrics.append(min(loop_count / 10.0, 1.0))
        
        # Conditional complexity
        if_count = len(re.findall(r'if |elif |else:', code))
        metrics.append(min(if_count / 15.0, 1.0))
        
        # Comment ratio
        comment_lines = len([l for l in lines if l.strip().startswith('#')])
        comment_ratio = comment_lines / max(len(lines), 1)
        metrics.append(comment_ratio)
        
        # Import count
        import_count = len(re.findall(r'^import |^from ', code, re.MULTILINE))
        metrics.append(min(import_count / 10.0, 1.0))
        
        # Try-except blocks
        try_count = len(re.findall(r'try:', code))
        metrics.append(min(try_count / 5.0, 1.0))
        
        # List/dict comprehensions (pythonic)
        comp_count = len(re.findall(r'\[.+for .+in .+\]|\{.+for .+in .+\}', code))
        metrics.append(min(comp_count / 5.0, 1.0))
        
        # Pad to 20 metrics
        metrics.extend([0.0] * (20 - len(metrics)))
        
        metric_tensor = torch.tensor(metrics[:20], dtype=torch.float32).unsqueeze(0)
        return self.metric_embed(metric_tensor)


class HRMEvaluator:
    """Main evaluator class that uses HRM for code assessment"""
    
    def __init__(self, model_path: Optional[str] = None, config_path: str = "hrm_config.json"):
        self.device = torch.device("cpu")  # CPU only
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize model
        self.model = HierarchicalReasoningModel(
            input_size=self.config.get('input_size', 512),
            hidden_size=self.config.get('hidden_size', 256)
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Load pre-trained weights if available
        if model_path:
            self.load_model(model_path)
        
        # Evaluation weights
        self.eval_weights = {
            'correctness': 0.40,
            'efficiency': 0.25,
            'readability': 0.20,
            'generality': 0.15
        }
        
        # Cache for repeated evaluations
        self.cache = {}
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {
                'hidden_size': 256,
                'n_cycles': 4,
                't_steps': 4,
                'confidence_threshold': 0.85,
                'fallback_to_llm': True
            }
    
    def load_model(self, model_path: str):
        """Load pre-trained model weights"""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded HRM model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def save_model(self, model_path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Saved HRM model to {model_path}")
    
    def evaluate(self, code: str, problem_description: str = "", use_cache: bool = True) -> EvaluationResult:
        """Evaluate code using HRM"""
        # Check cache
        cache_key = hash(code + problem_description)
        if use_cache and cache_key in self.cache:
            logger.info("Using cached evaluation")
            return self.cache[cache_key]
        
        # Encode code
        with torch.no_grad():
            code_features = self.model.code_encoder(code)
            
            # Run hierarchical reasoning
            scores, features = self.model(
                code_features,
                n_cycles=self.config.get('n_cycles', 4),
                t_steps=self.config.get('t_steps', 4)
            )
        
        # Calculate overall score
        overall = sum(scores[k] * self.eval_weights[k] 
                     for k in ['correctness', 'efficiency', 'readability', 'generality'])
        
        # Determine decision
        confidence = scores['confidence']
        if overall >= 0.80 and confidence >= self.config.get('confidence_threshold', 0.85):
            decision = 'ACCEPT'
        elif overall < 0.60:
            decision = 'REJECT'
        else:
            decision = 'NEEDS_LLM' if self.config.get('fallback_to_llm', True) else 'REJECT'
        
        # Generate reasoning
        reasoning = self.generate_reasoning(scores, features)
        
        # Create result
        result = EvaluationResult(
            correctness=scores['correctness'],
            efficiency=scores['efficiency'],
            readability=scores['readability'],
            generality=scores['generality'],
            overall=overall,
            confidence=confidence,
            decision=decision,
            reasoning=reasoning
        )
        
        # Cache result
        if use_cache:
            self.cache[cache_key] = result
        
        return result
    
    def generate_reasoning(self, scores: Dict[str, float], features: List) -> Dict[str, str]:
        """Generate human-readable reasoning from HRM features"""
        reasoning = {}
        
        # Correctness reasoning
        if scores['correctness'] < 0.7:
            reasoning['correctness'] = "Code may have syntax errors or logical issues"
        elif scores['correctness'] < 0.9:
            reasoning['correctness'] = "Code appears mostly correct with minor concerns"
        else:
            reasoning['correctness'] = "Code structure and logic appear sound"
        
        # Efficiency reasoning
        if scores['efficiency'] < 0.6:
            reasoning['efficiency'] = "Potential performance issues detected"
        elif scores['efficiency'] < 0.8:
            reasoning['efficiency'] = "Acceptable efficiency with room for optimization"
        else:
            reasoning['efficiency'] = "Efficient implementation detected"
        
        # Readability reasoning
        if scores['readability'] < 0.7:
            reasoning['readability'] = "Code could benefit from better structure and naming"
        else:
            reasoning['readability'] = "Code is well-structured and readable"
        
        # Generality reasoning
        if scores['generality'] < 0.7:
            reasoning['generality'] = "Solution may not handle edge cases well"
        else:
            reasoning['generality'] = "Solution appears robust and general"
        
        return reasoning
    
    def batch_evaluate(self, code_samples: List[Tuple[str, str]]) -> List[EvaluationResult]:
        """Evaluate multiple code samples efficiently"""
        results = []
        for code, description in code_samples:
            result = self.evaluate(code, description)
            results.append(result)
        return results
    
    def update_from_feedback(self, code: str, actual_score: float, description: str = ""):
        """Update model based on LLM feedback (for continuous learning)"""
        # This would be implemented with actual training logic
        # For now, just log the feedback
        logger.info(f"Received feedback - Actual score: {actual_score}")
        
    def get_statistics(self) -> Dict:
        """Get evaluation statistics"""
        return {
            'cache_size': len(self.cache),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'config': self.config
        }