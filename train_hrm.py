"""
Training module for HRM evaluator
Trains on historical solutions from knowledge base
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import argparse
import os
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
import random

from hrm_evaluator import HierarchicalReasoningModel, CodeEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeEvaluationDataset(Dataset):
    """Dataset for training HRM on code evaluation"""
    
    def __init__(self, data_path: str = "knowledge_base.json", augment: bool = False):
        self.samples = []
        self.augment = augment
        
        # Load from knowledge base
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                kb = json.load(f)
                
            for problem, data in kb.items():
                if 'solution' in data:
                    # Use fitness score if available, otherwise estimate
                    score = data.get('fitness_score', 0.75)
                    
                    # Create training sample
                    sample = {
                        'code': data['solution'],
                        'problem': problem,
                        'scores': self._score_to_components(score),
                        'overall': score
                    }
                    self.samples.append(sample)
                    
                    # Add augmented samples
                    if augment:
                        self.samples.extend(self._augment_sample(sample))
        
        # Add synthetic training data if knowledge base is small
        if len(self.samples) < 100:
            logger.info(f"Adding synthetic samples (current: {len(self.samples)})")
            self.samples.extend(self._generate_synthetic_samples(100 - len(self.samples)))
        
        logger.info(f"Loaded {len(self.samples)} training samples")
    
    def _score_to_components(self, overall_score: float) -> Dict[str, float]:
        """Decompose overall score into component scores"""
        # Simple heuristic decomposition
        base = overall_score
        variation = 0.1
        
        return {
            'correctness': np.clip(base + random.uniform(-variation, variation), 0, 1),
            'efficiency': np.clip(base + random.uniform(-variation, variation), 0, 1),
            'readability': np.clip(base + random.uniform(-variation, variation), 0, 1),
            'generality': np.clip(base + random.uniform(-variation, variation), 0, 1),
        }
    
    def _augment_sample(self, sample: Dict) -> List[Dict]:
        """Create augmented versions of a sample"""
        augmented = []
        
        # Slightly modify scores to create variations
        for _ in range(2):
            new_sample = sample.copy()
            new_scores = {}
            for key, value in sample['scores'].items():
                # Add small noise
                new_scores[key] = np.clip(value + random.uniform(-0.05, 0.05), 0, 1)
            new_sample['scores'] = new_scores
            new_sample['overall'] = np.mean(list(new_scores.values()))
            augmented.append(new_sample)
        
        return augmented
    
    def _generate_synthetic_samples(self, n: int) -> List[Dict]:
        """Generate synthetic training samples"""
        samples = []
        
        # Template code patterns
        templates = [
            # Good code pattern
            """def solve_{name}(input_data):
    \"\"\"Well-documented function\"\"\"
    result = []
    for item in input_data:
        if validate(item):
            result.append(process(item))
    return result""",
            
            # Medium code pattern
            """def f(x):
    r = []
    for i in x:
        r.append(i * 2)
    return r""",
            
            # Poor code pattern
            """def x(a,b,c,d,e,f):
    return a+b+c+d+e+f"""
        ]
        
        scores = [0.85, 0.60, 0.35]  # Corresponding quality scores
        
        for i in range(n):
            template_idx = i % len(templates)
            code = templates[template_idx].replace('{name}', f'problem_{i}')
            score = scores[template_idx] + random.uniform(-0.1, 0.1)
            
            samples.append({
                'code': code,
                'problem': f'synthetic_problem_{i}',
                'scores': self._score_to_components(score),
                'overall': score
            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class HRMTrainer:
    """Trainer for HRM evaluator model"""
    
    def __init__(self, model_path: str = "models/hrm_evaluator.pth"):
        self.device = torch.device("cpu")
        self.model_path = model_path
        
        # Ensure model directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Initialize model
        self.model = HierarchicalReasoningModel()
        self.model.to(self.device)
        
        # Initialize encoder separately for processing
        self.encoder = CodeEncoder(512, 256)
        
        # Load existing model if available
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded existing model from {model_path}")
            except:
                logger.info("Starting with fresh model")
        
        # Loss functions
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
    
    def train(self, dataset: CodeEvaluationDataset, epochs: int = 10, 
              batch_size: int = 8, val_split: float = 0.2):
        """Train the HRM model"""
        # Split dataset
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        # Create data loaders
        train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in progress_bar:
                # Encode code samples
                batch_features = []
                batch_targets = []
                
                for sample in batch:
                    # Encode code to features
                    code_features = self.encoder(sample['code'])
                    batch_features.append(code_features)
                    
                    # Prepare target scores
                    targets = torch.tensor([
                        sample['scores']['correctness'],
                        sample['scores']['efficiency'],
                        sample['scores']['readability'],
                        sample['scores']['generality']
                    ], dtype=torch.float32)
                    batch_targets.append(targets)
                
                # Stack batch
                if batch_features:
                    batch_features = torch.cat(batch_features, dim=0)
                    batch_targets = torch.stack(batch_targets)
                    
                    # Forward pass
                    scores, _ = self.model(batch_features)
                    
                    # Compute losses for each component
                    pred_scores = torch.stack([
                        torch.sigmoid(self.model.correctness_head(
                            self.model.H_init.expand(batch_features.shape[0], -1))),
                        torch.sigmoid(self.model.efficiency_head(
                            self.model.H_init.expand(batch_features.shape[0], -1))),
                        torch.sigmoid(self.model.readability_head(
                            self.model.H_init.expand(batch_features.shape[0], -1))),
                        torch.sigmoid(self.model.generality_head(
                            self.model.H_init.expand(batch_features.shape[0], -1)))
                    ], dim=1).squeeze(-1)
                    
                    loss = self.criterion(pred_scores, batch_targets)
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    
                    train_losses.append(loss.item())
                    progress_bar.set_postfix({'loss': loss.item()})
            
            # Validation phase
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch_features = []
                    batch_targets = []
                    
                    for sample in batch:
                        code_features = self.encoder(sample['code'])
                        batch_features.append(code_features)
                        
                        targets = torch.tensor([
                            sample['scores']['correctness'],
                            sample['scores']['efficiency'],
                            sample['scores']['readability'],
                            sample['scores']['generality']
                        ], dtype=torch.float32)
                        batch_targets.append(targets)
                    
                    if batch_features:
                        batch_features = torch.cat(batch_features, dim=0)
                        batch_targets = torch.stack(batch_targets)
                        
                        scores, _ = self.model(batch_features)
                        
                        pred_scores = torch.stack([
                            torch.sigmoid(self.model.correctness_head(
                                self.model.H_init.expand(batch_features.shape[0], -1))),
                            torch.sigmoid(self.model.efficiency_head(
                                self.model.H_init.expand(batch_features.shape[0], -1))),
                            torch.sigmoid(self.model.readability_head(
                                self.model.H_init.expand(batch_features.shape[0], -1))),
                            torch.sigmoid(self.model.generality_head(
                                self.model.H_init.expand(batch_features.shape[0], -1)))
                        ], dim=1).squeeze(-1)
                        
                        loss = self.criterion(pred_scores, batch_targets)
                        val_losses.append(loss.item())
            
            # Calculate epoch metrics
            avg_train_loss = np.mean(train_losses) if train_losses else 0
            avg_val_loss = np.mean(val_losses) if val_losses else 0
            
            # Update learning rate
            self.scheduler.step(avg_val_loss)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"├── Train Loss: {avg_train_loss:.4f}")
            print(f"├── Val Loss: {avg_val_loss:.4f}")
            print(f"└── LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), self.model_path)
                print(f"✓ Saved best model (val_loss: {avg_val_loss:.4f})")
        
        print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
        return self.model
    
    def evaluate_model(self, dataset: CodeEvaluationDataset) -> Dict[str, float]:
        """Evaluate model performance"""
        self.model.eval()
        loader = DataLoader(dataset, batch_size=1)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for sample in loader:
                code_features = self.encoder(sample['code'][0])
                scores, _ = self.model(code_features)
                
                pred_overall = np.mean([
                    scores['correctness'],
                    scores['efficiency'],
                    scores['readability'],
                    scores['generality']
                ])
                
                all_predictions.append(pred_overall)
                all_targets.append(sample['overall'].item())
        
        # Calculate metrics
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        
        # Accuracy within threshold
        threshold = 0.15
        accurate = np.sum(np.abs(predictions - targets) < threshold) / len(targets)
        
        return {
            'mse': mse,
            'mae': mae,
            'accuracy': accurate,
            'samples': len(targets)
        }


def main():
    parser = argparse.ArgumentParser(description='Train HRM evaluator')
    parser.add_argument('--data', type=str, default='knowledge_base.json',
                      help='Path to training data')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                      help='Batch size for training')
    parser.add_argument('--augment', action='store_true',
                      help='Use data augmentation')
    parser.add_argument('--recent', type=int, default=0,
                      help='Train on N most recent solutions')
    parser.add_argument('--verbose', action='store_true',
                      help='Verbose output')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("=" * 50)
    print("HRM Evaluator Training")
    print("=" * 50)
    
    # Create dataset
    print("\n📚 Loading training data...")
    dataset = CodeEvaluationDataset(args.data, augment=args.augment)
    
    # Initialize trainer
    trainer = HRMTrainer()
    
    # Train model
    print("\n🎯 Starting training...")
    model = trainer.train(dataset, epochs=args.epochs, batch_size=args.batch_size)
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    metrics = trainer.evaluate_model(dataset)
    
    print("\nFinal Metrics:")
    print(f"├── MSE: {metrics['mse']:.4f}")
    print(f"├── MAE: {metrics['mae']:.4f}")
    print(f"├── Accuracy: {metrics['accuracy']:.2%}")
    print(f"└── Samples: {metrics['samples']}")
    
    print("\n✅ Training complete! Model saved to models/hrm_evaluator.pth")


if __name__ == "__main__":
    main()