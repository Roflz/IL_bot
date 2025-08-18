#!/usr/bin/env python3
"""
Evaluation Framework for Imitation Learning Model
Comprehensive Metrics and Performance Analysis
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import defaultdict

from .imitation_hybrid_model import ImitationHybridModel
from .imitation_loss import CombinedImitationLoss

class ImitationLearningEvaluator:
    """Comprehensive evaluator for imitation learning model"""
    
    def __init__(self, model: ImitationHybridModel, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.criterion = CombinedImitationLoss()
        
        # Evaluation metrics storage
        self.metrics_history = defaultdict(list)
        
    def evaluate_model(self, test_loader, verbose: bool = True) -> Dict[str, float]:
        """
        Comprehensive model evaluation
        
        Args:
            test_loader: DataLoader for test data
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        # Initialize metric collectors
        metrics = {
            'mouse_position_accuracy': [],
            'click_accuracy': [],
            'key_accuracy': [],
            'action_sequence_accuracy': [],
            'temporal_consistency': [],
            'game_context_accuracy': [],
            'overall_loss': [],
            'inference_time': []
        }
        
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Move to device
                current_gamestate = batch['current_gamestate'].to(self.device)
                temporal_sequence = batch['temporal_sequence'].to(self.device)
                screenshot = batch['screenshot'].to(self.device)
                action_target = batch['action_target'].to(self.device)
                
                # Measure inference time
                start_time = time.time()
                predictions = self.model(current_gamestate, screenshot, temporal_sequence)
                inference_time = time.time() - start_time
                
                # Calculate individual metrics
                mouse_acc = self._calculate_mouse_accuracy(predictions, action_target)
                click_acc = self._calculate_click_accuracy(predictions, action_target)
                key_acc = self._calculate_key_accuracy(predictions, action_target)
                sequence_acc = self._calculate_action_sequence_accuracy(predictions, action_target)
                temporal_cons = self._calculate_temporal_consistency(predictions, action_target)
                context_acc = self._calculate_game_context_accuracy(predictions, action_target, current_gamestate)
                
                # Calculate overall loss
                loss = self.criterion(predictions, {
                    'mouse_position': action_target[:, :2],
                    'mouse_click': action_target[:, 2:4],
                    'key_press': action_target[:, 4:54],
                    'scroll': action_target[:, 54:56],
                    'confidence': action_target[:, 56:57],
                    'action_count': action_target[:, 57:73]
                })[0]
                
                # Store metrics
                metrics['mouse_position_accuracy'].append(mouse_acc)
                metrics['click_accuracy'].append(click_acc)
                metrics['key_accuracy'].append(key_acc)
                metrics['action_sequence_accuracy'].append(sequence_acc)
                metrics['temporal_consistency'].append(temporal_cons)
                metrics['game_context_accuracy'].append(context_acc)
                metrics['overall_loss'].append(loss.item())
                metrics['inference_time'].append(inference_time)
                
                total_samples += len(current_gamestate)
                
                if verbose and (batch_idx + 1) % 10 == 0:
                    print(f"Evaluated {batch_idx + 1} batches...")
        
        # Calculate final metrics
        final_metrics = {}
        for metric_name, values in metrics.items():
            if metric_name == 'inference_time':
                final_metrics[metric_name] = np.mean(values) * 1000  # Convert to milliseconds
            else:
                final_metrics[metric_name] = np.mean(values)
        
        # Add additional metrics
        final_metrics['total_samples'] = total_samples
        final_metrics['throughput'] = total_samples / np.sum(metrics['inference_time'])
        
        if verbose:
            self._print_evaluation_results(final_metrics)
        
        # Store in history
        for metric_name, value in final_metrics.items():
            if metric_name not in ['total_samples']:
                self.metrics_history[metric_name].append(value)
        
        return final_metrics
    
    def _calculate_mouse_accuracy(self, predictions: Dict[str, torch.Tensor], 
                                 targets: torch.Tensor, 
                                 threshold: float = 5.0) -> float:
        """Calculate mouse position accuracy within threshold pixels"""
        pred_pos = predictions['mouse_position']
        target_pos = targets[:, :2]
        
        # Calculate Euclidean distance
        distances = torch.norm(pred_pos - target_pos, dim=1)
        
        # Count predictions within threshold
        within_threshold = (distances <= threshold).float()
        
        return within_threshold.mean().item()
    
    def _calculate_click_accuracy(self, predictions: Dict[str, torch.Tensor], 
                                 targets: torch.Tensor) -> float:
        """Calculate click type accuracy"""
        pred_clicks = torch.sigmoid(predictions['mouse_click'])
        target_clicks = targets[:, 2:4]
        
        # Convert to binary predictions
        pred_binary = (pred_clicks > 0.5).float()
        
        # Calculate accuracy for each click type
        accuracies = []
        for i in range(2):  # left and right click
            pred_i = pred_binary[:, i]
            target_i = target_clicks[:, i]
            acc = (pred_i == target_i).float().mean().item()
            accuracies.append(acc)
        
        return np.mean(accuracies)
    
    def _calculate_key_accuracy(self, predictions: Dict[str, torch.Tensor], 
                               targets: torch.Tensor) -> float:
        """Calculate key press accuracy"""
        pred_keys = predictions['key_press']
        target_keys = targets[:, 4:54]
        
        # Convert to class predictions
        pred_classes = torch.argmax(pred_keys, dim=1)
        target_classes = torch.argmax(target_keys, dim=1)
        
        # Calculate accuracy
        accuracy = (pred_classes == target_classes).float().mean().item()
        
        return accuracy
    
    def _calculate_action_sequence_accuracy(self, predictions: Dict[str, torch.Tensor], 
                                          targets: torch.Tensor) -> float:
        """Calculate action sequence accuracy"""
        pred_count = torch.argmax(predictions['action_count'], dim=1)
        target_count = targets[:, 57:73]  # Action count section
        
        # Convert target to class
        target_classes = torch.argmax(target_count, dim=1)
        
        # Calculate accuracy
        accuracy = (pred_count == target_classes).float().mean().item()
        
        return accuracy
    
    def _calculate_temporal_consistency(self, predictions: Dict[str, torch.Tensor], 
                                       targets: torch.Tensor) -> float:
        """Calculate temporal consistency score"""
        # This is a simplified version - in practice, you'd analyze the full sequence
        # For now, we'll use the confidence predictions as a proxy
        
        pred_confidence = torch.sigmoid(predictions['confidence'])
        target_confidence = targets[:, 56:57]
        
        # Calculate correlation between predicted and target confidence
        correlation = torch.corrcoef(torch.stack([pred_confidence.squeeze(), target_confidence.squeeze()]))[0, 1]
        
        # Convert to 0-1 scale
        consistency_score = (correlation + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        return consistency_score.item()
    
    def _calculate_game_context_accuracy(self, predictions: Dict[str, torch.Tensor], 
                                        targets: torch.Tensor, 
                                        gamestates: torch.Tensor) -> float:
        """Calculate if actions are appropriate for game context"""
        # This is a simplified version - in practice, you'd have more sophisticated
        # logic to determine what actions are valid in what contexts
        
        # For now, we'll use a simple heuristic: check if mouse positions are within
        # reasonable bounds for the game window
        
        pred_pos = predictions['mouse_position']
        
        # Check if positions are within reasonable bounds (0-800, 0-600)
        x_in_bounds = ((pred_pos[:, 0] >= 0) & (pred_pos[:, 0] <= 800)).float()
        y_in_bounds = ((pred_pos[:, 1] >= 0) & (pred_pos[:, 1] <= 600)).float()
        
        # Calculate percentage of valid positions
        valid_positions = (x_in_bounds & y_in_bounds).float().mean().item()
        
        return valid_positions
    
    def _print_evaluation_results(self, metrics: Dict[str, float):
        """Print formatted evaluation results"""
        print("\n" + "="*60)
        print("🎯 IMITATION LEARNING MODEL EVALUATION RESULTS")
        print("="*60)
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"  Mouse Position Accuracy: {metrics['mouse_position_accuracy']:.2%}")
        print(f"  Click Accuracy: {metrics['click_accuracy']:.2%}")
        print(f"  Key Press Accuracy: {metrics['key_accuracy']:.2%}")
        print(f"  Action Sequence Accuracy: {metrics['action_sequence_accuracy']:.2%}")
        print(f"  Temporal Consistency: {metrics['temporal_consistency']:.2%}")
        print(f"  Game Context Accuracy: {metrics['game_context_accuracy']:.2%}")
        
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"  Overall Loss: {metrics['overall_loss']:.4f}")
        print(f"  Inference Time: {metrics['inference_time']:.2f} ms")
        print(f"  Throughput: {metrics['throughput']:.1f} samples/sec")
        print(f"  Total Samples: {metrics['total_samples']:,}")
        
        print(f"\n📈 PERFORMANCE ASSESSMENT:")
        self._assess_performance(metrics)
    
    def _assess_performance(self, metrics: Dict[str, float]):
        """Assess overall model performance"""
        # Define performance thresholds
        thresholds = {
            'excellent': 0.90,
            'good': 0.80,
            'fair': 0.70,
            'poor': 0.60
        }
        
        # Calculate average accuracy across key metrics
        key_accuracies = [
            metrics['mouse_position_accuracy'],
            metrics['click_accuracy'],
            metrics['key_accuracy'],
            metrics['action_sequence_accuracy']
        ]
        
        avg_accuracy = np.mean(key_accuracies)
        
        # Determine performance level
        if avg_accuracy >= thresholds['excellent']:
            level = "EXCELLENT"
            emoji = "🌟"
        elif avg_accuracy >= thresholds['good']:
            level = "GOOD"
            emoji = "✅"
        elif avg_accuracy >= thresholds['fair']:
            level = "FAIR"
            emoji = "⚠️"
        elif avg_accuracy >= thresholds['poor']:
            level = "POOR"
            emoji = "❌"
        else:
            level = "UNACCEPTABLE"
            emoji = "🚫"
        
        print(f"  Overall Performance: {emoji} {level}")
        print(f"  Average Accuracy: {avg_accuracy:.2%}")
        
        # Performance recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if avg_accuracy < thresholds['good']:
            print(f"  - Consider additional training epochs")
            print(f"  - Review data quality and augmentation")
            print(f"  - Adjust model architecture if needed")
        elif avg_accuracy < thresholds['excellent']:
            print(f"  - Fine-tune hyperparameters")
            print(f"  - Increase training data diversity")
        else:
            print(f"  - Model is ready for deployment!")
            print(f"  - Consider real-world testing")
    
    def plot_metrics_history(self, save_path: Optional[str] = None):
        """Plot metrics history over time"""
        if not self.metrics_history:
            print("No metrics history available for plotting")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Imitation Learning Model Performance Over Time', fontsize=16)
        
        # Plot accuracy metrics
        ax1 = axes[0, 0]
        for metric_name, values in self.metrics_history.items():
            if 'accuracy' in metric_name or 'consistency' in metric_name:
                ax1.plot(values, label=metric_name.replace('_', ' ').title())
        ax1.set_title('Accuracy Metrics')
        ax1.set_xlabel('Evaluation Run')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2 = axes[0, 1]
        if 'overall_loss' in self.metrics_history:
            ax2.plot(self.metrics_history['overall_loss'], 'r-', label='Overall Loss')
        ax2.set_title('Loss Over Time')
        ax2.set_xlabel('Evaluation Run')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Plot performance metrics
        ax3 = axes[1, 0]
        if 'inference_time' in self.metrics_history:
            ax3.plot(self.metrics_history['inference_time'], 'g-', label='Inference Time (ms)')
        if 'throughput' in self.metrics_history:
            ax3_twin = ax3.twinx()
            ax3_twin.plot(self.metrics_history['throughput'], 'b-', label='Throughput (samples/sec)')
            ax3_twin.set_ylabel('Throughput')
        ax3.set_title('Performance Metrics')
        ax3.set_xlabel('Evaluation Run')
        ax3.set_ylabel('Inference Time (ms)')
        ax3.legend(loc='upper left')
        if 'throughput' in self.metrics_history:
            ax3_twin.legend(loc='upper right')
        ax3.grid(True)
        
        # Plot metric correlations
        ax4 = axes[1, 1]
        if len(self.metrics_history) > 1:
            # Create correlation matrix
            metric_names = list(self.metrics_history.keys())
            if len(metric_names) > 1:
                correlation_matrix = np.zeros((len(metric_names), len(metric_names)))
                for i, name1 in enumerate(metric_names):
                    for j, name2 in enumerate(metric_names):
                        if len(self.metrics_history[name1]) > 1 and len(self.metrics_history[name2]) > 1:
                            correlation_matrix[i, j] = np.corrcoef(
                                self.metrics_history[name1], 
                                self.metrics_history[name2]
                            )[0, 1]
                
                # Plot correlation heatmap
                im = ax4.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                ax4.set_xticks(range(len(metric_names)))
                ax4.set_yticks(range(len(metric_names)))
                ax4.set_xticklabels([name.replace('_', ' ').title() for name in metric_names], rotation=45)
                ax4.set_yticklabels([name.replace('_', ' ').title() for name in metric_names])
                ax4.set_title('Metric Correlations')
                
                # Add colorbar
                plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Metrics history plot saved to {save_path}")
        
        plt.show()
    
    def save_evaluation_results(self, filepath: str, metrics: Dict[str, float]):
        """Save evaluation results to file"""
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': metrics,
            'model_info': self.model.get_model_info(),
            'device': str(self.device)
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Evaluation results saved to {filepath}")
    
    def compare_models(self, model_paths: List[str], test_loader) -> Dict[str, Dict[str, float]]:
        """Compare multiple models on the same test set"""
        comparison_results = {}
        
        for model_path in model_paths:
            print(f"\n🔍 Evaluating model: {model_path}")
            
            # Load model
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Evaluate
            metrics = self.evaluate_model(test_loader, verbose=False)
            comparison_results[Path(model_path).stem] = metrics
        
        # Print comparison
        self._print_model_comparison(comparison_results)
        
        return comparison_results
    
    def _print_model_comparison(self, comparison_results: Dict[str, Dict[str, float]]):
        """Print comparison between models"""
        print("\n" + "="*80)
        print("📊 MODEL COMPARISON RESULTS")
        print("="*80)
        
        # Get all metric names
        all_metrics = set()
        for model_metrics in comparison_results.values():
            all_metrics.update(model_metrics.keys())
        
        # Print header
        metric_names = sorted(list(all_metrics))
        header = f"{'Model':<20}"
        for metric in metric_names:
            if 'accuracy' in metric or 'consistency' in metric:
                header += f"{metric.replace('_', ' ').title():<20}"
            else:
                header += f"{metric.replace('_', ' ').title():<15}"
        print(header)
        print("-" * len(header))
        
        # Print results for each model
        for model_name, metrics in comparison_results.items():
            row = f"{model_name:<20}"
            for metric in metric_names:
                if metric in metrics:
                    value = metrics[metric]
                    if 'accuracy' in metric or 'consistency' in metric:
                        row += f"{value:.3f}".ljust(20)
                    else:
                        row += f"{value:.3f}".ljust(15)
                else:
                    row += "N/A".ljust(20 if 'accuracy' in metric or 'consistency' in metric else 15)
            print(row)

if __name__ == "__main__":
    print("Testing Evaluation Framework...")
    
    # Create dummy model for testing
    model = ImitationHybridModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create evaluator
    evaluator = ImitationLearningEvaluator(model, device)
    
    print(f"✅ Evaluation framework created successfully!")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test evaluation methods
    print("\n🧪 Testing evaluation methods...")
    
    # Create dummy test data
    batch_size = 4
    gamestate_features = torch.randn(batch_size, 73)
    temporal_sequence = torch.randn(batch_size, 10, 73)
    screenshot = torch.randn(batch_size, 3, 224, 224)
    action_target = torch.randn(batch_size, 106)
    
    # Test individual metric calculations
    predictions = model(gamestate_features, screenshot, temporal_sequence)
    
    mouse_acc = evaluator._calculate_mouse_accuracy(predictions, action_target)
    click_acc = evaluator._calculate_click_accuracy(predictions, action_target)
    key_acc = evaluator._calculate_key_accuracy(predictions, action_target)
    
    print(f"✅ Individual metrics calculated:")
    print(f"  Mouse accuracy: {mouse_acc:.3f}")
    print(f"  Click accuracy: {click_acc:.3f}")
    print(f"  Key accuracy: {key_acc:.3f}")
    
    print(f"\n🎯 Evaluation framework ready for use!")














