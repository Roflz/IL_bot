#!/usr/bin/env python3
"""
OSRS Imitation Learning Model Evaluation Script
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
from collections import defaultdict

from model_architecture import OSRSImitationModel


class ModelEvaluator:
    """
    Evaluates the trained OSRS imitation learning model.
    """
    
    def __init__(
        self,
        model_path: str,
        data_dir: str = "data/training_data",
        device: str = None
    ):
        self.model_path = model_path
        self.data_dir = Path(data_dir)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model()
        
        # Load data
        self.input_sequences = np.load(self.data_dir / "input_sequences.npy")
        with open(self.data_dir / "raw_action_data.json", 'r') as f:
            self.action_data = json.load(f)
        
        print(f"Model loaded from: {model_path}")
        print(f"Device: {self.device}")
        print(f"Input sequences: {self.input_sequences.shape}")
        print(f"Action data: {len(self.action_data)} samples")
    
    def _load_model(self) -> OSRSImitationModel:
        """Load the trained model."""
        model = OSRSImitationModel(
            input_features=128,
            sequence_length=10,
            hidden_size=256,
            num_layers=2,
            dropout=0.2
        )
        
        # Load trained weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        return model
    
    def evaluate_model(self, num_samples: int = 50) -> Dict:
        """
        Evaluate the model on a subset of data.
        
        Args:
            num_samples: Number of samples to evaluate
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"\nEvaluating model on {num_samples} samples...")
        
        # Sample random sequences
        indices = np.random.choice(len(self.input_sequences), num_samples, replace=False)
        
        metrics = {
            'mouse_position_accuracy': [],
            'click_accuracy': [],
            'key_accuracy': [],
            'scroll_accuracy': [],
            'confidence_scores': [],
            'prediction_times': [],
            'sample_predictions': []
        }
        
        for i, idx in enumerate(indices):
            # Get input sequence
            input_sequence = self.input_sequences[idx]
            
            # Get ground truth actions (simplified for now)
            ground_truth = self._get_ground_truth(idx)
            
            # Time the prediction
            start_time = time.time()
            
            # Generate prediction
            with torch.no_grad():
                input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(self.device)
                prediction = self.model(input_tensor)
                probabilities = self.model.get_action_probabilities(prediction)
            
            prediction_time = time.time() - start_time
            
            # Calculate metrics
            sample_metrics = self._calculate_sample_metrics(probabilities, ground_truth)
            
            # Store metrics
            for key, value in sample_metrics.items():
                if key in metrics:
                    metrics[key].append(value)
            
            metrics['prediction_times'].append(prediction_time)
            
            # Store sample prediction for analysis
            if i < 5:  # Store first 5 samples
                sample_data = {
                    'sequence_idx': idx,
                    'ground_truth': ground_truth,
                    'prediction': {k: v.cpu().numpy() for k, v in prediction.items()},
                    'probabilities': {k: v.cpu().numpy() for k, v in probabilities.items()},
                    'metrics': sample_metrics
                }
                metrics['sample_predictions'].append(sample_data)
            
            if (i + 1) % 10 == 0:
                print(f"Evaluated {i + 1}/{num_samples} samples")
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(metrics)
        
        return metrics, aggregate_metrics
    
    def _get_ground_truth(self, sequence_idx: int) -> Dict[str, np.ndarray]:
        """
        Get ground truth actions for a sequence (simplified).
        
        Args:
            sequence_idx: Index of the sequence
            
        Returns:
            Dictionary of ground truth actions
        """
        # TODO: Implement proper ground truth extraction from action_data
        # For now, return dummy ground truth
        
        return {
            'mouse_position': np.array([400.0, 300.0]),
            'mouse_click': np.array([0, 1, 0]),  # One-hot: left click
            'key_press': np.array([0] * 50),     # No key
            'scroll': np.array([0.0, 0.0]),      # No scroll
            'confidence': np.array([1.0])         # High confidence
        }
    
    def _calculate_sample_metrics(
        self, 
        prediction: Dict[str, torch.Tensor], 
        ground_truth: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Calculate metrics for a single sample."""
        metrics = {}
        
        # Mouse position accuracy (Euclidean distance)
        if 'mouse_position' in prediction and 'mouse_position' in ground_truth:
            pred_pos = prediction['mouse_position'].cpu().numpy().flatten()
            true_pos = ground_truth['mouse_position']
            distance = np.linalg.norm(pred_pos - true_pos)
            metrics['mouse_position_distance'] = distance
            metrics['mouse_position_accuracy'] = 1.0 / (1.0 + distance)  # Higher is better
        
        # Click accuracy (classification accuracy)
        if 'mouse_click' in prediction and 'mouse_click' in ground_truth:
            pred_click = prediction['mouse_click'].cpu().numpy().flatten()
            true_click = ground_truth['mouse_click']
            pred_class = np.argmax(pred_click)
            true_class = np.argmax(true_click)
            metrics['click_accuracy'] = float(pred_class == true_class)
        
        # Key press accuracy (classification accuracy)
        if 'key_press' in prediction and 'key_press' in ground_truth:
            pred_key = prediction['key_press'].cpu().numpy().flatten()
            true_key = ground_truth['key_press']
            pred_class = np.argmax(pred_key)
            true_class = np.argmax(true_key)
            metrics['key_accuracy'] = float(pred_class == true_class)
        
        # Scroll accuracy (Euclidean distance)
        if 'scroll' in prediction and 'scroll' in ground_truth:
            pred_scroll = prediction['scroll'].cpu().numpy().flatten()
            true_scroll = ground_truth['scroll']
            distance = np.linalg.norm(pred_scroll - true_scroll)
            metrics['scroll_distance'] = distance
            metrics['scroll_accuracy'] = 1.0 / (1.0 + distance)
        
        # Confidence score
        if 'confidence' in prediction:
            metrics['confidence_score'] = prediction['confidence'].cpu().numpy().flatten()[0]
        
        return metrics
    
    def _calculate_aggregate_metrics(self, metrics: Dict) -> Dict[str, float]:
        """Calculate aggregate metrics across all samples."""
        aggregate = {}
        
        for key, values in metrics.items():
            if key == 'sample_predictions':
                continue
            
            if values:
                aggregate[f'{key}_mean'] = np.mean(values)
                aggregate[f'{key}_std'] = np.std(values)
                aggregate[f'{key}_min'] = np.min(values)
                aggregate[f'{key}_max'] = np.max(values)
        
        return aggregate
    
    def print_evaluation_results(self, metrics: Dict, aggregate_metrics: Dict):
        """Print comprehensive evaluation results."""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        print(f"\n📊 AGGREGATE METRICS:")
        print(f"{'Metric':<30} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 70)
        
        for key, value in aggregate_metrics.items():
            if 'mean' in key:
                base_key = key.replace('_mean', '')
                mean_val = aggregate_metrics[f'{base_key}_mean']
                std_val = aggregate_metrics[f'{base_key}_std']
                min_val = aggregate_metrics[f'{base_key}_min']
                max_val = aggregate_metrics[f'{base_key}_max']
                
                print(f"{base_key:<30} {mean_val:<10.4f} {std_val:<10.4f} {min_val:<10.4f} {max_val:<10.4f}")
        
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"Average prediction time: {np.mean(metrics['prediction_times']):.4f} seconds")
        print(f"Predictions per second: {1.0 / np.mean(metrics['prediction_times']):.1f}")
        
        print(f"\n🎯 ACCURACY BREAKDOWN:")
        if 'mouse_position_accuracy' in aggregate_metrics:
            pos_acc = aggregate_metrics['mouse_position_accuracy_mean']
            print(f"Mouse position accuracy: {pos_acc:.4f} (higher is better)")
        
        if 'click_accuracy' in aggregate_metrics:
            click_acc = aggregate_metrics['click_accuracy_mean']
            print(f"Click accuracy: {click_acc:.4f} ({click_acc*100:.1f}%)")
        
        if 'key_accuracy' in aggregate_metrics:
            key_acc = aggregate_metrics['key_accuracy_mean']
            print(f"Key press accuracy: {key_acc:.4f} ({key_acc*100:.1f}%)")
    
    def visualize_predictions(self, metrics: Dict, save_path: str = "evaluation_results.png"):
        """Create visualizations of the evaluation results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('OSRS Imitation Learning Model Evaluation Results', fontsize=16)
        
        # 1. Mouse Position Accuracy Distribution
        if 'mouse_position_accuracy' in metrics and metrics['mouse_position_accuracy']:
            axes[0, 0].hist(metrics['mouse_position_accuracy'], bins=20, alpha=0.7, color='blue')
            axes[0, 0].set_title('Mouse Position Accuracy Distribution')
            axes[0, 0].set_xlabel('Accuracy Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Click Accuracy Distribution
        if 'click_accuracy' in metrics and metrics['click_accuracy']:
            axes[0, 1].hist(metrics['click_accuracy'], bins=[0, 0.5, 1], alpha=0.7, color='green')
            axes[0, 1].set_title('Click Accuracy Distribution')
            axes[0, 1].set_xlabel('Correct (1) / Incorrect (0)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Key Press Accuracy Distribution
        if 'key_accuracy' in metrics and metrics['key_accuracy']:
            axes[0, 2].hist(metrics['key_accuracy'], bins=[0, 0.5, 1], alpha=0.7, color='red')
            axes[0, 2].set_title('Key Press Accuracy Distribution')
            axes[0, 2].set_xlabel('Correct (1) / Incorrect (0)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Prediction Time Distribution
        if 'prediction_times' in metrics and metrics['prediction_times']:
            axes[1, 0].hist(metrics['prediction_times'], bins=20, alpha=0.7, color='orange')
            axes[1, 0].set_title('Prediction Time Distribution')
            axes[1, 0].set_xlabel('Time (seconds)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Confidence Score Distribution
        if 'confidence_scores' in metrics and metrics['confidence_scores']:
            axes[1, 1].hist(metrics['confidence_scores'], bins=20, alpha=0.7, color='purple')
            axes[1, 1].set_title('Confidence Score Distribution')
            axes[1, 1].set_xlabel('Confidence Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Sample Predictions vs Ground Truth
        if 'sample_predictions' in metrics and metrics['sample_predictions']:
            sample = metrics['sample_predictions'][0]
            if 'mouse_position' in sample['prediction']:
                pred_pos = sample['prediction']['mouse_position'][0]
                true_pos = sample['ground_truth']['mouse_position']
                
                axes[1, 2].scatter([true_pos[0]], [true_pos[1]], c='green', s=100, label='Ground Truth', marker='o')
                axes[1, 2].scatter([pred_pos[0]], [pred_pos[1]], c='red', s=100, label='Prediction', marker='x')
                axes[1, 2].set_title('Sample: Mouse Position Prediction')
                axes[1, 2].set_xlabel('X Coordinate')
                axes[1, 2].set_ylabel('Y Coordinate')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📈 Visualization saved to: {save_path}")
        plt.show()
    
    def test_single_prediction(self, sequence_idx: int = None) -> Dict:
        """
        Test the model on a single sequence and show detailed results.
        
        Args:
            sequence_idx: Index of sequence to test (random if None)
            
        Returns:
            Dictionary with prediction details
        """
        if sequence_idx is None:
            sequence_idx = np.random.randint(0, len(self.input_sequences))
        
        print(f"\n🧪 TESTING SINGLE PREDICTION (Sequence {sequence_idx})")
        print("=" * 50)
        
        # Get input sequence
        input_sequence = self.input_sequences[sequence_idx]
        
        # Generate prediction
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(self.device)
            prediction = self.model(input_tensor)
            probabilities = self.model.get_action_probabilities(prediction)
        
        # Get ground truth
        ground_truth = self._get_ground_truth(sequence_idx)
        
        # Display results
        print(f"Input sequence shape: {input_sequence.shape}")
        print(f"Sequence features range: [{input_sequence.min():.2f}, {input_sequence.max():.2f}]")
        
        print(f"\n🎯 PREDICTIONS:")
        for key, value in prediction.items():
            if key == 'mouse_position':
                pred_pos = value.cpu().numpy()[0]
                true_pos = ground_truth[key]
                distance = np.linalg.norm(pred_pos - true_pos)
                print(f"  {key}: Predicted ({pred_pos[0]:.1f}, {pred_pos[1]:.1f}) | "
                      f"True ({true_pos[0]:.1f}, {true_pos[1]:.1f}) | Distance: {distance:.1f}")
            
            elif key == 'mouse_click':
                pred_probs = probabilities[key].cpu().numpy()[0]
                pred_class = np.argmax(pred_probs)
                true_class = np.argmax(ground_truth[key])
                click_types = ['None', 'Left', 'Right']
                print(f"  {key}: Predicted {click_types[pred_class]} ({pred_probs[pred_class]:.3f}) | "
                      f"True: {click_types[true_class]}")
            
            elif key == 'key_press':
                pred_probs = probabilities[key].cpu().numpy()[0]
                pred_class = np.argmax(pred_probs)
                true_class = np.argmax(ground_truth[key])
                print(f"  {key}: Predicted key {pred_class} ({pred_probs[pred_class]:.3f}) | "
                      f"True: key {true_class}")
            
            elif key == 'scroll':
                pred_scroll = value.cpu().numpy()[0]
                true_scroll = ground_truth[key]
                print(f"  {key}: Predicted ({pred_scroll[0]:.1f}, {pred_scroll[1]:.1f}) | "
                      f"True ({true_scroll[0]:.1f}, {true_scroll[1]:.1f})")
            
            elif key == 'confidence':
                conf = float(value.cpu().numpy()[0])
                print(f"  {key}: {conf:.3f}")
        
        return {
            'sequence_idx': sequence_idx,
            'prediction': prediction,
            'probabilities': probabilities,
            'ground_truth': ground_truth
        }


def main():
    """Main evaluation function."""
    print("OSRS Imitation Learning Model Evaluation")
    print("=" * 50)
    
    # Check if model exists
    model_path = "best_model.pth"
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train_model.py")
        return
    
    # Create evaluator
    evaluator = ModelEvaluator(model_path)
    
    # Evaluate model
    metrics, aggregate_metrics = evaluator.evaluate_model(num_samples=100)
    
    # Print results
    evaluator.print_evaluation_results(metrics, aggregate_metrics)
    
    # Create visualizations
    evaluator.visualize_predictions(metrics)
    
    # Test single prediction
    evaluator.test_single_prediction()
    
    print(f"\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
