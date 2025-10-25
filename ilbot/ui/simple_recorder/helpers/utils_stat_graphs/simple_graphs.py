#!/usr/bin/env python3
"""
Simple statistical distribution graphs for utils.py methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def create_timing_distributions():
    """Create timing distribution curves for different parameters."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Timing Distribution Methods', fontsize=16, fontweight='bold')
    
    # Common parameters
    min_seconds = 0.5
    max_seconds = 3.0
    x = np.linspace(0, 4, 1000)
    
    # 1. Normal Distribution (sleep_normal)
    ax1 = axes[0, 0]
    ax1.set_title('Normal Distribution (sleep_normal)', fontweight='bold')
    
    center_bias_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for i, center_bias in enumerate(center_bias_values):
        center = (min_seconds + max_seconds) / 2
        std_dev = (max_seconds - min_seconds) / (4 + center_bias * 4)
        
        # Generate normal distribution
        y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - center) / std_dev) ** 2)
        y = y / np.max(y)  # Normalize for comparison
        
        ax1.plot(x, y, color=colors[i], linewidth=2, 
                label=f'center_bias={center_bias}')
    
    ax1.axvline(min_seconds, color='red', linestyle='--', alpha=0.7, label='Min/Max bounds')
    ax1.axvline(max_seconds, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Probability Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Exponential Distribution (sleep_exponential)
    ax2 = axes[0, 1]
    ax2.set_title('Exponential Distribution (sleep_exponential)', fontweight='bold')
    
    lambda_values = [0.5, 1.0, 1.5, 2.0, 3.0]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for i, lambda_param in enumerate(lambda_values):
        # Generate exponential distribution samples
        exp_samples = np.random.exponential(1/lambda_param, 10000)
        # Scale to our range
        scaled_samples = min_seconds + (exp_samples / (1 + exp_samples)) * (max_seconds - min_seconds)
        scaled_samples = np.clip(scaled_samples, min_seconds, max_seconds)
        
        ax2.hist(scaled_samples, bins=50, density=True, alpha=0.7, 
                color=colors[i], label=f'λ={lambda_param}')
    
    ax2.axvline(min_seconds, color='red', linestyle='--', alpha=0.7, label='Min/Max bounds')
    ax2.axvline(max_seconds, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Probability Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Beta Distribution (sleep_beta)
    ax3 = axes[1, 0]
    ax3.set_title('Beta Distribution (sleep_beta)', fontweight='bold')
    
    alpha_beta_pairs = [(1, 1), (2, 2), (3, 2), (2, 3), (4, 4)]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for i, (alpha, beta) in enumerate(alpha_beta_pairs):
        # Generate beta distribution samples
        beta_samples = np.random.beta(alpha, beta, 10000)
        # Scale to our range
        scaled_samples = min_seconds + beta_samples * (max_seconds - min_seconds)
        
        ax3.hist(scaled_samples, bins=50, density=True, alpha=0.7, 
                color=colors[i], label=f'α={alpha}, β={beta}')
    
    ax3.axvline(min_seconds, color='red', linestyle='--', alpha=0.7, label='Min/Max bounds')
    ax3.axvline(max_seconds, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Probability Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Comparison of all methods
    ax4 = axes[1, 1]
    ax4.set_title('Method Comparison', fontweight='bold')
    
    # Generate samples for each method
    methods = {
        'Uniform': np.random.uniform(min_seconds, max_seconds, 10000),
        'Normal (bias=0.5)': np.random.normal((min_seconds + max_seconds)/2, 
                                             (max_seconds - min_seconds)/6, 10000),
        'Exponential (λ=1.5)': np.clip(min_seconds + (np.random.exponential(1/1.5, 10000) / 
                                    (1 + np.random.exponential(1/1.5, 10000))) * (max_seconds - min_seconds), 
                                    min_seconds, max_seconds),
        'Beta (α=2, β=2)': min_seconds + np.random.beta(2, 2, 10000) * (max_seconds - min_seconds)
    }
    
    colors = ['blue', 'green', 'orange', 'purple']
    for i, (method, samples) in enumerate(methods.items()):
        ax4.hist(samples, bins=50, density=True, alpha=0.6, 
                color=colors[i], label=method)
    
    ax4.axvline(min_seconds, color='red', linestyle='--', alpha=0.7, label='Min/Max bounds')
    ax4.axvline(max_seconds, color='red', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Probability Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('timing_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_rectangle_heatmaps():
    """Create heatmaps for rectangle clicking methods."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Rectangle Clicking Distribution Methods', fontsize=16, fontweight='bold')
    
    # Define rectangle bounds
    min_x, max_x, min_y, max_y = 0, 100, 0, 100
    center_x, center_y = 50, 50
    
    # Generate sample points for each method
    n_samples = 10000
    
    # 1. Center (exact center)
    ax1 = axes[0, 0]
    ax1.set_title('Center (rect_center_xy)', fontweight='bold')
    
    # Show just the center point
    ax1.scatter([center_x], [center_y], c='red', s=100, marker='x', linewidth=3)
    ax1.set_xlim(min_x, max_x)
    ax1.set_ylim(min_y, max_y)
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.grid(True, alpha=0.3)
    
    # Add rectangle outline
    from matplotlib.patches import Rectangle
    rect = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, 
                    linewidth=2, edgecolor='black', facecolor='none')
    ax1.add_patch(rect)
    
    # 2. Random (uniform)
    ax2 = axes[0, 1]
    ax2.set_title('Random (rect_random_xy)', fontweight='bold')
    
    x_rand = np.random.uniform(min_x, max_x, n_samples)
    y_rand = np.random.uniform(min_y, max_y, n_samples)
    
    ax2.hexbin(x_rand, y_rand, gridsize=20, cmap='Blues')
    ax2.set_xlim(min_x, max_x)
    ax2.set_ylim(min_y, max_y)
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    ax2.grid(True, alpha=0.3)
    
    # Add rectangle outline
    rect = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, 
                    linewidth=2, edgecolor='black', facecolor='none')
    ax2.add_patch(rect)
    
    # 3. Normal distribution
    ax3 = axes[0, 2]
    ax3.set_title('Normal (rect_normal_xy)', fontweight='bold')
    
    center_bias = 0.5
    std_dev_x = (max_x - min_x) / (4 + center_bias * 4)
    std_dev_y = (max_y - min_y) / (4 + center_bias * 4)
    
    x_norm = np.random.normal(center_x, std_dev_x, n_samples)
    y_norm = np.random.normal(center_y, std_dev_y, n_samples)
    x_norm = np.clip(x_norm, min_x, max_x)
    y_norm = np.clip(y_norm, min_y, max_y)
    
    ax3.hexbin(x_norm, y_norm, gridsize=20, cmap='Greens')
    ax3.set_xlim(min_x, max_x)
    ax3.set_ylim(min_y, max_y)
    ax3.set_xlabel('X coordinate')
    ax3.set_ylabel('Y coordinate')
    ax3.grid(True, alpha=0.3)
    
    # Add rectangle outline
    rect = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, 
                    linewidth=2, edgecolor='black', facecolor='none')
    ax3.add_patch(rect)
    
    # 4. Beta distribution
    ax4 = axes[1, 0]
    ax4.set_title('Beta (rect_beta_xy)', fontweight='bold')
    
    alpha, beta = 2.0, 2.0
    beta_x = np.random.beta(alpha, beta, n_samples)
    beta_y = np.random.beta(alpha, beta, n_samples)
    x_beta = min_x + beta_x * (max_x - min_x)
    y_beta = min_y + beta_y * (max_y - min_y)
    
    ax4.hexbin(x_beta, y_beta, gridsize=20, cmap='Reds')
    ax4.set_xlim(min_x, max_x)
    ax4.set_ylim(min_y, max_y)
    ax4.set_xlabel('X coordinate')
    ax4.set_ylabel('Y coordinate')
    ax4.grid(True, alpha=0.3)
    
    # Add rectangle outline
    rect = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, 
                    linewidth=2, edgecolor='black', facecolor='none')
    ax4.add_patch(rect)
    
    # 5. Triangular distribution
    ax5 = axes[1, 1]
    ax5.set_title('Triangular (rect_triangular_xy)', fontweight='bold')
    
    mode_bias = 0.5
    mode_x = min_x + mode_bias * (max_x - min_x)
    mode_y = min_y + mode_bias * (max_y - min_y)
    
    x_tri = np.random.triangular(min_x, mode_x, max_x, n_samples)
    y_tri = np.random.triangular(min_y, mode_y, max_y, n_samples)
    
    ax5.hexbin(x_tri, y_tri, gridsize=20, cmap='Purples')
    ax5.set_xlim(min_x, max_x)
    ax5.set_ylim(min_y, max_y)
    ax5.set_xlabel('X coordinate')
    ax5.set_ylabel('Y coordinate')
    ax5.grid(True, alpha=0.3)
    
    # Add rectangle outline
    rect = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, 
                    linewidth=2, edgecolor='black', facecolor='none')
    ax5.add_patch(rect)
    
    # 6. Comparison heatmap
    ax6 = axes[1, 2]
    ax6.set_title('Method Comparison', fontweight='bold')
    
    # Create a combined heatmap showing all methods
    all_x = np.concatenate([x_rand, x_norm, x_beta, x_tri])
    all_y = np.concatenate([y_rand, y_norm, y_beta, y_tri])
    
    ax6.hexbin(all_x, all_y, gridsize=25, cmap='viridis')
    ax6.set_xlim(min_x, max_x)
    ax6.set_ylim(min_y, max_y)
    ax6.set_xlabel('X coordinate')
    ax6.set_ylabel('Y coordinate')
    ax6.grid(True, alpha=0.3)
    
    # Add rectangle outline
    rect = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, 
                    linewidth=2, edgecolor='black', facecolor='none')
    ax6.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig('rectangle_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all statistical distribution graphs."""
    print("Generating statistical distribution graphs...")
    
    print("Creating timing distributions...")
    create_timing_distributions()
    
    print("Creating rectangle heatmaps...")
    create_rectangle_heatmaps()
    
    print("All graphs generated successfully!")
    print("Files created:")
    print("- timing_distributions.png")
    print("- rectangle_heatmaps.png")

if __name__ == "__main__":
    main()
