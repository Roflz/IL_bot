#!/usr/bin/env python3
"""
Pretty output utilities for training with color coding and clean formatting
"""

import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime

class Colors:
    """ANSI color codes for terminal output"""
    # Basic colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    
    # Bright colors
    BRIGHT_RED = '\033[1;91m'
    BRIGHT_GREEN = '\033[1;92m'
    BRIGHT_YELLOW = '\033[1;93m'
    BRIGHT_BLUE = '\033[1;94m'
    BRIGHT_MAGENTA = '\033[1;95m'
    BRIGHT_CYAN = '\033[1;96m'
    BRIGHT_WHITE = '\033[1;97m'
    
    # Background colors
    BG_RED = '\033[101m'
    BG_GREEN = '\033[102m'
    BG_YELLOW = '\033[103m'
    BG_BLUE = '\033[104m'
    
    # Styles
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    
    # Dim colors for less important info
    DIM = '\033[2m'
    DIM_WHITE = '\033[2;97m'
    DIM_CYAN = '\033[2;96m'

class PrettyPrinter:
    """Pretty printer with color coding and clean formatting"""
    
    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors
        self.start_time = datetime.now()
        
    def _colorize(self, text: str, color: str) -> str:
        """Apply color if colors are enabled"""
        if self.use_colors and os.getenv('TERM') != 'dumb':
            return f"{color}{text}{Colors.RESET}"
        return text
    
    def _format_number(self, value: float, decimals: int = 2, width: int = 8) -> str:
        """Format numbers with consistent width"""
        if abs(value) < 0.001:
            return f"{0:>{width}.{decimals}f}"
        elif abs(value) < 1:
            return f"{value:>{width}.{decimals}f}"
        elif abs(value) < 100:
            return f"{value:>{width}.{decimals}f}"
        else:
            return f"{value:>{width}.1f}"
    
    def _get_status_color(self, value: float, target: float, tolerance: float = 0.1) -> str:
        """Get color based on how close value is to target"""
        diff = abs(value - target) / max(target, 0.001)
        if diff < tolerance:
            return Colors.BRIGHT_GREEN
        elif diff < tolerance * 2:
            return Colors.YELLOW
        else:
            return Colors.RED
    
    def print_header(self, title: str, subtitle: str = ""):
        """Print a beautiful header"""
        print()
        print(self._colorize("=" * 80, Colors.BRIGHT_CYAN))
        print(self._colorize(f"🚀 {title}", Colors.BRIGHT_WHITE + Colors.BOLD))
        if subtitle:
            print(self._colorize(f"   {subtitle}", Colors.CYAN))
        print(self._colorize("=" * 80, Colors.BRIGHT_CYAN))
        print()
    
    def print_epoch_start(self, epoch: int, total_epochs: int):
        """Print epoch start with progress"""
        progress = epoch / total_epochs
        bar_length = 30
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print()
        print(self._colorize(f"📊 Epoch {epoch}/{total_epochs}", Colors.BRIGHT_BLUE + Colors.BOLD))
        print(self._colorize(f"   Progress: [{bar}] {progress:.0%}", Colors.BLUE))
        print()
    
    def print_training_progress(self, batch_idx: int, total_batches: int, loss: float):
        """Print training progress with color coding"""
        if (batch_idx + 1) % 5 == 0:  # Every 5 batches
            progress = (batch_idx + 1) / total_batches
            bar_length = 20
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            # Color code loss based on typical ranges
            if loss < 1.0:
                loss_color = Colors.BRIGHT_GREEN
            elif loss < 5.0:
                loss_color = Colors.YELLOW
            else:
                loss_color = Colors.RED
            
            print(f"  {self._colorize('🎯', Colors.BLUE)} Batch {batch_idx + 1:>3}/{total_batches} "
                  f"[{bar}] {progress:.0%} "
                  f"Loss: {self._colorize(f'{loss:.2f}', loss_color)}")
    
    def print_epoch_summary(self, epoch: int, train_loss: float, val_loss: float, 
                          best_val: float, patience_left: int, is_best: bool = False):
        """Print clean epoch summary"""
        print()
        print(self._colorize("📈 Epoch Summary", Colors.BRIGHT_WHITE + Colors.BOLD))
        print(self._colorize("─" * 50, Colors.CYAN))
        
        # Training loss with color coding
        train_color = Colors.BRIGHT_GREEN if train_loss < 2.0 else Colors.YELLOW if train_loss < 5.0 else Colors.RED
        print(f"  {self._colorize('🎯', Colors.BLUE)} Training Loss:  {self._colorize(f'{train_loss:.3f}', train_color)}")
        
        # Validation loss with color coding
        val_color = Colors.BRIGHT_GREEN if val_loss < 2.0 else Colors.YELLOW if val_loss < 5.0 else Colors.RED
        print(f"  {self._colorize('🔍', Colors.MAGENTA)} Validation Loss: {self._colorize(f'{val_loss:.3f}', val_color)}")
        
        # Best validation loss
        print(f"  {self._colorize('🏆', Colors.YELLOW)} Best Val Loss:   {self._colorize(f'{best_val:.3f}', Colors.BRIGHT_YELLOW)}")
        
        # Improvement status
        if is_best:
            print(f"  {self._colorize('✨', Colors.BRIGHT_GREEN)} New Best! Model saved")
        else:
            patience_color = Colors.RED if patience_left <= 2 else Colors.YELLOW if patience_left <= 5 else Colors.GREEN
            print(f"  {self._colorize('⏳', Colors.DIM_WHITE)} Patience left: {self._colorize(f'{patience_left}', patience_color)}")
        
        print()
    
    def print_behavioral_analysis(self, analysis: Dict[str, Any], epoch: int):
        """Print clean behavioral analysis"""
        print(self._colorize("🧠 Behavioral Analysis", Colors.BRIGHT_WHITE + Colors.BOLD))
        print(self._colorize("─" * 50, Colors.CYAN))
        
        # Timing analysis - simplified
        if 'timing' in analysis:
            timing = analysis['timing']
            pred_timing = timing.get('mean_timing', 0.0)
            target_timing = timing.get('target_mean_timing', 0.0)
            
            # Color code based on how close prediction is to target
            timing_color = self._get_status_color(pred_timing, target_timing, 0.2)
            
            print(f"  {self._colorize('⏱️', Colors.BLUE)} Timing Prediction:")
            print(f"    Predicted: {self._colorize(f'{pred_timing:.3f}s', timing_color)} "
                  f"Target: {self._colorize(f'{target_timing:.3f}s', Colors.DIM_WHITE)}")
            
            # Show improvement direction
            if pred_timing > target_timing * 1.5:
                print(f"    {self._colorize('⚠️  Model predicting too slow', Colors.YELLOW)}")
            elif pred_timing < target_timing * 0.5:
                print(f"    {self._colorize('⚠️  Model predicting too fast', Colors.YELLOW)}")
            else:
                print(f"    {self._colorize('✅ Good timing prediction', Colors.BRIGHT_GREEN)}")
        
        # Actions per gamestate - simplified
        if 'actions_per_gamestate' in analysis:
            actions = analysis['actions_per_gamestate']
            pred_actions = actions.get('mean_actions_pred', 0.0)
            target_actions = actions.get('mean_actions_target', 0.0)
            
            actions_color = self._get_status_color(pred_actions, target_actions, 0.3)
            
            print(f"  {self._colorize('🎯', Colors.MAGENTA)} Actions per Gamestate:")
            print(f"    Predicted: {self._colorize(f'{pred_actions:.1f}', actions_color)} "
                  f"Target: {self._colorize(f'{target_actions:.1f}', Colors.DIM_WHITE)}")
            
            # Show improvement direction
            if pred_actions > target_actions * 2:
                print(f"    {self._colorize('⚠️  Model predicting too many actions', Colors.YELLOW)}")
            elif pred_actions < target_actions * 0.5:
                print(f"    {self._colorize('⚠️  Model predicting too few actions', Colors.YELLOW)}")
            else:
                print(f"    {self._colorize('✅ Good action count prediction', Colors.BRIGHT_GREEN)}")
        
        # Event distribution - simplified
        if 'event_distribution' in analysis:
            events = analysis['event_distribution']
            print(f"  {self._colorize('📊', Colors.CYAN)} Event Distribution:")
            
            event_names = ['CLICK', 'KEY', 'SCROLL', 'MOVE']
            for i, name in enumerate(event_names):
                if f'event_{i}_confidence' in events:
                    conf = events[f'event_{i}_confidence']
                    conf_color = Colors.BRIGHT_GREEN if conf > 0.7 else Colors.YELLOW if conf > 0.4 else Colors.RED
                    print(f"    {name}: {self._colorize(f'{conf:.2f}', conf_color)}")
        
        print()
    
    def print_memory_usage(self):
        """Print CUDA memory usage if available"""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**2
                reserved = torch.cuda.memory_reserved(0) / 1024**2
                
                # Color code based on memory usage
                if allocated > 4000:  # > 4GB
                    mem_color = Colors.RED
                elif allocated > 2000:  # > 2GB
                    mem_color = Colors.YELLOW
                else:
                    mem_color = Colors.GREEN
                
                print(f"  {self._colorize('💾', Colors.DIM_CYAN)} Memory: "
                      f"{self._colorize(f'{allocated:.0f}MB', mem_color)} allocated, "
                      f"{self._colorize(f'{reserved:.0f}MB', Colors.DIM_WHITE)} reserved")
        except ImportError:
            pass
    
    def print_final_results(self, train_losses: list, val_losses: list, best_val: float):
        """Print final training results"""
        print()
        self.print_header("Training Complete!", "Final Results Summary")
        
        # Final losses
        final_train = train_losses[-1] if train_losses else 0.0
        final_val = val_losses[-1] if val_losses else 0.0
        
        print(f"  {self._colorize('🎯', Colors.BLUE)} Final Training Loss:   {self._colorize(f'{final_train:.3f}', Colors.BRIGHT_BLUE)}")
        print(f"  {self._colorize('🔍', Colors.MAGENTA)} Final Validation Loss: {self._colorize(f'{final_val:.3f}', Colors.BRIGHT_MAGENTA)}")
        print(f"  {self._colorize('🏆', Colors.YELLOW)} Best Validation Loss:  {self._colorize(f'{best_val:.3f}', Colors.BRIGHT_YELLOW)}")
        
        # Training time
        elapsed = datetime.now() - self.start_time
        print(f"  {self._colorize('⏱️', Colors.CYAN)} Training Time:         {self._colorize(f'{elapsed}', Colors.BRIGHT_CYAN)}")
        
        # Loss improvement
        if len(train_losses) > 1:
            improvement = train_losses[0] - train_losses[-1]
            improvement_color = Colors.BRIGHT_GREEN if improvement > 0 else Colors.RED
            print(f"  {self._colorize('📈', Colors.GREEN)} Loss Improvement:     {self._colorize(f'{improvement:+.3f}', improvement_color)}")
        
        print()
        print(self._colorize("=" * 80, Colors.BRIGHT_CYAN))
        print()
    
    def print_debug_info(self, message: str, level: str = "INFO"):
        """Print debug information with appropriate color coding"""
        if level == "ERROR":
            print(f"  {self._colorize('❌', Colors.RED)} {self._colorize(message, Colors.RED)}")
        elif level == "WARNING":
            print(f"  {self._colorize('⚠️', Colors.YELLOW)} {self._colorize(message, Colors.YELLOW)}")
        elif level == "SUCCESS":
            print(f"  {self._colorize('✅', Colors.GREEN)} {self._colorize(message, Colors.GREEN)}")
        else:  # INFO
            print(f"  {self._colorize('ℹ️', Colors.CYAN)} {self._colorize(message, Colors.DIM_CYAN)}")

# Global instance
printer = PrettyPrinter()
