#!/usr/bin/env python3
"""
Argument parser for evaluation and training
"""

import argparse

def create_eval_parser():
    """Create argument parser for evaluation"""
    parser = argparse.ArgumentParser("OSRS Imitation Learning Evaluation")
    
    # Data arguments
    parser.add_argument("--data_dir", required=True, help="Directory containing validation data")
    parser.add_argument("--model_path", required=True, help="Path to trained model checkpoint")
    
    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run evaluation on")
    
    # Reporting arguments
    parser.add_argument("--report_examples", type=int, default=6, help="How many raw example rows to print during validation")
    parser.add_argument("--time_clamp_nonneg", action="store_true", help="Clamp time predictions to non-negative values")
    parser.add_argument("--force_exclusive_event_pred", action="store_true", default=True, help="Force exclusive event prediction (no MULTI)")
    parser.add_argument("--topk_print", type=int, default=5, help="k for top-k histograms in the validation report.")
    parser.add_argument("--time_clamp_nonneg", action="store_true",
                       help="Clamp time to >=0 when printing mean_pred; negatives still reported separately.")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory to save evaluation results")
    parser.add_argument("--save_predictions", action="store_true", help="Save raw predictions to file")
    
    return parser

def create_training_parser():
    """Create argument parser for training with additional options"""
    parser = argparse.ArgumentParser("OSRS Imitation Learning Training")
    
    # Data arguments
    parser.add_argument("--data_dir", required=True, help="Directory containing training data")
    parser.add_argument("--targets_version", default=None, choices=[None, "v1", "v2"], help="Targets version to use")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--disable_auto_batch", action="store_true", help="Disable automatic batch size optimization")
    
    # Model arguments
    parser.add_argument("--time_positive", action="store_true", 
                       help="Force non-negative time by applying softplus on the log1p domain.")
    parser.add_argument("--time_head_bias", type=float, default=None,
                       help="Init bias for time head (log1p domain); if unset, uses config default.")
    parser.add_argument("--event_cls_weights", nargs=4, type=float, help="Class weights for event classification [MOVE, CLICK, KEY, SCROLL]")
    parser.add_argument("--exclusive_event", action="store_true",
                       help="Use event argmax to gate sub-heads in reporting / inference; eliminates MULTI.")
    
    # Time arguments
    parser.add_argument("--use_log1p_time", type=lambda s: s.lower()!="false", default=True, help="Use log1p transformation for time")
    parser.add_argument("--time_div_ms", type=float, default=1000.0, help="Time division factor in milliseconds")
    parser.add_argument("--time_clip_s", type=float, default=None, help="Time clipping in seconds")
    
    # Loss weights
    parser.add_argument("--lw_time", type=float, default=0.3, help="Time loss weight")
    parser.add_argument("--lw_x", type=float, default=2.0, help="X coordinate loss weight")
    parser.add_argument("--lw_y", type=float, default=2.0, help="Y coordinate loss weight")
    parser.add_argument("--lw_button", type=float, default=1.0, help="Button loss weight")
    parser.add_argument("--lw_key_action", type=float, default=1.0, help="Key action loss weight")
    parser.add_argument("--lw_key_id", type=float, default=1.0, help="Key ID loss weight")
    parser.add_argument("--lw_scroll_y", type=float, default=1.0, help="Scroll Y loss weight")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on")
    
    return parser
