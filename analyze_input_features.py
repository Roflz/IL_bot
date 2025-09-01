#!/usr/bin/env python3
import numpy as np

# Load and analyze the input data in detail
data_dir = 'data/recording_sessions/20250831_113719/06_final_training_data'
gamestate = np.load(f'{data_dir}/gamestate_sequences.npy')
action_input = np.load(f'{data_dir}/action_input_sequences.npy')

print('=== GAMESTATE SEQUENCES ANALYSIS ===')
print(f'Shape: {gamestate.shape}')
print(f'Overall range: {gamestate.min():.1f} to {gamestate.max():.1f}')
print(f'Overall mean: {gamestate.mean():.1f}, std: {gamestate.std():.1f}')

# Analyze each feature dimension
print(f'\nPer-feature analysis (first 10 features):')
for i in range(min(10, gamestate.shape[-1])):
    feature_data = gamestate[..., i]
    non_zero = feature_data[feature_data != 0]
    print(f'  Feature {i:2d}: range {feature_data.min():8.1f} to {feature_data.max():8.1f}, mean {feature_data.mean():8.1f}, std {feature_data.std():8.1f}')
    if len(non_zero) > 0:
        print(f'           non-zero: {non_zero.min():8.1f} to {non_zero.max():8.1f}, mean {non_zero.mean():8.1f}')

print(f'\n=== ACTION INPUT SEQUENCES ANALYSIS ===')
print(f'Shape: {action_input.shape}')
print(f'Overall range: {action_input.min():.1f} to {action_input.max():.1f}')
print(f'Overall mean: {action_input.mean():.1f}, std: {action_input.std():.1f}')

# Analyze each feature dimension
print(f'\nPer-feature analysis (all 7 features):')
for i in range(action_input.shape[-1]):
    feature_data = action_input[..., i]
    non_zero = feature_data[feature_data != 0]
    print(f'  Feature {i:2d}: range {feature_data.min():8.1f} to {feature_data.max():8.1f}, mean {feature_data.mean():8.1f}, std {feature_data.std():8.1f}')
    if len(non_zero) > 0:
        print(f'           non-zero: {non_zero.min():8.1f} to {non_zero.max():8.1f}, mean {non_zero.mean():8.1f}')

# Check for special values
print(f'\n=== SPECIAL VALUES ANALYSIS ===')
print(f'Gamestate -1 values: {(gamestate == -1).sum()} ({(gamestate == -1).mean()*100:.1f}%)')
print(f'Action input -1 values: {(action_input == -1).sum()} ({(action_input == -1).mean()*100:.1f}%)')
print(f'Gamestate 0 values: {(gamestate == 0).sum()} ({(gamestate == 0).mean()*100:.1f}%)')
print(f'Action input 0 values: {(action_input == 0).sum()} ({(action_input == 0).mean()*100:.1f}%)')
