import numpy as np

# Load action targets
data = np.load('data/recording_sessions/20250824_183745/processed/action_targets.npy')
print('Action targets shape:', data.shape)
print('Total elements:', data.size)
print('Non-zero elements:', np.count_nonzero(data))
print('Max value:', np.max(data))
print('Min value:', np.min(data))

# Find non-zero positions
non_zero = np.nonzero(data)
print(f'\nFound {len(non_zero[0])} non-zero elements')

# Look at first few non-zero actions
print('\nFirst 10 non-zero actions:')
for i in range(min(10, len(non_zero[0]))):
    batch_idx = non_zero[0][i]
    action_idx = non_zero[1][i]
    feature_idx = non_zero[2][i]
    value = data[batch_idx, action_idx, feature_idx]
    print(f'Batch {batch_idx}, Action {action_idx}, Feature {feature_idx}: {value}')

# Look at a specific batch with non-zero actions
if len(non_zero[0]) > 0:
    batch_idx = non_zero[0][0]
    print(f'\nExamining batch {batch_idx}:')
    batch_data = data[batch_idx]
    print(f'Batch shape: {batch_data.shape}')
    
    # Find non-zero actions in this batch
    batch_non_zero = np.nonzero(batch_data)
    print(f'Non-zero elements in batch {batch_idx}: {len(batch_non_zero[0])}')
    
    # Show first few non-zero actions
    for i in range(min(5, len(batch_non_zero[0]))):
        action_idx = batch_non_zero[0][i]
        feature_idx = batch_non_zero[1][i]
        value = batch_data[action_idx, feature_idx]
        print(f'  Action {action_idx}, Feature {feature_idx}: {value}')
        
        # Show the full action vector
        if feature_idx == 0:  # If this is the first feature of an action
            action_vector = batch_data[action_idx]
            print(f'    Full action vector: {action_vector}')
