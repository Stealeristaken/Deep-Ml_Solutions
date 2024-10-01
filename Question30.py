"""

"""

import numpy as np

import numpy as np

def batch_iterator(X, batch_size, y=None, shuffle=False):
    n_samples = X.shape[0]
    
    if shuffle:
        indices = np.random.permutation(n_samples)
    else:
        indices = np.arange(n_samples)
    
    batches = []
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        if y is not None:
            batches.append((X[batch_indices].tolist(), y[batch_indices].tolist()))
        else:
            batches.append(X[batch_indices].tolist())
    
    return batches