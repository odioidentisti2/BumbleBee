import torch
from parameters import GLOB



# # This specifically fixes attention backward non-determinism
# # but causes 10-20% slowdown in training (only the last row or also the first 2?)
# if DEVICE.type == 'cuda':
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     # This forces deterministic algorithms in attention operations
#     torch.use_deterministic_algorithms(True, warn_only=True)

def set_random_seed():
    seed = GLOB.get('random_seed')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cv_subsets(dataset, num_folds):
    dataset_size = len(dataset)
    indices = torch.randperm(dataset_size).tolist()
    fold_size = dataset_size // num_folds 
    for fold in range(num_folds):
        # Create indices
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < num_folds - 1 else dataset_size
        test_indices = indices[test_start:test_end]
        train_indices = indices[:test_start] + indices[test_end:]        
        # Create subsets
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        test_subset = torch.utils.data.Subset(dataset, test_indices)
        yield train_subset, test_subset

def random_subsets(dataset, test_fraction=0.2):
    dataset_size = len(dataset)
    indices = torch.randperm(dataset_size).tolist()
    test_size = int(dataset_size * test_fraction)
    # Create indices
    test_indices = indices[0:test_size]
    train_indices = indices[test_size:]        
    # Create subsets
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    return train_subset, test_subset

# import csv

# def write_csv_from_subsets(train_subset, test_subset, filename):
#     with open(filename, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["SMILES", "Experimental value", "Status"])
        
#         # Write training data
#         for idx in train_subset.indices:
#             data = train_subset.dataset[idx]
#             writer.writerow([data.smiles, data.y.item(), "Training"])
        
#         # Write test data
#         for idx in test_subset.indices:
#             data = test_subset.dataset[idx]
#             writer.writerow([data.smiles, data.y.item(), "Test"])