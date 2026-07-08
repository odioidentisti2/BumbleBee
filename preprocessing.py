import torch
import reproducibility


def cv_subsets(dataset_size, num_folds):
    g = reproducibility.torch_generator()
    indices = torch.randperm(dataset_size, generator=g).tolist()
    fold_size = dataset_size // num_folds
    for fold in range(num_folds):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < num_folds - 1 else dataset_size
        test_indices = indices[test_start:test_end]
        train_indices = indices[:test_start] + indices[test_end:]
        yield train_indices, test_indices

def random_subsets(dataset_size, test_fraction=0.2):
    g = reproducibility.torch_generator()
    indices = torch.randperm(dataset_size, generator=g).tolist()
    test_size = int(dataset_size * test_fraction)
    # Create indices
    test_indices = indices[0:test_size]
    train_indices = indices[test_size:]  
    return train_indices, test_indices

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