import torch

def cv_statistics(fold_results, task):
    num_epochs = len(fold_results[0])

    # Compute mean per epoch index
    mean_metrics = []
    for i in range(num_epochs):
        vals = [v[i] for v in fold_results]
        mean_metrics.append(sum(vals) / len(vals))

    if task == 'regression':
        best_idx = min(range(len(mean_metrics)), key=lambda i: mean_metrics[i])
    else:  # classification - higher is better
        best_idx = max(range(len(mean_metrics)), key=lambda i: mean_metrics[i])
    best_epoch = (best_idx + 1) * 5

    # Get metrics at best epoch
    metrics_at_best = [v[best_idx] for v in fold_results]
    mean_metric = sum(metrics_at_best) / len(metrics_at_best)
    std_metric = (sum((x - mean_metric)**2 for x in metrics_at_best) / len(metrics_at_best)) ** 0.5

    print(f"\n{'='*50}\nCROSS-VALIDATION RESULTS (Epoch {best_epoch})\n{'='*50}")
    print(f"Test metric:   {mean_metric:.3f} ± {std_metric:.3f}")
    print(f"\nIndividual fold metrics: {[f'{m:.3f}' for m in metrics_at_best]}")


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