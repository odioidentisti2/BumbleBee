import torch
from utils import print_header


class MetricTracker:
    """Base class for tracking metrics across multiple runs."""
    
    # Subclasses should override this
    values = {}
    
    def __init__(self):
        self.stats = []
        self.values['logits'] = []  # DEBUG
        self.values['predictions'] = []  # DEBUG
    
    def init(self):
        """Start tracking a new run."""
        self.stats.append(self.values.copy())
    
    def update(self, logits, targets):
        """Update statistics for current run."""
        self.stats[-1]['logits'].append(logits)  # DEBUG
    
    def metric(self, index=-1):
        """Compute metric for specific run. Default: last run."""
        pass
    
    def metrics(self):
        """Get metrics for all runs."""
        return [self.metric(i) for i in range(len(self.stats))]


class R2Tracker(MetricTracker):
    """Tracks R² (coefficient of determination) for regression tasks."""
    
    values = {
        'targets_sum': 0.0,
        'targets_squared_sum': 0.0,
        'residuals_squared_sum': 0.0,
        'total_samples': 0
    }
    
    def __init__(self):
        super().__init__()

    def update(self, logits, targets):
        super().update(logits, targets)
        stats = self.stats[-1]
        stats['total_samples'] += len(targets)
        stats['targets_sum'] += targets.sum().item()
        stats['targets_squared_sum'] += (targets ** 2).sum().item()
        stats['residuals_squared_sum'] += ((targets - logits) ** 2).sum().item()

    def metric(self, index=-1):
        run = self.stats[index]
        ss_total = run['targets_squared_sum'] - (run['targets_sum'] ** 2) / run['total_samples']
        return 1 - (run['residuals_squared_sum'] / ss_total) if ss_total != 0 else 0


class AccuracyTracker(MetricTracker):
    
    values = {
        'num_correct': 0,
        'total_samples': 0,
    }

    def __init__(self):
        super().__init__()

    def update(self, logits, targets):
        super().update(logits, targets)
        stats = self.stats[-1]
        preds = (torch.sigmoid(logits) > 0.5)
        stats['predictions'].append(preds)  # DEBUG
        stats['total_samples'] += len(targets)
        stats['num_correct'] += (preds == targets).sum().item()

    def metric(self, index=-1):
        run = self.stats[index]
        return run['num_correct'] / run['total_samples'] if run['total_samples'] > 0 else 0


class CVTracker:
    
    def __init__(self):
        self.fold_results = []
    
    def add_fold(self, fold_metrics):
        self.fold_results.append(fold_metrics)
    
    def compute_statistics(self):
        num_epochs = len(self.fold_results[0])
        
        # Compute mean per epoch across folds
        epoch_means = []
        for epoch_idx in range(num_epochs):
            epoch_values = [fold[epoch_idx] for fold in self.fold_results]
            epoch_means.append(sum(epoch_values) / len(epoch_values))
        
        # Find best epoch
        best_epoch_idx = max(range(len(epoch_means)), key=lambda i: epoch_means[i])
        best_epoch = (best_epoch_idx + 1) * 5
        
        # Get metrics at best epoch across all folds
        metrics_at_best = [fold[best_epoch_idx] for fold in self.fold_results]
        mean_metric = sum(metrics_at_best) / len(metrics_at_best)
        std_metric = (
            sum((x - mean_metric)**2 for x in metrics_at_best) / len(metrics_at_best)
        ) ** 0.5
        
        return {
            'best_epoch': best_epoch,
            'best_epoch_index': best_epoch_idx,
            'mean': mean_metric,
            'std': std_metric,
            'fold_metrics': metrics_at_best,
            'epoch_means': epoch_means
        }
    
    def summary(self):
        stats = self.compute_statistics()
        
        print_header(f"CV RESULTS (Epoch {stats['best_epoch']})")
        print(f"Test metric:   {stats['mean']:.3f} ± {stats['std']:.3f}")
        print(f"\nIndividual fold metrics: {[f'{m:.3f}' for m in stats['fold_metrics']]}")
        
        print_header("MEAN METRICS ACROSS ALL EPOCHS")
        for epoch_idx, mean_val in enumerate(stats['epoch_means']):
            epoch_num = (epoch_idx + 1) * 5
            marker = " ← BEST" if epoch_idx == stats['best_epoch_index'] else ""
            print(f"Epoch {epoch_num:3d}: {mean_val:.3f}{marker}")


# def cv_output(fold_results):
#     """Compute and print cross-validation statistics."""
#     num_epochs = len(fold_results[0])

#     # Compute mean per epoch index
#     mean_metrics = []
#     for i in range(num_epochs):
#         vals = [v[i] for v in fold_results]
#         mean_metrics.append(sum(vals) / len(vals))

#     best_idx = max(range(len(mean_metrics)), key=lambda i: mean_metrics[i])
#     best_epoch = (best_idx + 1) * 5

#     # Get metrics at best epoch
#     metrics_at_best = [v[best_idx] for v in fold_results]
#     mean_metric = sum(metrics_at_best) / len(metrics_at_best)
#     std_metric = (sum((x - mean_metric)**2 for x in metrics_at_best) / len(metrics_at_best)) ** 0.5

#     print(f"\n{'='*50}\nCROSS-VALIDATION RESULTS (Epoch {best_epoch})\n{'='*50}")
#     print(f"Test metric:   {mean_metric:.3f} ± {std_metric:.3f}")
#     print(f"\nIndividual fold metrics: {[f'{m:.3f}' for m in metrics_at_best]}")
    
#     # Print mean metrics across all epochs
#     print(f"\n{'='*50}\nMEAN METRICS ACROSS ALL EPOCHS\n{'='*50}")
#     for epoch_idx, mean_val in enumerate(mean_metrics):
#         epoch_num = (epoch_idx + 1) * 5
#         marker = " ← BEST" if epoch_idx == best_idx else ""
#         print(f"Epoch {epoch_num:3d}: {mean_val:.3f}{marker}")
