import torch

class Evaluator:
    def __init__(self, task):
        self.task = task
        self.stats = {
            'num_correct': 0,
            'targets_sum': 0.0,
            'targets_squared_sum': 0.0,
            'residuals_squared_sum': 0.0,
            'total_samples': 0
        }

    def update(self, logits, targets, num_graphs):
        self.stats['total_samples'] += num_graphs
        if self.task == 'binary_classification':
            preds = (logits > 0.5)
            self.stats['num_correct'] += (preds == targets).sum().item()
        else:  # Regression
            self.stats['targets_sum'] += targets.sum().item()
            self.stats['targets_squared_sum'] += (targets ** 2).sum().item()
            self.stats['residuals_squared_sum'] += ((targets - logits) ** 2).sum().item()

    def output(self):
        if self.task == 'binary_classification':
            return self.stats['num_correct'] / self.stats['total_samples'] if self.stats['total_samples'] > 0 else 0
        else:  # Regression
            mean_targets = self.stats['targets_sum'] / self.stats['total_samples']
            ss_total = self.stats['targets_squared_sum'] - (self.stats['targets_sum'] ** 2) / self.stats['total_samples']
            return 1 - (self.stats['residuals_squared_sum'] / ss_total) if ss_total != 0 else 0


def cv_stats(fold_results):
    num_epochs = len(fold_results[0])

    # Compute mean per epoch index
    mean_metrics = []
    for i in range(num_epochs):
        vals = [v[i] for v in fold_results]
        mean_metrics.append(sum(vals) / len(vals))

    best_idx = max(range(len(mean_metrics)), key=lambda i: mean_metrics[i])
    best_epoch = (best_idx + 1) * 5

    # Get metrics at best epoch
    metrics_at_best = [v[best_idx] for v in fold_results]
    mean_metric = sum(metrics_at_best) / len(metrics_at_best)
    std_metric = (sum((x - mean_metric)**2 for x in metrics_at_best) / len(metrics_at_best)) ** 0.5

    print(f"\n{'='*50}\nCROSS-VALIDATION RESULTS (Epoch {best_epoch})\n{'='*50}")
    print(f"Test metric:   {mean_metric:.3f} ± {std_metric:.3f}")
    print(f"\nIndividual fold metrics: {[f'{m:.3f}' for m in metrics_at_best]}")
    
    # Print mean metrics across all epochs
    print(f"\n{'='*50}\nMEAN METRICS ACROSS ALL EPOCHS\n{'='*50}")
    for epoch_idx, mean_val in enumerate(mean_metrics):
        epoch_num = (epoch_idx + 1) * 5
        marker = " ← BEST" if epoch_idx == best_idx else ""
        print(f"Epoch {epoch_num:3d}: {mean_val:.3f}{marker}")
