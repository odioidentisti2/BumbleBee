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