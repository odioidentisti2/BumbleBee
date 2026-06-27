import time
import torch
from statistics import AccuracyTracker, R2Tracker
from parameters import train_params as PARAMS
from copy import deepcopy  # early stop


class Trainer:

    def __init__(self, task, device):
        self.device = device
        self.task = task
        self.optim = None
        if self.task == 'binary_classification':
            self.criterion = BinaryHingeLoss()
            self.statistics = AccuracyTracker()
        else:  # regression
            self.criterion = torch.nn.MSELoss()  # Mean Squared Error
            self.statistics =  R2Tracker()
        self.count = 0  # This global used for injection is bad.....

    def set_baseline(self, targets):
        if self.task == 'binary_classification':
            self.baseline = 0.5  # Decision boundary
            # Otherwise, if I use the mean with unbalanced datasets, it can be that in the heat-map there's no red nor green, still it's toxic
        else:
            self.baseline = sum(targets) / len(targets)
        
    def _train(self, model, loader):
        model.train()  # set training mode  
        model = model.to(self.device)
        total_loss = 0
        total = 0
        for batch in loader:
            batch = batch.to(self.device)
            batch = self._injected_batch(batch)  # INJECTION
            targets = batch.y
            logits = model(batch)  # Forward pass
            loss = self.criterion(logits, targets)  # Calculate loss
            self.optim.zero_grad(); loss.backward(); self.optim.step()  # Learning step
            total_loss += loss.item() * batch.num_graphs
            total += batch.num_graphs
        return total_loss / total

    def _eval(self, model, loader):
        model.eval()  # set evaluation mode
        model = model.to(self.device)
        total_loss = 0
        total = 0
        self.statistics.init()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                targets = batch.y
                logits = model(batch)
                # logits, _ = model(batch, return_attention=True)
                loss = self.criterion(logits, targets)
                total_loss += loss.item() * batch.num_graphs
                total += batch.num_graphs
                # Record statistics
                self.statistics.update(logits.detach().cpu(), targets.detach().cpu())
        return total_loss / total

    def train(self, model, loader, val_loader=None):
        model.task = self.task
        self.optim = torch.optim.AdamW(model.parameters(), lr=PARAMS['lr'])
        max_epochs = 100  # max(1, PARAMS['max_steps'] // len(loader))
        val_interval = stopper = None

        # Configuration: validation + early stop
        if val_loader:
            val_interval = 5  # max(1, round(max_epochs / 100))
            if PARAMS['early_stop']:
                stopper = EarlyStop()

        # Baseline injection (for Explainer)
        self.set_baseline(loader.dataset.targets)
        print(f"\nBaseline target: {self.baseline:.2f}")  # DEBUG

        # Training loop
        print("\nTraining...")
        start_time = time.time()
        for epoch in range(1, max_epochs + 1):    
            loss = self._train(model, loader)
            print(f"Epoch {epoch}: Loss {loss:.3f}   ({time.time() - start_time:.0f}s)")

            # Validation + early stop
            if val_interval and epoch % val_interval == 0:
                metric = self.eval(model, val_loader, flag='Validation')
                if stopper and stopper.check(metric, model, epoch):
                    stopper.restore(model)
                    print(f"EARLY STOP: best model epoch {stopper.best_epoch}")
                    break

    def eval(self, model, loader, flag):
        if flag == 'Test': 
            print("\nTesting...")
        loss = self._eval(model, loader)
        metric = self.statistics.metric()
        print(f"> {flag}: Loss {loss:.3f}  Metric {metric:.3f}")
        return metric

    def _injected_batch(self, batch, interval=1000):
        # WHAT IF NUMBER OF SAMPLES IS LESS THAN INTERVAL???
        """Deterministically inject synthetic zero-feature samples every N molecules."""
        global_indices = \
            torch.arange(self.count, self.count + batch.num_graphs) % interval == 0
        local_indices = torch.where(global_indices)[0]
        
        if len(local_indices) > 0:
            # Zero features
            for idx in local_indices:
                graph_mask = (batch.batch == idx)
                batch.x[graph_mask] = 0
                edge_mask = graph_mask[batch.edge_index[0]]
                batch.edge_attr[edge_mask] = 0            
            # Set target for baseline
            batch.y[local_indices] = self.baseline
        
        self.count += batch.num_graphs
        return batch
    
    def calibration(self, model, loader):
        """Collect calibration data on training set for the Explainer."""
        print("\nCalibrating...")
        start_time = time.time()
        model = model.to('cpu')
        # model = model.to(self.device)
        model.eval()
        training_attn_weights = []
        training_predictions = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to('cpu')
                # batch = batch.to(self.device)
                preds, attn_weights = model(batch, return_attention=True)
                training_predictions.append(preds.detach().cpu())
                training_attn_weights.extend([aw.detach().cpu() for aw in attn_weights])
        pred_tensor = torch.cat(training_predictions)
        attn_factors = torch.stack([aw.max() * aw.numel() for aw in training_attn_weights])
        # att_factor_top = attn_factors.mean().item() + attn_factors.std().item()
        calibration = {
            "attn_factor_mean": attn_factors.mean().item(),
            "attn_factor_std": attn_factors.std().item(),
            # "att_factor_top": attn_factors.mean().item() + attn_factors.std().item(),
            "prediction_mean": pred_tensor.mean().item(),
            "prediction_std": pred_tensor.std().item(),
            "prediction_min": pred_tensor.min().item(),
            "prediction_max": pred_tensor.max().item(),
        }
        print(f"Calibration time: {time.time() - start_time:.0f}s")
        return calibration


class EarlyStop:
    def __init__(self):
        self.best_state = None
        self.best_metric = -float("inf")
        self.best_epoch = None
        self.counter = 0
        self.patience = 5
        self.min_delta = 0

    def check(self, metric, model, epoch):
        if metric > self.best_metric + self.min_delta:
            self.best_metric = metric
            self.best_state = deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# Custom Hinge that handles target=0.5 for baseline injection
class BinaryHingeLoss(torch.nn.Module):
    def forward(self, pred, target):
        y = 2 * target - 1  # Convert {0,1} → {-1,+1}
        loss = torch.zeros_like(pred)
        mask_zero = (y == 0)  # target = 0.5 -> y = 0
        mask_nonzero = ~mask_zero
        loss[mask_zero] = pred[mask_zero].abs()  # Push injected samples toward 0       
        # loss[mask_zero] = (pred[mask_zero] + 1).abs()  # Push injected samples toward -1
        loss[mask_nonzero] = torch.clamp(1 - y[mask_nonzero] * pred[mask_nonzero], min=0)
        return loss.mean()

# class BinaryHingeLoss(torch.nn.Module):
#     def forward(self, pred, target):
#         y = 2 * target - 1  # Convert {0,1} → {-1,+1}
#         loss = torch.clamp(1 - y * pred, min=0)  # max(0, 1 - y*pred)
#         return loss.mean()
