import time
import torch
from statistics import AccuracyTracker, R2Tracker
from parameters import OPTIMAL_BATCH_SIZE, train_params as PARAMS
from copy import deepcopy  # early stop


class Trainer:

    def __init__(self, task, device):
        self.device = device
        self.task = task
        self.optim = None
        if self.task == 'binary_classification':
            self.criterion = BinaryHingeLoss()
            self.tracker = AccuracyTracker()
        else:  # regression
            self.criterion = torch.nn.MSELoss()  # Mean Squared Error
            self.tracker =  R2Tracker()
        
    def _train(self, model, loader):
        model.train()  # set training mode  
        model = model.to(self.device)
        total_loss = 0
        total = 0
        for batch in loader:
            batch = batch.to(self.device)
            targets = batch.y
            logits = model(batch)  # Forward pass
            loss = self.criterion(logits, targets)  # Calculate loss
            self.optim.zero_grad(); loss.backward(); self.optim.step()  # Learning step
            total_loss += loss.item() * batch.num_graphs
            total += batch.num_graphs
        return total_loss / total  # mean loss

    def _eval(self, model, loader):
        model.eval()  # set evaluation mode
        model = model.to(self.device)
        total_loss = 0
        total = 0
        self.tracker.init()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                targets = batch.y
                logits = model(batch)
                loss = self.criterion(logits, targets)
                total_loss += loss.item() * batch.num_graphs
                total += batch.num_graphs
                # Record progress
                self.tracker.update(logits.detach().cpu(), targets.detach().cpu())
        return total_loss / total  # mean loss

    def train(self, model, trainingset, val_set=None):
        model.task = self.task
        self.optim = torch.optim.AdamW(model.parameters(), lr=PARAMS['lr'])
        max_epochs = max(1, PARAMS['max_steps'] * PARAMS['train_batch_size'] // len(trainingset))  # DEBUG

        # Configuration of validation + early stop
        val_interval = stopper = None
        if val_set:
            val_interval = max(1, round(max_epochs / 100))  # DEBUG
            if PARAMS['early_stop']:
                stopper = EarlyStop()

        # Training loop
        loader = trainingset.get_loader(batch_size=PARAMS['train_batch_size'], is_train=True)
        start_time = time.time()
        for epoch in range(1, max_epochs + 1):
            loss = self._train(model, loader)
            print(f"Epoch {epoch}: Loss {loss:.3f}   ({time.time() - start_time:.0f}s)")

            # Validation + early stop
            if val_interval and epoch % val_interval == 0:
                metric = self.eval(model, val_set, flag='Validation')
                if stopper and stopper.check(metric, model, epoch):
                    stopper.restore(model)
                    print(f"EARLY STOP: best model epoch {stopper.best_epoch}")
                    break

    def eval(self, model, testset, flag):
        loader = testset.get_loader(batch_size=OPTIMAL_BATCH_SIZE[self.device.type])
        loss = self._eval(model, loader)
        metric = self.tracker.metric()
        print(f"> {flag}: Loss {loss:.3f}  Metric {metric:.3f}")
        return metric
    
    def calibrate(self, model, trainingset):
        model.track_attention()
        loader = trainingset.get_loader(batch_size=OPTIMAL_BATCH_SIZE[self.device.type])
        loss = self._eval(model, loader)
        logit_tensor = torch.cat(self.tracker.stats[-1]["logits"])  # It gets logits from tracker (not very clean)
        attn_factors = torch.stack([aw.max() * aw.numel() for aw in model._attention_store])
        model.track_attention(enable=False)
        model.calibration_data = {
            "attn_factor_mean": attn_factors.mean().item(),
            "attn_factor_std": attn_factors.std().item(),
            "logit_mean": logit_tensor.mean().item(),
            "logit_std": logit_tensor.std().item(),
            "logit_min": logit_tensor.min().item(),
            "logit_max": logit_tensor.max().item(),
        }

        metric = self.tracker.metric()
        print(f"> Train: Loss {loss:.3f}  Metric {metric:.3f}")
        return model.calibration_data


class EarlyStop:
    def __init__(self):
        self.best_state = None
        self.best_metric = -float("inf")
        self.best_epoch = None
        self.counter = 0
        self.patience = 5
        self.min_delta = 0

    # def check(self, metric, model, epoch):
    #     improvement = metric > self.best_metric + self.min_delta
        
    #     print(f"  [EarlyStop] Epoch {epoch}: metric={metric:.4f}, best={self.best_metric:.4f}, "
    #         f"improvement={improvement}, counter={self.counter}/{self.patience}")
        
    #     if improvement:
    #         self.best_metric = metric
    #         self.best_state = deepcopy(model.state_dict())
    #         self.best_epoch = epoch
    #         self.counter = 0
    #         print(f"  [EarlyStop] ✓ New best! Counter reset to 0")
    #     else:
    #         self.counter += 1
    #         print(f"  [EarlyStop] ✗ No improvement. Counter: {self.counter}/{self.patience}")
        
    #     should_stop = self.counter >= self.patience
    #     if should_stop:
    #         print(f"  [EarlyStop] STOPPING NOW! Patience {self.patience} reached")
        
    #     return should_stop

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
