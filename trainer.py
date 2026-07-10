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
            self.statistics = AccuracyTracker()
        else:  # regression
            self.criterion = torch.nn.MSELoss()  # Mean Squared Error
            self.statistics =  R2Tracker()
        
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
        mean_loss = total_loss / total
        return mean_loss

    def _eval(self, model, loader, return_attention=False):
        model.eval()  # set evaluation mode
        model = model.to(self.device)
        total_loss = 0
        total = 0
        all_attn = []
        self.statistics.init()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                targets = batch.y
                if return_attention:
                    logits, attn_weights = model(batch, return_attention=True)
                    all_attn.extend([aw.detach().cpu() for aw in attn_weights])
                else:
                    logits = model(batch)
                loss = self.criterion(logits, targets)
                total_loss += loss.item() * batch.num_graphs
                total += batch.num_graphs
                # Record statistics
                self.statistics.update(logits.detach().cpu(), targets.detach().cpu())
        mean_loss = total_loss / total
        return (mean_loss, all_attn) if return_attention else mean_loss

    def train(self, model, trainingset, val_set=None):
        model.task = self.task
        self.optim = torch.optim.AdamW(model.parameters(), lr=PARAMS['lr'])
        max_epochs = 10  # max(1, PARAMS['max_steps'] // len(train_set))  # DEBUG
        val_interval = stopper = None

        # Configuration of validation + early stop
        if val_set:
            val_interval = 5  # max(1, round(max_epochs / 100))  # DEBUG
            if PARAMS['early_stop']:
                stopper = EarlyStop()

        # Training loop
        loader = trainingset.get_loader(batch_size=PARAMS['train_batch_size'], is_train=True)
        # loader2 = trainingset.get_loader(batch_size=PARAMS['train_batch_size'], is_train=True)
        start_time = time.time()
        for epoch in range(1, max_epochs + 1):
            loss = self._train(model, loader)
            print(f"Epoch {epoch}: Loss {loss:.3f}   ({time.time() - start_time:.0f}s)")  
            # loss2 = self._eval(model, loader2)
            # print(f"_eval: Loss {loss2:.3f}   ({time.time() - start_time:.0f}s)")

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
        metric = self.statistics.metric()
        print(f"> {flag}: Loss {loss:.3f}  Metric {metric:.3f}")
        return metric
    
    def calibrate(self, model, trainingset):
        loader = trainingset.get_loader(batch_size=OPTIMAL_BATCH_SIZE[self.device.type])
        loss, attn_weights = self._eval(model, loader, return_attention=True)
        logit_tensor = torch.cat(self.statistics.stats[-1]["logits"])
        attn_factors = torch.stack([aw.max() * aw.numel() for aw in attn_weights])
        model.calibration_data = {
            "attn_factor_mean": attn_factors.mean().item(),
            "attn_factor_std": attn_factors.std().item(),
            "logit_mean": logit_tensor.mean().item(),
            "logit_std": logit_tensor.std().item(),
            "logit_min": logit_tensor.min().item(),
            "logit_max": logit_tensor.max().item(),
        }
        metric = self.statistics.metric()
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
