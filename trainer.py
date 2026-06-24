import time
import torch
from statistics import AccuracyTracker, R2Tracker
from parameters import train_params as PARAMS
from copy import deepcopy  # early stop


class Trainer:

    def __init__(self, task, device):
        self.device = device
        self.optim = None
        self.task = task
        if self.task == 'binary_classification':
            self.criterion = BinaryHingeLoss()
            self.statistics = AccuracyTracker()
        else:  # regression
            self.criterion = torch.nn.MSELoss()  # Mean Squared Error
            self.statistics =  R2Tracker()
        self.count = 0

    def set_baseline_target(self, targets):
        if PARAMS['inject']:
            if self.task == 'binary_classification':
                self.baseline_target = 0.5  # Decision boundary
            else:
                self.baseline_target = sum(targets) / len(targets)
            print(f"\nBaseline target: {self.baseline_target:.2f}")
        
    def _train(self, model, loader):
        model = model.to(self.device)
        total_loss = 0
        total = 0
        for batch in loader:
            batch = batch.to(self.device)
            targets = batch.y
            if PARAMS['inject']:
                batch = self._injected_batch(batch)  # INJECTION
            logits = model(batch)  # forward pass
            loss = self.criterion(logits, targets)  # calculate loss
            # Learning: zero grad; backward pass; update weights
            self.optim.zero_grad(); loss.backward(); self.optim.step()
            total_loss += loss.item() * batch.num_graphs
            total += batch.num_graphs
        return total_loss / total

    def _eval(self, model, loader):
        self.statistics.init()
        model = model.to(self.device)
        total_loss = 0
        total = 0
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
        self.optim = torch.optim.AdamW(model.parameters(), lr=PARAMS['lr'])
        max_epochs = max(1, PARAMS['max_steps'] // len(loader))

        if val_loader:
            stopper = EarlyStop()
            val_interval = 1  #max(1, round(max_epochs / 100))
            early_stop = True
        else:
            early_stop = False

        print("\nTraining...")
        start_time = time.time()
        for epoch in range(1, max_epochs + 1):
            model.train()  # set training mode      
            loss = self._train(model, loader)
            print(f"Epoch {epoch}: Loss {loss:.3f}   ({time.time() - start_time:.0f}s)")

            # Early stop
            if early_stop and epoch % val_interval == 0:
       
                print("START EVAL!!!!!!!")
                state_before = deepcopy(model.state_dict())
                rng_before = torch.get_rng_state()

                metric = self.eval(model, val_loader, flag='Validation')

                rng_after = torch.get_rng_state()
                print("RNG STATE AFTER EVAL EQUAL?", torch.equal(rng_before, rng_after))
                state_after = model.state_dict()
                                
                for k in state_before:
                    if not torch.equal(state_before[k], state_after[k]):
                        print(f"Changed: {k}")

                if stopper.check(metric, model, epoch):
                    stopper.restore(model)
                    print(f"EARLY STOP: best model epoch {stopper.best_epoch}")
                    break

    def eval(self, model, loader, flag):
        model.eval()  # set evaluation mode
        if flag == 'Test': 
            print("\nTesting...")
        rng_before = torch.get_rng_state()
        loss = self._eval(model, loader)
        rng_after = torch.get_rng_state()
        print("RNG STATE AFTER _eval EQUAL?", torch.equal(rng_before, rng_after))
        metric = self.statistics.metric()
        print(f"> {flag}: Loss {loss:.3f}  Metric {metric:.3f}")
        return metric

    def _injected_batch(self, batch, interval=1000):
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
            batch.y[local_indices] = self.baseline_target
        
        self.count += batch.num_graphs
        return batch
    
    @staticmethod
    def calibration_stats(model, loader):
        """Collect attention weights statistics on training set for Explainer."""
        model = model.to('cpu')
        training_attn_weights = []
        training_predictions = []  # DEBUG
        with torch.no_grad():
            for batch in loader:
                batch = batch.to('cpu')
                preds, attn_weights = model(batch, return_attention=True)
                training_predictions.extend(preds)  # DEBUG
                training_attn_weights.extend(attn_weights)
        training_att_factors = torch.stack([aw.max() * aw.numel() for aw in training_attn_weights])
        model.att_factor_top = training_att_factors.mean().item() + training_att_factors.std().item()
        model.training_predictions = torch.tensor(training_predictions)  # DEBUG


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
