import time
import torch
from statistics import AccuracyTracker, R2Tracker
from parameters import trainer_params as PARAMS


class Trainer:

    def __init__(self, task, device):
        self.device = device
        self.optim = None
        if task == 'binary_classification':
            self.criterion = BinaryHingeLoss()
            # self.criterion = torch.nn.BCEWithLogitsLoss()
            self.statistics = AccuracyTracker()
        else:
            self.criterion = torch.nn.MSELoss()  # Mean Squared Error for regression
            # self.criterion = torch.nn.L1Loss()  # Mean Absolute Error
            self.statistics =  R2Tracker()
        self.count = 0

    def set_baseline_target(self, targets):
        self.mean_target = sum(targets) / len(targets)
        print(f"\nMean target in training set: {self.mean_target:.2f}")
        
    def _train(self, model, loader):
        model = model.to(self.device)
        total_loss = 0
        total = 0
        for batch in loader:
            batch = batch.to(self.device)
            targets = batch.y  #.to(self.device)
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
                targets = batch.y  # .to(self.device)
                logits = model(batch)
                loss = self.criterion(logits, targets)
                total_loss += loss.item() * batch.num_graphs
                total += batch.num_graphs
                # Record statistics
                self.statistics.update(logits.detach().cpu(), targets.detach().cpu())
        return total_loss / total

    def train(self, model, loader, val_loader=None):  
        model.train()  # set training mode      
        self.optim = torch.optim.AdamW(model.parameters(), lr=PARAMS['lr'])
        print("\nTraining...")
        start_time = time.time()
        for epoch in range(1, PARAMS['epochs'] + 1):
            loss = self._train(model, loader)
            print(f"Epoch {epoch}: Loss {loss:.3f}   ({time.time() - start_time:.0f}s)")
            if val_loader and epoch % 5 == 0:
                self.eval(model, val_loader, flag='Validation')

    def eval(self, model, loader, flag):
        model.eval()  # set evaluation mode
        if flag == 'Test': 
            print("\nTesting...")
        loss = self._eval(model, loader)
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
            batch.y[local_indices] = 0.5  # self.mean_target
        
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


# class BinaryHingeLoss(torch.nn.Module):
#     def forward(self, pred, target):
#         y = 2 * target - 1  # Convert {0,1} → {-1,+1}
#         loss = torch.clamp(1 - y * pred, min=0)  # max(0, 1 - y*pred)
#         return loss.mean()

# Custom Hinge that handles target=0.5 for baseline injection
class BinaryHingeLoss(torch.nn.Module):
    def forward(self, pred, target):
        y = 2 * target - 1  # Convert {0,1} → {-1,+1}
        loss = torch.zeros_like(pred)
        mask_zero = (y == 0)  # target = 0 -> y = 0
        mask_nonzero = ~mask_zero
        loss[mask_zero] = pred[mask_zero].abs()
        # Push injected samples toward -1
        # loss[mask_zero] = (pred[mask_zero] + 1).abs()
        loss[mask_nonzero] = torch.clamp(1 - y[mask_nonzero] * pred[mask_nonzero], min=0)
        return loss.mean()