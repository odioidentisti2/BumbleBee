import time
import torch
from statistics import AccuracyTracker, R2Tracker
from parameters import trainer_params as PARAMS


class Trainer:

    def __init__(self, task, device):
        self.device = device
        self.opt = None
        if task == 'binary_classification':
            self.criterion = BinaryHingeLoss()
            # self.criterion = torch.nn.BCEWithLogitsLoss()
            self.statistics = AccuracyTracker()
        else:
            self.criterion = torch.nn.MSELoss()  # Mean Squared Error for regression
            # self.criterion = torch.nn.L1Loss()  # Mean Absolute Error
            self.statistics =  R2Tracker()
        self.count = 0

    def _injected_batch(self, batch, target, interval=100):
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
            # Set targets for baseline
            batch.y[local_indices] = target
        
        self.count += batch.num_graphs
        return batch
        
    def _train(self, model, loader):
        model = model.to(self.device)
        total_loss = 0
        total = 0
        for batch in loader:
            batch = batch.to(self.device)
            targets = batch.y.view(-1).to(self.device)
            batch = self._injected_batch(batch, 2.11)  # model.stats['target_mean'])
            # batch = self._injected_batch(batch, 0.5)  # Classification
            logits = model(batch)  # forward pass
            loss = self.criterion(logits, targets)  # calculate loss
            # Learning: zero grad; backward pass; update weights
            self.opt.zero_grad(); loss.backward(); self.opt.step()
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
                targets = batch.y.view(-1).to(self.device)
                logits = model(batch)
                loss = self.criterion(logits, targets)
                total_loss += loss.item() * batch.num_graphs
                total += batch.num_graphs
                # Record statistics
                self.statistics.update(logits.detach().cpu(), targets.detach().cpu())
        return total_loss / total

    def train(self, model, loader, val_loader=None):  
        model.train()  # set training mode      
        self.opt = torch.optim.AdamW(model.parameters(), lr=PARAMS['lr'])
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
    
    @staticmethod  # Experimental
    def calc_stats(model, calibration_loader):
        model = model.to('cpu')
        predictions = []
        train_attn_weights = []
        with torch.no_grad():
            for batch in calibration_loader:
                batch = batch.to('cpu')
                preds, attn_weights = model(batch, return_attention=True)
                predictions.extend(preds)
                train_attn_weights.extend(attn_weights)
        import numpy as np
        model.stats = {}
        # IG
        targets = np.array(predictions)
        model.stats['target_mean'] = float(targets.mean())
        model.stats['target_std'] = float(targets.std())
        model.stats['target_min'] = float(targets.min())
        model.stats['target_max'] = float(targets.max())
        # Attention
        att_factor = np.array([aw.max() * len(aw) for aw in train_attn_weights])
        model.stats['attention_factor_mean'] = float(att_factor.mean())
        model.stats['attention_factor_std'] = float(att_factor.std())



class BinaryHingeLoss(torch.nn.Module):
    def forward(self, pred, target):
        y = 2 * target - 1  # Convert {0,1} → {-1,+1}
        loss = torch.clamp(1 - y * pred, min=0)  # max(0, 1 - y*pred)
        return loss.mean()