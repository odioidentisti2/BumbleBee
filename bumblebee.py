# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from molecular_data import MoleculeDataset


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.ln0 = nn.LayerNorm(dim_V)
        self.ln1 = nn.LayerNorm(dim_V)
        self.fc_r = nn.Sequential(
            nn.Linear(dim_V, dim_V),
            nn.ReLU(),
            nn.Linear(dim_V, dim_V)
        )

    def forward(self, Q, K):
        Q_ = self.fc_q(Q)
        K_ = self.fc_k(K)
        V_ = self.fc_v(K)
        Q_ = torch.cat(Q_.split(Q_.size(-1) // self.num_heads, dim=2), dim=0)
        K_ = torch.cat(K_.split(K_.size(-1) // self.num_heads, dim=2), dim=0)
        V_ = torch.cat(V_.split(V_.size(-1) // self.num_heads, dim=2), dim=0)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / (K_.size(-1) ** 0.5), dim=2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), dim=0), dim=2)
        O = self.ln0(O)
        O = O + self.fc_r(O)
        return self.ln1(O)

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads)
    def forward(self, X):
        return self.mab(X, X)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads)
    def forward(self, X):
        S = self.S.repeat(X.size(0), 1, 1)
        return self.mab(S, X)

class ESAMoleculeClassifier(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=128, output_dim=1, num_heads=4, num_inds=1):
        super().__init__()
        self.node_proj = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.edge_proj = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.edge_set_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.encoder = nn.Sequential(
            SAB(hidden_dim, hidden_dim, num_heads),
            SAB(hidden_dim, hidden_dim, num_heads),
            SAB(hidden_dim, hidden_dim, num_heads)
        )
        self.pooling = PMA(hidden_dim, num_heads, num_inds)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_inds, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, batch_data):
        edge_feats_list = []
        for data in batch_data:
            x, edge_index, edge_attr = data['x'], data['edge_index'], data['edge_attr']
            h = self.node_proj(x)
            e = self.edge_proj(edge_attr)
            src, dst = edge_index
            edge_inputs = torch.cat([h[src], h[dst], e], dim=1)
            edge_feats = self.edge_set_proj(edge_inputs)
            edge_feats_list.append(edge_feats)

        max_edges = max(f.size(0) for f in edge_feats_list)
        d = edge_feats_list[0].size(1)
        B = len(edge_feats_list)
        E = torch.zeros(B, max_edges, d, device=edge_feats_list[0].device)
        for i, f in enumerate(edge_feats_list):
            E[i, :f.size(0)] = f

        for layer in self.encoder:
            E = layer(E)

        pooled = self.pooling(E)
        logits = self.classifier(pooled.view(pooled.size(0), -1)).squeeze(-1)
        return logits

def collate_fn(batch):
    return batch

def train(model, dataset, optimizer, criterion, epoch):
    model.train()  # set training mode
    total_loss = 0
    correct = 0
    total = 0
    for batch in dataset.batch_iter(batch_size=BATCH_SIZE):
        for item in batch:
            item['x'] = item['x'].to(DEVICE)
            item['edge_index'] = item['edge_index'].to(DEVICE)
            item['edge_attr'] = item['edge_attr'].to(DEVICE)
            item['y'] = item['y'].to(DEVICE)
        targets = torch.cat([item['y'] for item in batch])
        optimizer.zero_grad()  # zero gradients
        logits = model(batch)  # forward pass
        loss = criterion(logits, targets)  # calculate loss
        loss.backward()  # backward pass
        optimizer.step()  # update weights

        # TEST UNIT
        if epoch == 1 and total == 0:  # only check the first batch
            import hashlib
            previous_hash = "e50a7f6ae6d8978b7c2a202538276ca6"  # None 
            output_hash = hashlib.md5(str(logits.detach().cpu().numpy().mean()).encode()).hexdigest()
            if previous_hash is not None and output_hash == previous_hash:
                print("TEST consistency: OK")
            else:
                print(f"Output hash: {output_hash}")

        # calculate accuracy
        total_loss += loss.item() * len(batch)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == targets).sum().item()
        total += len(batch)
    return total_loss / total, correct / total

def main():
    print("\nGPU:", torch.cuda.is_available())
    dataset = MoleculeDataset('DATASETS/MUTA_SARPY_4204.csv')
    model = ESAMoleculeClassifier(dataset.node_dim, dataset.edge_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, EPOCHS+1):
        loss, acc = train(model, dataset, optimizer, criterion, epoch)
        print(f"Epoch {epoch:02d} - Loss: {loss:.4f} - Accuracy: {acc:.4f}")

    print("Training complete.")
    return model


BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Set seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

main()