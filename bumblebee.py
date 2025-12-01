import time
from unittest import loader
import torch
from torch_geometric.loader import DataLoader
from utils import set_random_seed
from molecular_data import GraphDataset, ATOM_DIM, BOND_DIM
from model import MAGClassifier
from crossvalidation import *
from explainer import Explainer


# Trainer.fit from pytorch_lightning does the same job
def train(model, loader, optimizer, criterion):
    model.train()  # set training mode
    model = model.to(DEVICE)
    total_loss = 0
    total = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        targets = batch.y.view(-1).to(DEVICE)
        optimizer.zero_grad()  # zero gradients
        logits = model(batch, BATCH_DEBUG)  # forward pass
        loss = criterion(logits, targets)  # calculate loss
        loss.backward()  # backward pass
        optimizer.step()  # update weights
        # statistics
        total_loss += loss.item() * batch.num_graphs
        total += batch.num_graphs
    return total_loss / total

def test(model, loader, criterion):
    model.eval()  # set evaluation mode
    model = model.to(DEVICE)
    total_loss = 0
    metric = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            targets = batch.y.view(-1).to(DEVICE)
            logits = model(batch, BATCH_DEBUG)  # forward pass
            loss = criterion(logits, targets)
            total_loss += loss.item() * batch.num_graphs
            total += batch.num_graphs
            if criterion.task == 'binary_classification':
                preds = (logits > 0.5)
                # preds = (torch.sigmoid(logits) > 0.5)
                metric += (preds == targets).sum().item()  # to compute Accuracy
            else:  # Regression
                metric += torch.sum(torch.abs(logits - targets)).item()  # to compute MAE
    return total_loss / total, metric / total

def training_loop_validation(loader, criterion, val_loader=None):
    print("\nTraining...")
    model = MAGClassifier(ATOM_DIM, BOND_DIM, glob['LAYER_TYPES']).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=glob['LR'])
    val_stats = []
    start_time = time.time()
    for epoch in range(1, glob['NUM_EPOCHS'] + 1):
        loss = train(model, loader, optimizer, criterion)
        print(f"Epoch {epoch}: Loss {loss:.3f}  Time {time.time() - start_time:.0f}s")
        if val_loader is not None and epoch % 5 == 0:            
            val_loss, val_metric = test(model, val_loader, criterion)
            print(f"> VALIDATION  Loss: {val_loss:.3f}  Metric: {val_metric:.3f}")
            val_stats.append(val_metric)
    if val_loader:
        return model, val_stats
    return model

def training_loop(loader, criterion):
    print("\nTraining...")
    model = MAGClassifier(ATOM_DIM, BOND_DIM, glob['LAYER_TYPES']).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=glob['LR'])
    start_time = time.time()
    for epoch in range(1, glob['NUM_EPOCHS'] + 1):
        loss = train(model, loader, optimizer, criterion)
        print(f"Epoch {epoch}: Loss {loss:.3f}  Time {time.time() - start_time:.0f}s")
    return model

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
    # Attention
    att_factor = np.array([aw.max() * len(aw) for aw in train_attn_weights])
    model.stats['attention_factor_mean'] = float(att_factor.mean())
    model.stats['attention_factor_std'] = float(att_factor.std())


def evaluate(model, loader, criterion, flag):
    print("\nEvaluating...")
    loss, metric = test(model, loader, criterion)
    print(f"{flag}: Loss {loss:.3f}  Metric {metric:.3f}")
    return loss, metric

def save(model, path=None):
    if not path:
        path = f"model_{time.strftime('%Y%m%d_%H%M')}.pt"
    ckpt = {
        'state_dict': model.state_dict(),
        'layer_types': getattr(model, 'layer_types'),
        'model_stats': getattr(model, 'stats', None),
    }
    torch.save(ckpt, path)
    print(f"\nModel checkpoint saved to: {path}")

def load(model_path):
    print(f"\nLoading model {model_path}")
    ckpt = torch.load(model_path, map_location=DEVICE)
    layer_types = ckpt.get('layer_types')
    model = MAGClassifier(ATOM_DIM, BOND_DIM, ckpt.get('layer_types')).to(DEVICE)
    model.stats = ckpt.get('model_stats', None)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"Loaded layer_types: {layer_types}")
    return model

def crossvalidation(dataset, criterion, folds=5):
    fold = 1
    fold_results = []
    start_time = time.time()  
    for train_subset, test_subset in cv_subsets(dataset, folds):
        # Reproducibility
        set_random_seed()
        g = torch.Generator()
        g.manual_seed(42)
        train_loader = DataLoader(train_subset, batch_size=glob['BATCH_SIZE'], shuffle=True, drop_last=True)
        test_loader = DataLoader(test_subset, batch_size=glob['BATCH_SIZE'], generator=g)

        print(f"\n{'='*50}\nFold {fold}/{folds}\n{'='*50}")
        print(f"Train size: {len(train_subset)}, Test size: {len(test_subset)}")
        _, val_stats = training_loop_validation(train_loader, criterion, test_loader)
        # loss, metric = evaluate(model, test_loader, criterion, flag=f"Fold {fold+1}")        
        fold_results.append(val_stats)
        fold += 1
    cv_statistics(fold_results, criterion.task)
    print(F"\nTOTAL TIME: {time.time() - start_time:.0f}s")
    print(f"{'='*50}\n")

def explain(model, dataset):
    model.eval()
    explainer = Explainer(model)
    print("\nCALIBRATION")
    print(f"INTEGRATED GRADIENT: {explainer.ig_top:.2f}")
    print(f"ATTENTION FACTOR: {explainer.att_factor_top:.2f}")
    intensity = 1
    for graph in dataset:
        repeat = True
        while repeat:
            explainer.attention(graph.clone(), intensity=intensity)  # why clone()?
            explainer.integrated_gradients(graph.clone(), intensity=intensity)
            # explain_with_mlp_IG(model, graph, intensity=current_intensity)
            # user_input = ''
            user_input = input("Press Enter to continue, '-' to halve intensity, '+' to double intensity: ")
            plus_count = user_input.count('+')
            minus_count = user_input.count('-')
            if plus_count + minus_count > 0:
                intensity *= (2 ** plus_count) / (2 ** minus_count)
                intensity *= (2 ** plus_count) / (2 ** minus_count)
            else:
                repeat = False  # Move to next molecule

def main(dataset_info, cv=False):
    ## Reproducibility
    set_random_seed()
    ## Print model stamp
    import pprint
    pprint.pprint(glob)

    path = dataset_info['path']
    task = dataset_info['task']
    if False:  # task == 'binary_classification':     
        # Default: reduction='mean', return mean loss over batch
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.MSELoss()  # Mean Squared Error for regression
        # criterion = torch.nn.L1Loss()  # Mean Absolute Error
    criterion.task = task

    if cv:
        print(f"\nCross-Validation on: ", path)
        dataset = GraphDataset(dataset_info)
        crossvalidation(dataset, criterion)
        return
  
    testset = None

    if 'split_header' in dataset_info:
        print(f"\nTraining set: {path} ('Training')")
        trainingset = GraphDataset(dataset_info, split='train')
    else:
        trainingset, testset = random_subsets(GraphDataset(dataset_info))
        print(f"\nTraining set: {path} ({len(trainingset)} samples)")

    # Train
    # train_loader = DataLoader(trainingset, batch_size=glob['BATCH_SIZE'], shuffle=True, drop_last=True)
    # model = training_loop(train_loader, criterion)
    # calc_stats(model, train_loader)

    # # Statistics on Training set
    # loader = DataLoader(trainingset, batch_size=glob['BATCH_SIZE'])
    # evaluate(model, loader, criterion, flag="Train")

    ## Save model
    # save(model, "MODEL_muta.pt")

    ## Load saved model
    model = load("MODEL_muta_MSE.pt")

    ## Test)
    if testset is None:
        print(f"\nTest set: {path} ('Test')")
        testset = GraphDataset(dataset_info, split='test')
    else:
        print(f"\nTest set: {path} ({len(testset)} samples)")
    test_loader = DataLoader(testset, batch_size=glob['BATCH_SIZE'])
    evaluate(model, test_loader, criterion, flag="Test")

    # # Explain
    explain(model, testset)
    # single_loader = DataLoader(testset, batch_size=1)
    # explain(model, single_loader)

if __name__ == "__main__":
    ## DEBUG
    BATCH_DEBUG = None
    # BATCH_DEBUG =  True  # Debug: use batch Attention even on CPU
    ## GLOBALS
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    glob = {
        "BATCH_SIZE": 32,  # I should try reducing waste since drop_last=True
        "LR": 1e-4,
        "NUM_EPOCHS": 15,
        "LAYER_TYPES": ['M', 'M', 'S', 'P'],  # 'MMSP'
    }
    import datasets
    print(f"\n{time.strftime("%Y-%m-%d %H:%M:%S")}")
    print(f"DEVICE: {DEVICE}")
    # for glob['LAYER_TYPES'] in (['M0','M0','M0','M0','P'],
    #                             ['M0','M0','M0','S','P'],
    #                             ['M0','M0','S','M0','P'],
    #                             ['M0','M0','S','S','P'],
    #                             ['M0','S','M0','M0','P'],
    #                             ['M0','S','M0','S','P'],
    #                             ['M0','S','S','M0','P'],
    #                             ['M0','S','S','S','P'],
    #                             ['M0', 'M1', 'M2', 'S', 'P'],
    #                         ):
    main(datasets.muta, cv=False)


    ## ESA: README
    # lr = 0.0001
    # BATCH_SIZE = 128
    # HIDDEN_DIM = 256  (= graph_dim)
    # MLP_hidden_dim = 256 (graph-level) or 512 (node-level)
    # NUM_HEADS = 16  (= 4 in the template)
    # LAYER_TYPES = 'MMSP' default; 'MSMSMP' graph-level
    # DROPOUT = 0 !!!!!!!!!
    # weight_decay = 1e-10 nel README (useless)
    ## ESA (hardcoded)
    # PMA seeds = 32
    # model's MLP hidden dimension = 128
    # model's MLP dropout = 0
