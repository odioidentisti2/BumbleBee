import time
from unittest import loader
import torch
from torch_geometric.loader import DataLoader

from molecular_data import GraphDataset, ATOM_DIM, BOND_DIM
from model import MAGClassifier
from explainer import Explainer
import utils
import statistics


def train(model, loader):
    model.train()  # set training mode
    model = model.to(DEVICE)
    total_loss = 0
    total = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        targets = batch.y.view(-1).to(DEVICE)
        model.optimizer.zero_grad()  # zero gradients
        logits = model(batch, BATCH_DEBUG)  # forward pass
        loss = model.criterion(logits, targets)  # calculate loss
        loss.backward()  # backward pass
        model.optimizer.step()  # update weights
        # statistics
        total_loss += loss.item() * batch.num_graphs
        total += batch.num_graphs
    return total_loss / total

def test(model, loader):
    model.eval()  # set evaluation mode
    model = model.to(DEVICE)
    total_loss = 0
    metric = 0
    total = 0
    evaluator = statistics.Evaluator(model.task)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            targets = batch.y.view(-1).to(DEVICE)
            logits = model(batch, BATCH_DEBUG)  # forward pass
            loss = model.criterion(logits, targets)
            total_loss += loss.item() * batch.num_graphs
            total += batch.num_graphs

            # Statistics
            if isinstance(model.criterion, torch.nn.BCEWithLogitsLoss):
                logits = torch.sigmoid(logits)
            evaluator.update(logits, targets, batch.num_graphs)
    metric = evaluator.output()
    return total_loss / total, metric

def training_loop(model, loader, task, val_loader=None):
    setup_training(model, task )
    print("\nTraining...")
    if val_loader: val_stats = []
    start_time = time.time()
    for epoch in range(1, glob['NUM_EPOCHS'] + 1):
        loss = train(model, loader)
        print(f"Epoch {epoch}: Loss {loss:.3f}   ({time.time() - start_time:.0f}s)")
        if val_loader and epoch % 5 == 0:
            loss, metric = evaluate(model, val_loader, flag='> Validation')
            val_stats.append(metric)
    if val_loader:
        return val_stats

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

def evaluate(model, loader, flag):
    if flag[0] != '>': 
        print("\nEvaluating...")
    loss, metric = test(model, loader)
    print(f"{flag}: Loss {loss:.3f}  Metric {metric:.3f}")
    return loss, metric

def save(model, path=None):
    if not path:
        path = f"model_{time.strftime('%Y%m%d_%H%M')}.pt"
    ckpt = {
        'state_dict': model.state_dict(),
        'layer_types': getattr(model, 'layer_types'),
        'task': getattr(model, 'task'),
        'model_stats': getattr(model, 'stats', None),
    }
    torch.save(ckpt, path)
    print(f"\nModel checkpoint saved to: {path}")

def load(model_path):
    print(f"\nLoading model {model_path}")
    ckpt = torch.load(model_path, map_location=DEVICE)
    layer_types = ckpt.get('layer_types')
    model = MAGClassifier(ATOM_DIM, BOND_DIM, ckpt.get('layer_types')).to(DEVICE)
    setup_training(model, ckpt.get('task'))
    model.stats = ckpt.get('model_stats', None)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"Loaded layer_types: {layer_types}")
    return model

def crossvalidation(dataset, task, folds=5):
    fold = 1
    fold_results = []
    start_time = time.time()  
    for train_subset, test_subset in utils.cv_subsets(dataset, folds):
        # Reproducibility
        utils.set_random_seed()
        g = torch.Generator()
        g.manual_seed(42)
        train_loader = DataLoader(train_subset, batch_size=glob['BATCH_SIZE'], shuffle=True, drop_last=True)
        test_loader = DataLoader(test_subset, batch_size=glob['BATCH_SIZE'], generator=g)

        print(f"\n{'='*50}\nFold {fold}/{folds}\n{'='*50}")
        print(f"Train size: {len(train_subset)}, Test size: {len(test_subset)}")
        model = MAGClassifier(ATOM_DIM, BOND_DIM, glob['LAYER_TYPES']).to(DEVICE)
        validation_stats = training_loop(model, train_loader, task, test_loader)
        # loss, metric = evaluate(model, test_loader, flag=f"Fold {fold+1}")        
        fold_results.append(validation_stats)
        fold += 1
    statistics.cv_stats(fold_results)
    print(F"\nTOTAL TIME: {time.time() - start_time:.0f}s")
    print(f"{'='*50}\n")

def explain(model, dataset):
    model.eval()
    explainer = Explainer(model)
    print("\nCALIBRATION")
    print(f"Prediction distribution mean/std: {model.stats['target_mean']:.2f} / {model.stats['target_std']:.2f}")
    print(f"IG top: {explainer.target_std:.2f}")
    print(f"ATT top: {explainer.att_factor_top:.2f}")
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
            else:
                repeat = False  # Move to next molecule

def setup_training(model, task):
    model.task = task
    model.optimizer = torch.optim.AdamW(model.parameters(), lr=glob['LR'])
    if task == 'binary_classification':
        model.criterion = torch.nn.BCEWithLogitsLoss()
    else:
        model.criterion = torch.nn.MSELoss()  # Mean Squared Error for regression
        # model.criterion = torch.nn.L1Loss()  # Mean Absolute Error

def main(dataset_info, cv=False):
    ## Reproducibility
    utils.set_random_seed()
    # print("\nRANDOM SEED = 30")
    ## Print model stamp
    import pprint
    pprint.pprint(glob)

    path = dataset_info['path']
    task = dataset_info['task']

    if cv:
        print(f"\nCross-Validation on: ", path)
        dataset = GraphDataset(dataset_info)
        crossvalidation(dataset, task)
        return

    # Load dataset(s)  
    if 'split_header' in dataset_info:
        print(f"\nTraining set: {path} ('Training')")
        trainingset = GraphDataset(dataset_info, split='train')
        testset = None
    else:
        trainingset, testset = utils.random_subsets(GraphDataset(dataset_info))
        print(f"\nTraining set: {path} ({len(trainingset)} samples)")

    # Train
    # train_loader = DataLoader(trainingset, batch_size=glob['BATCH_SIZE'], shuffle=True, drop_last=True)
    # model = MAGClassifier(ATOM_DIM, BOND_DIM, glob['LAYER_TYPES'])
    # training_loop(model, train_loader, task)
    # calc_stats(model, train_loader)  # Needed for Explainer

    ## Statistics on Training set
    # loader = DataLoader(trainingset, batch_size=glob['BATCH_SIZE'])
    # evaluate(model, loader, flag="Train")

    ## Save model
    # save(model, "MODELS/logp.pt")

    ## Load saved model
    model = load("MODELS/MODEL_logp.pt")

    ## Test
    if testset is None:
        print(f"\nTest set: {path} ('Test')")
        testset = GraphDataset(dataset_info, split='test')
    else:
        print(f"\nTest set: {path} ({len(testset)} samples)")
    test_loader = DataLoader(testset, batch_size=glob['BATCH_SIZE'])
    evaluate(model, test_loader, flag="Test")

    ## Explain
    explain(model, testset)

if __name__ == "__main__":
    ## DEBUG
    BATCH_DEBUG = None
    # BATCH_DEBUG =  True  # Debug: use batch Attention even on CPU
    ## GLOBALS
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    glob = {
        "BATCH_SIZE": 32,  # I should try reducing waste since drop_last=True
        "LR": 1e-4,
        "NUM_EPOCHS": 100,
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
    import attention
    for attention.PMA.K in (1):
        print(f"\n\n### PMA seeds = {attention.PMA.K} ###")
        main(datasets.logp, cv=True)
    # main(datasets.logp, cv=True)


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
