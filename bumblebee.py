import time
import torch
from torch_geometric.loader import DataLoader

from parameters import GLOB
import datasets

from molecular_data import GraphDataset, ATOM_DIM, BOND_DIM
from model import MAG
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
        logits = model(batch, BATCH_DEBUG=BATCH_DEBUG)  # forward pass
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
            logits = model(batch, BATCH_DEBUG=BATCH_DEBUG)  # forward pass
            loss = model.criterion(logits, targets)
            total_loss += loss.item() * batch.num_graphs
            total += batch.num_graphs

            # Statistics
            if model.task == 'binary_classification':
                logits = torch.sigmoid(logits)
            evaluator.update(logits, targets, batch.num_graphs)
    metric = evaluator.output()
    return total_loss / total, metric

def training_loop(model, loader, task, val_loader=None):
    setup_training(model, task )
    print("\nTraining...")
    if val_loader: val_stats = []
    start_time = time.time()
    for epoch in range(1, GLOB['epochs'] + 1):
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
    model.stats['target_min'] = float(targets.min())
    model.stats['target_max'] = float(targets.max())
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
    model = MAG(ATOM_DIM, BOND_DIM, ckpt.get('layer_types')).to(DEVICE)
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
        g.manual_seed(GLOB['random_seed'])
        train_loader = DataLoader(train_subset, batch_size=GLOB['batch_size'], shuffle=True, drop_last=True)
        test_loader = DataLoader(test_subset, batch_size=GLOB['batch_size'], generator=g)

        print(f"\n{'='*50}\nFold {fold}/{folds}\n{'='*50}")
        print(f"Train size: {len(train_subset)}, Test size: {len(test_subset)}")
        model = MAG(ATOM_DIM, BOND_DIM, GLOB['layer_types']).to(DEVICE)
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
    print(f"Prediction range: {model.stats['target_min']:.2f} to {model.stats['target_max']:.2f}")
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
    
class BinaryHingeLoss(torch.nn.Module):
    def forward(self, pred, target):
        y = 2 * target - 1  # Convert {0,1} → {-1,+1}
        loss = torch.clamp(1 - y * pred, min=0)  # max(0, 1 - y*pred)
        return loss.mean()
    
def setup_training(model, task):
    model.task = task
    model.optimizer = torch.optim.AdamW(model.parameters(), lr=GLOB['lr'])
    if task == 'binary_classification':
        model.criterion = BinaryHingeLoss()
        # model.criterion = torch.nn.BCEWithLogitsLoss()
    else:
        model.criterion = torch.nn.MSELoss()  # Mean Squared Error for regression
        # model.criterion = torch.nn.L1Loss()  # Mean Absolute Error

def main(dataset_info, model_name=None, cv=False):
    ## Print model stamp
    import pprint
    pprint.pprint(GLOB)
    ## Reproducibility
    utils.set_random_seed()

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
        print(f"\nLoading dataset: {path}")
        trainingset, testset = utils.random_subsets(GraphDataset(dataset_info))

    if not model_name:  # Train
       
        print(f"\nTraining set: {len(trainingset)} samples")
        train_loader = DataLoader(trainingset, batch_size=GLOB['batch_size'], shuffle=True, drop_last=True)
        model =  MAG(ATOM_DIM, BOND_DIM, GLOB['layer_types'])
        training_loop(model, train_loader, task)
        calc_stats(model, train_loader)  # Needed for Explainer

        ## Statistics on Training set
        # loader = DataLoader(trainingset, batch_size=GLOB['batch_size'])
        # evaluate(model, loader, flag="Train")

        ## Save model
        # save(model, "MODELS/logp_MMM_100e.pt")

    else:  # Load saved model

        model = load(f"MODELS/{model_name}")

    ## Test
    if testset is None:
        print(f"\nTest set: {path} ('Test')")
        testset = GraphDataset(dataset_info, split='test')
    else:
        print(f"\nTest set: {len(testset)} samples")
    test_loader = DataLoader(testset, batch_size=GLOB['batch_size'])
    evaluate(model, test_loader, flag="Test")

    ## Explain
    explain(model, testset)


if __name__ == "__main__":
    ## CUDA reproducibility
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # TF32 (Tensor cores con precisione ridotta)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # cuDNN operations (convolutions, batch norm, etc.)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Forza algoritmi deterministici per operazioni specifiche
    # (scatter, index, embedding backward)
    torch.backends.cudnn.enabled = True
    # torch.use_deterministic_algorithms(True)
    ## GLOBALS
    BATCH_DEBUG =  False  # Debug: use batch Attention even on CPU
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{time.strftime("%Y-%m-%d %H:%M:%S")}")
    print(f"DEVICE: {DEVICE}\n")
    model_name = None
    # model_name = 'logp_MMS_100e.pt'
    # model_name = 'muta_MMM_100e.pt'

    main(datasets.muta, model_name, cv=False)


    


#  TO BE PLACED INSID MAIN TO DEBUG SEED DIVERSITY
    # with torch.no_grad():
    #     pma = model.esa.decoder[-1].attention  # shortcut

    #     seeds = pma.S.squeeze(0)
    #     print("Seed pairwise dist mean:", torch.cdist(seeds, seeds).mean().item())

    #     q = pma.mha.fc_q(seeds)
    #     q_ln = pma.mha.ln_q(q)
    #     print("Query pairwise dist mean:", torch.cdist(q_ln, q_ln).mean().item())

    #     sample = next(iter(DataLoader(testset, batch_size=GLOB['batch_size'])))
    #     sample = sample.to(DEVICE)
    #     q_batch = pma.mha.ln_q(pma.mha.fc_q(pma.S.repeat(sample.num_graphs, 1, 1)))
    #     enc_out = model.get_encoder_output(sample, BATCH_DEBUG)
    #     k_batch = pma.mha.ln_k(pma.mha.fc_k(enc_out))
    #     logits = torch.einsum('bshd,bthd->bhst',
    #                           q_batch.view(sample.num_graphs, -1, pma.mha.num_heads, 128 // pma.mha.num_heads),
    #                           k_batch.view(sample.num_graphs, -1, pma.mha.num_heads, 128 // pma.mha.num_heads))
    #     print("Attention logit std per head:", logits.std(dim=-1).mean().item())

    #     print("Key pairwise distances (first graph, first 10 edges):")
    #     k_sample = k_batch[0, :10]  # [10, num_heads * head_dim]
    #     print(torch.cdist(k_sample, k_sample).mean().item())
        
    #     print("\nEncoder output variance per edge (first graph):")
    #     print(enc_out[0].var(dim=0).mean().item())  # variance across feature dims

    #     return

