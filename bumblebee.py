import time
import torch
from torch_geometric.loader import DataLoader
import numpy as np

from molecular_data import GraphDataset, ATOM_DIM, BOND_DIM
from trainer import Trainer
from model import MAG
from explainer import Explainer
import utils
import statistics

import datasets
from parameters import main_params as PARAMS

from scipy.stats import pearsonr


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save(model, path=None):
    if not path:
        path = f"model_{time.strftime('%Y%m%d_%H%M')}.pt"
    ckpt = {
        'state_dict': model.state_dict(),
        'att_factor_top': getattr(model, 'att_factor_top'),
        'training_predictions': getattr(model, 'training_predictions', None),  # DEBUG
    }
    torch.save(ckpt, path)
    print(f"\nModel checkpoint saved to: {path}")

def load(model_path, device):
    print(f"\nLoading model {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    model = MAG(ATOM_DIM, BOND_DIM).to(device)
    model.att_factor_top = ckpt.get('att_factor_top')
    model.training_predictions = ckpt.get('training_predictions')  # DEBUG
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model

def crossvalidation(trainer, dataset_info, folds=5):
    print(f"\nCross-Validation on: ", dataset_info['path'])
    dataset = GraphDataset(dataset_info)
    from preprocessing import cv_subsets
    cv_tracker = statistics.CVTracker()
    start_time = time.time()
    
    for fold, (train_subset, test_subset) in enumerate(cv_subsets(dataset, folds), start=1):
        utils.print_header(f"Fold {fold}/{folds}")
        print(f"Train size: {len(train_subset)}, Test size: {len(test_subset)}")
        # Reproducibility
        set_random_seed(PARAMS['random_seed'])
        g = torch.Generator()
        g.manual_seed(PARAMS['random_seed'])

        train_loader = DataLoader(train_subset, batch_size=PARAMS['batch_size'], shuffle=True, drop_last=True)
        test_loader = DataLoader(test_subset, batch_size=PARAMS['batch_size'], generator=g)
        
        model = MAG(ATOM_DIM, BOND_DIM)
        trainer.train(model, train_loader, val_loader=test_loader)   
        cv_tracker.add_fold(trainer.statistics.metrics())    
    
    cv_tracker.summary()  # Print summary    
    print(f"\nTOTAL TIME: {time.time() - start_time:.0f}s")


def main(device, dataset_info, model_name=None, cv=False):
    print('MODEL PARAMETERS:')
    import parameters, pprint
    for name in dir(parameters):
        if name.endswith('_params'):
            pprint.pprint(getattr(parameters, name))

    ## Reproducibility
    set_random_seed(PARAMS['random_seed'])

    trainer = Trainer(dataset_info['task'], device)

    if cv:
        crossvalidation(trainer, dataset_info)
        return

    if not model_name:  # Train model
        ## Load training set
        trainingset = GraphDataset(dataset_info, split=dataset_info['train_split'])
        train_loader = DataLoader(trainingset, batch_size=PARAMS['batch_size'], shuffle=True, drop_last=True)

        ## Train model
        model =  MAG(ATOM_DIM, BOND_DIM)
        trainer.mean_target = np.mean(trainingset.targets)  # For injection baseline  (why np???)
        print(f"\nMean target in training set: {trainer.mean_target:.2f}")
        trainer.train(model, train_loader)
        trainer.calibration_stats(model, train_loader)  # Needed for Explainer

        ## Statistics on Training set
        # loader = DataLoader(trainingset, batch_size=PARAMS['batch_size'])
        # trainer.eval(model, loader, flag="Train")

        ## Save model
        save(model, "MODELS/logp_rand15.pt")

    else:  # Load saved model
        model = load(f"MODELS/{model_name}", device)
        # assert model.task == trainer.task

    ## Test
    testset = GraphDataset(dataset_info, split=dataset_info['test_split'])
    test_loader = DataLoader(testset, batch_size=PARAMS['batch_size'])
    trainer.eval(model, test_loader, flag="Test")

    ## Explain
    # explain(model, testset)
    explainer = Explainer(model)
    aw, ig = explainer.explain(testset)

    return aw, ig

if __name__ == "__main__":
    print(f"\n{time.strftime("%Y-%m-%d %H:%M:%S")}")

    ## CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEVICE: {device}\n")

    ## Reproducibility
    # if device.type == 'cuda':
    #     import os
    #     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(True)

    model_name = None
    # model_name = 'logp_rand42.pt'
    # model_name = 'muta_benchmark.pt'
    
    # print("RANDOM = 15\n")
    # main(device, datasets.logp_split, model_name, cv=False)



import torch.nn.functional as F

def robustness(list1, list2):
    """Cosine similarity between two lists of tensors."""
    assert len(list1) == len(list2), "Lists must have same length"
    
    per_sample_robustness = []
    for t1, t2 in zip(list1, list2):
        assert len(t1) == len(t2), f"Length mismatch: {len(t1)} vs {len(t2)}"
        sim = F.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0)).item()
        per_sample_robustness.append(sim)
    
    mean_robustness = np.mean(per_sample_robustness)
    std_robustness = np.std(per_sample_robustness)
    return per_sample_robustness, mean_robustness, std_robustness


def robustness_pearson(list1, list2):
    per_sample_robustness = []
    for t1, t2 in zip(list1, list2):
        assert len(t1) == len(t2), f"Length mismatch: {len(t1)} vs {len(t2)}"
        corr, _ = pearsonr(t1.cpu().numpy(), t2.cpu().numpy())
        per_sample_robustness.append(corr)
    
    mean_robustness = np.mean(per_sample_robustness)
    std_robustness = np.std(per_sample_robustness)
    return per_sample_robustness, mean_robustness, std_robustness


def robustness_mae(list1, list2):
    per_sample_robustness = []
    for t1, t2 in zip(list1, list2):
        assert len(t1) == len(t2), f"Length mismatch: {len(t1)} vs {len(t2)}"
        mae = torch.abs(t1 - t2).mean().item()
        per_sample_robustness.append(mae)
    
    mean_robustness = np.mean(per_sample_robustness)
    std_robustness = np.std(per_sample_robustness)
    return per_sample_robustness, mean_robustness, std_robustness


aw42, ig42 = main(device, datasets.muta, 'logp_rand42.pt')
aw15, ig15 = main(device, datasets.muta, 'logp_rand15.pt')
aw42_inj, ig42_inj = main(device, datasets.logp_split, 'logp_rand42_inj.pt')
aw15_inj, ig15_inj = main(device, datasets.logp_split, 'logp_rand15_inj.pt')

per_sample_cos, mean_cos, std_cos = robustness(ig42, ig15)
per_sample_pearson, mean_pearson, std_pearson = robustness_pearson(ig42, ig15)
per_sample_mae, mean_mae, std_mae = robustness_mae(ig42, ig15)

per_sample_cos_inj, mean_cos_inj, std_cos_inj = robustness(ig42_inj, ig15_inj)
per_sample_pearson_inj, mean_pearson_inj, std_pearson_inj = robustness_pearson(ig42_inj, ig15_inj)
per_sample_mae_inj, mean_mae_inj, std_mae_inj = robustness_mae(ig42_inj, ig15_inj)

print(f"IG robustness (ig42 vs ig15):")
print(f"  Cosine Similarity:  {mean_cos:.4f} ± {std_cos:.4f}")
print(f"  Pearson Correlation: {mean_pearson:.4f} ± {std_pearson:.4f}")
print(f"  MAE:                {mean_mae:.4f} ± {std_mae:.4f}")

print(f"\nIG robustness with injection (ig42_inj vs ig15_inj):")
print(f"  Cosine Similarity:  {mean_cos_inj:.4f} ± {std_cos_inj:.4f}")
print(f"  Pearson Correlation: {mean_pearson_inj:.4f} ± {std_pearson_inj:.4f}")
print(f"  MAE:                {mean_mae_inj:.4f} ± {std_mae_inj:.4f}")


    # m1 = main(device, datasets.muta, 'muta_benchmark.pt')
    # l1 = torch.cat(m1.stats[-1]['logits'])

    # m2 = main(device, datasets.muta, 'muta_RAND30.pt')
    # l2 = torch.cat(m2.stats[-1]['logits'])

    # logits_close = torch.allclose(l1, l2, rtol=1e-5, atol=1e-8)
    # logit_diff = (l1 - l2).abs()
    # print(f"Logits close: {logits_close}")
    # print(f"Logits difference: min={logit_diff.min().item():.6f}, max={logit_diff.max().item():.6f}, mean={logit_diff.mean().item():.6f}, std={logit_diff.std().item():.6f}   \n")
    
    # # Binary predictions
    # pred1 = torch.cat(m1.stats[-1]['predictions'])
    # pred2 = torch.cat(m2.stats[-1]['predictions'])
    # agree = sum(p1 == p2 for p1, p2 in zip(pred1, pred2))
    # print(f"\nModels agreement: {agree} / {len(pred1)}")
    # diff_idx = (pred1 != pred2).nonzero(as_tuple=True)[0]
    # print(f"Discordant: {diff_idx.numel()}") or (print("\n".join(
    #     f"[{i}] p1={int(pred1[i])} p2={int(pred2[i])} l1={l1[i].tolist()} l2={l2[i].tolist()}"
    #     for i in diff_idx[:20].tolist())))


#  TO BE PLACED INSID MAIN TO DEBUG SEED DIVERSITY
    # with torch.no_grad():
    #     pma = model.esa.decoder[-1].attention  # shortcut

    #     seeds = pma.S.squeeze(0)
    #     print("Seed pairwise dist mean:", torch.cdist(seeds, seeds).mean().item())

    #     q = pma.mha.fc_q(seeds)
    #     q_ln = pma.mha.ln_q(q)
    #     print("Query pairwise dist mean:", torch.cdist(q_ln, q_ln).mean().item())

    #     sample = next(iter(DataLoader(testset, batch_size=PARAMS['batch_size'])))
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

