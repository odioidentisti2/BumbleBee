import torch
from torch_geometric.loader import DataLoader

from molecular_data import GraphDataset, ATOM_DIM, BOND_DIM
from trainer import Trainer
from model import MAG
from explainer import Explainer
import utils
import statistics

import datasets
from parameters import print_parameters, main_params as PARAMS


## CPU or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save(model, path):
    # if not path:
    #     path = f"model_{time.strftime('%Y%m%d_%H%M')}.pt"
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

def crossvalidation(dataset_info, device, folds=5):
    from preprocessing import cv_subsets
    print(f"\nCross-Validation on: ", dataset_info['path'])
    dataset = GraphDataset(dataset_info)
    cv_tracker = statistics.CVTracker()
    
    for fold, (train_subset, test_subset) in enumerate(cv_subsets(dataset, folds), start=1):
        print_parameters()
        utils.print_header(f"Fold {fold}/{folds}")
        print(f"Train size: {len(train_subset)}, Test size: {len(test_subset)}")
        # Reproducibility
        set_random_seed(PARAMS['random_seed'])
        g = torch.Generator()
        g.manual_seed(PARAMS['random_seed'])

        train_loader = DataLoader(train_subset, batch_size=PARAMS['batch_size'], shuffle=True, drop_last=True)
        test_loader = DataLoader(test_subset, batch_size=PARAMS['batch_size'], generator=g)
        
        model = MAG(ATOM_DIM, BOND_DIM)
        trainer = Trainer(dataset_info['task'], device)
        trainer.set_baseline_target(dataset.targets)  # WARNING: using full dataset targets!!!!!!
        trainer.train(model, train_loader, val_loader=test_loader)   
        cv_tracker.add_fold(trainer.statistics.metrics())    
    
    cv_tracker.summary()  # Print summary    

def main_loop(dataset_info, device, model_name=None):
    print_parameters()

    ## Reproducibility
    set_random_seed(PARAMS['random_seed'])
    
    trainer = Trainer(dataset_info['task'], device)

    if not model_name:  # Train model
        ## Load training set
        trainingset = GraphDataset(dataset_info, split=dataset_info['train_split'])
        train_loader = DataLoader(trainingset, batch_size=PARAMS['batch_size'], shuffle=True, drop_last=True)

        ## Train model
        model =  MAG(ATOM_DIM, BOND_DIM)
        trainer.set_baseline_target(trainingset.targets)  # For injection baseline
        trainer.train(model, train_loader)
        trainer.calibration_stats(model, train_loader)  # Needed for Explainer

        ## Statistics on Training set
        # loader = DataLoader(trainingset, batch_size=PARAMS['batch_size'])
        # trainer.eval(model, loader, flag="Train")

        ## Save model
        # save(model, "MODELS/logp_rand15.pt")

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
    # return ig


if __name__ == "__main__":
    import time
    print(f"\n{time.strftime("%Y-%m-%d %H:%M:%S")}")
    print(f"DEVICE: {device}")
    model_name = None

    ## Reproducibility
    # if device.type == 'cuda':
    #     import os
    #     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(True)

    ## Inputs
    dataset_info = datasets.logp_split
    model_name = 'logp_rand42.pt'
    # model_name = 'muta_benchmark.pt'

    start_time = time.time()
    # crossvalidation(dataset_info, device)   
    # print("RANDOM = 15\n")
    main_loop(dataset_info, device, model_name)
    print(f"\nTOTAL TIME: {time.time() - start_time:.0f}s")




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

