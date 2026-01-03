import time
import torch
from torch_geometric.loader import DataLoader

from molecular_data import GraphDataset, ATOM_DIM, BOND_DIM
from trainer import Trainer
from model import MAG
from explainer import explain
import utils
import statistics

import datasets
from parameters import main_params as PARAMS


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save(model, path=None):
    if not path:
        path = f"model_{time.strftime('%Y%m%d_%H%M')}.pt"
    ckpt = {
        'state_dict': model.state_dict(),
        'model_stats': getattr(model, 'stats', None),
    }
    torch.save(ckpt, path)
    print(f"\nModel checkpoint saved to: {path}")

def load(model_path, device):
    print(f"\nLoading model {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    model = MAG(ATOM_DIM, BOND_DIM).to(device)
    model.stats = ckpt.get('model_stats', None)
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
        trainer.train(model, train_loader)
        trainer.calc_stats(model, train_loader)  # Needed for Explainer

        ## Statistics on Training set
        # loader = DataLoader(trainingset, batch_size=PARAMS['batch_size'])
        # trainer.eval(model, loader, flag="Train")

        ## Save model
        # save(model, "MODELS/muta_RAND30.pt")

    else:  # Load saved model
        model = load(f"MODELS/{model_name}", device)
        # assert model.task == trainer.task

    ## Test
    testset = GraphDataset(dataset_info, split=dataset_info['test_split'])
    test_loader = DataLoader(testset, batch_size=PARAMS['batch_size'])
    trainer.eval(model, test_loader, flag="Test")

    ## Explain
    explain(model, testset)

    return model


if __name__ == "__main__":

    ## CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{time.strftime("%Y-%m-%d %H:%M:%S")}")
    print(f"DEVICE: {device}\n")

    ## Reproducibility
    if device.type == 'cuda':
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

    model_name = None
    # model_name = 'logp_benchmark.pt'
    # model_name = 'muta_benchmark.pt'
    main(device, datasets.logp_split, model_name, cv=False)






    # m1 = main(datasets.muta, 'muta_benchmark.pt')
    # pred1 = torch.cat(m1.statistics.stats[-1]['predictions'])
    # l1 = torch.cat(m1.statistics.stats[-1]['logits'])

    # m2 = main(datasets.muta, 'muta_RAND30.pt')
    # pred2 = torch.cat(m2.statistics.stats[-1]['predictions'])
    # l2 = torch.cat(m2.statistics.stats[-1]['logits'])

    # agree = sum(p1 == p2 for p1, p2 in zip(pred1, pred2))
    # logits_close = torch.allclose(l1, l2, rtol=1e-5, atol=1e-8)
    # logit_diff = (l1 - l2).abs()
    # print(f"\nModels agreement: {agree} / {len(pred1)}")
    # print(f"Logits close: {logits_close}")
    # print(f"Logits difference: min={logit_diff.min().item():.6f}, max={logit_diff.max().item():.6f}, mean={logit_diff.mean().item():.6f}, std={logit_diff.std().item():.6f}   \n")
    # c=1



    


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

