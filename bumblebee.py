import torch
from torch_geometric.loader import DataLoader

from molecular_data import GraphDataset, ATOM_DIM, BOND_DIM
from trainer import Trainer
from model import MAG
from explainer import Explainer
import utils
import statistics

import datasets
from parameters import print_parameters, train_params as PARAMS


## CPU or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_batch_size = {'cpu': 8, 'cuda': 64}[device.type]  # Optimal size for speed/memory tradeoff


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
    print(f"\nCross-Validation on: {dataset_info['path']}")
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

        train_loader = DataLoader(train_subset, batch_size=PARAMS['train_batch_size'], shuffle=True, drop_last=True)
        test_loader = DataLoader(test_subset, batch_size=test_batch_size, generator=g)
        
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
        print(f"\nTraining set: {dataset_info['path']}")
        trainingset = GraphDataset(dataset_info, split=dataset_info['train_split'])
        train_loader = DataLoader(trainingset, batch_size=PARAMS['train_batch_size'], shuffle=True, drop_last=True)


        val_loader = None
        ## Load validation set
        print(f"\nValidation set: {dataset_info['path']}")
        validation_set = GraphDataset(dataset_info, split=dataset_info['test_split'])
        val_loader = DataLoader(validation_set, batch_size=test_batch_size)

        rng_before = torch.get_rng_state()
        for batch in val_loader:
            rng_after = torch.get_rng_state()
            print("RNG STATE AFTER for batch EQUAL?", torch.equal(rng_before, rng_after))

        ## Train model
        model = MAG(ATOM_DIM, BOND_DIM)
        trainer.set_baseline_target(trainingset.targets)  # For injection baseline
        trainer.train(model, train_loader, val_loader)
        trainer.calibration_stats(model, train_loader)  # Needed for Explainer

        ## Statistics on Training setset_baseline_target
        # loader = DataLoader(trainingset, batch_size=PARAMS['batch_size'])
        # trainer.eval(model, loader, flag="Train")

        ## Save model
        # save(model, "MODELS/muta.pt")

    else:  # Load saved model
        model = load(f"MODELS/{model_name}", device)
        # there should be task inside model......    

    model.task = dataset_info['task']

    ## Test
    print(f"\nTest set: {dataset_info['path']}")
    testset = GraphDataset(dataset_info, split=dataset_info['test_split'])
    print("\nTEST BATCH SIZE = 2")
    test_loader = DataLoader(testset, batch_size=2)
    trainer.eval(model, test_loader, flag="Test")


    ## Explain
    # utils.print_header("CALIBRATION")
    # print(f"Prediction distribution mean/std: {model.training_predictions.mean():.2f} / {model.training_predictions.std():.2f}")
    # print(f"Prediction range: {model.training_predictions.min():.2f} to {model.training_predictions.max():.2f}")
    # explainer = Explainer(att_top=model.att_factor_top, ig_top=model.training_predictions.std().item())
    # return explainer.explain(model, test_loader)


if __name__ == "__main__":
    import time
    print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"DEVICE: {device}")
    model_name = None

    _datasets = []
    _datasets.append(datasets.logp_split)
    # _datasets.append(datasets.muta)

    for dataset_info in _datasets:
        # model_name = 'logp.pt'
        # model_name = 'muta.pt'

        ## Reproducibility
        if dataset_info['task'] == 'binary_classification':  # MSE criterion (regression) looks deterministic (but it needs more tests)
            if device.type == 'cuda':
                import os
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.use_deterministic_algorithms(True)
            print("(DETERMINISTIC algorithms)")

        # print("TRAINER.EVAL HAS RETURN_ATTENTION = TRUE!!!!!")

        start_time = time.time()
        # crossvalidation(dataset_info, device)   
        main_loop(dataset_info, device, model_name)
        print(f"\nTOTAL TIME: {time.time() - start_time:.0f}s")




#  TO BE PLACED INSIDE MAIN TO DEBUG SEED DIVERSITY
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

