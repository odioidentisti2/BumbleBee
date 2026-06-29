import torch

from molecular_data import Dataset, InjectedDataset,  ATOM_DIM, BOND_DIM
from trainer import Trainer
from model import MAG
from explainer import Explainer
from reproducibility import use_deterministic_algorithms, set_torch_seed
import utils
import statistics

from pprint import pprint
import datasets
from parameters import print_parameters


def save(path, model, calibration=None):
    # if not path:
    #     path = f"model_{time.strftime('%Y%m%d_%H%M')}.pt"
    ckpt = {
        'task': model.task,
        'state_dict': model.state_dict(),
        'calibration': calibration,
    }
    torch.save(ckpt, path)
    print(f"\nModel saved to: {path}")

def load(model_path, device):
    ckpt = torch.load(model_path, map_location=device)
    model = MAG(ATOM_DIM, BOND_DIM).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.task = ckpt['task']
    calibration = ckpt.get('calibration')
    print(f"\nLoaded model from {model_path} with task: {model.task}")
    return model, calibration


def crossvalidation(dataset_info, device, folds=5):
    from preprocessing import cv_subsets
    print(f"\nCross-Validation on: {dataset_info['path']}")
    dataset = Dataset(dataset_info)
    cv_tracker = statistics.CVTracker()
    
    for fold, (train_subset, test_subset) in enumerate(cv_subsets(dataset, folds), start=1):
        print_parameters()        
        set_torch_seed()  # Reproducibility

        utils.print_header(f"Fold {fold}/{folds}")
        print(f"Train size: {len(train_subset)}, Test size: {len(test_subset)}")

        # ADD GENERATOR!!!!!!!!!!!!!!!!!!!!!!!!
        # train_loader = DataLoader(train_subset, batch_size=PARAMS['train_batch_size'], generator=g(), \
        train_loader = DataLoader(train_subset, batch_size=PARAMS['train_batch_size'], \
                                  shuffle=True, drop_last=True)
        test_loader = DataLoader(test_subset, batch_size=OPTIMAL_BATCH_SIZE[device.type], generator=g())
        
        model = MAG(ATOM_DIM, BOND_DIM)
        trainer = Trainer(dataset_info['task'], device)
        trainer.train(model, train_loader, val_loader=test_loader)   
        cv_tracker.add_fold(trainer.statistics.metrics())    
    
    cv_tracker.summary()  # Print summary


def main_loop(dataset_info, device, model_name=None):
    print_parameters()
    set_torch_seed()  # Reproducibility
    
    trainer = Trainer(dataset_info['task'], device)

    if not model_name:  # Train model
        ### Load training set
        print(f"\nTraining set: {dataset_info['path']}")
        trainingset = InjectedDataset(dataset_info, split=dataset_info['train_split'])
        validation_set = None

        ### Load validation set
        # print(f"\nValidation set: {dataset_info['path']}")
        # validation_set = GraphDataset(dataset_info, split=dataset_info['test_split'])

        ### Train model
        model = MAG(ATOM_DIM, BOND_DIM)
        print("\nTraining...")
        calibration_data = trainer.train(model, trainingset, validation_set)

        ## Statistics on Training set
        # print("\nEvaluating on training set...")
        # trainer.eval(model, trainingset, flag="Train")  # WHY IS THIS SO DIFFERENT FROM TRAINING LOSS?

        ### Save model
        model_name = "MUTA_new_inj.pt"
        save(f"MODELS/{model_name}", model, calibration_data)

    else:  # Load saved model
        model, calibration_data = load(f"MODELS/{model_name}", device)

    ### Test
    print(f"\nTest set: {dataset_info['path']}")
    testset = Dataset(dataset_info, split=dataset_info['test_split'])
    print("\nTesting...")
    trainer.eval(model, testset, flag="Test")

    ### Explain
    utils.print_header("CALIBRATION")
    explainer = Explainer(calibration_data)
    pprint(explainer.calibration)
    return explainer.explain(model, testset)


if __name__ == "__main__":
    import time
    print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')}")

    ### CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEVICE: {device}")

    model_name = None
    _datasets = []
    # _datasets.append(datasets.logp_split)
    _datasets.append(datasets.muta)

    for dataset_info in _datasets:
        # model_name = 'logp_calibration.pt'
        # model_name = 'muta_calibration.pt'

        ### Reproducibility  (MSELoss => regression is deterministic enough ?)
        if dataset_info['task'] == 'binary_classification':
            use_deterministic_algorithms(device)

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

