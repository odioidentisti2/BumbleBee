import torch

from molecular_data import Dataset, InjectedDataset, load_from_csv
from trainer import Trainer
from model import MAG
from explainer import Explainer
from preprocessing import cv_subsets
import reproducibility
import statistics
import utils

from pprint import pprint
import datasets
from parameters import print_parameters


def save(path, model):
    ckpt = {
        'task': model.task,
        'state_dict': model.state_dict(),
        'calibration': getattr(model, 'calibration_data', None),
    }
    torch.save(ckpt, path)
    print(f"\nModel saved to: {path}")

def load(model_path, device):
    ckpt = torch.load(model_path, map_location=device)
    model = MAG(device)
    model.load_state_dict(ckpt['state_dict'])
    model.task = ckpt['task']
    model.calibration_data = ckpt['calibration']
    print(f"\nLoaded model from {model_path} with task: {model.task}")
    return model

# def load_dataset(dataset_info, train=False):


def crossvalidation(dataset_info, device, folds=5):
    print(f"\nCross-Validation on: {utils.get_name(dataset_info['path'])}")
    graphs = load_from_csv(dataset_info)
    cv_tracker = statistics.CVTracker()
    
    for fold, (train_indices, test_indices) in enumerate(cv_subsets(len(graphs), folds), start=1):
        print_parameters()
        reproducibility.set_torch_seed()
        utils.print_header(f"Fold {fold}/{folds}")

        trainingset  = InjectedDataset([graphs[i] for i in train_indices])
        testset      = Dataset([graphs[i] for i in test_indices])
        print(f"Train size: {len(trainingset)}, Test size: {len(testset)}")

        trainer = Trainer(dataset_info['task'], device)
        trainer.tracker.accumulate = True  # keep all runs
        trainer.train(MAG(device), trainingset, val_set=testset)
        cv_tracker.add_fold(trainer.tracker.metrics())    
    
    cv_tracker.summary()  # Print summary


def main_loop(trainingset_info, testset_info, device, model_name=None):
    print_parameters()
    reproducibility.set_torch_seed()
    
    trainer = Trainer(trainingset_info['task'], device)

    if not model_name:  # Train model

        ### Training set
        print(f"\nTraining set: {utils.get_name(trainingset_info['path'])}")
        train_molecules = load_from_csv(trainingset_info)
        trainingset = InjectedDataset(train_molecules)

        ### Load validation set (optional)
        validation_set = None
        print(f"\nValidation set: {utils.get_name(testset_info['path'])}")
        val_molecules = load_from_csv(testset_info)
        validation_set = Dataset(val_molecules)

        ### Train model
        model = MAG(device)
        print("\nTraining...")
        trainer.train(model, trainingset, val_set=validation_set)

        ### Calibration (for Explainer)
        print("\nCalibrating...")
        trainer.calibrate(model, trainingset)

        ## Statistics on Training set
        # print("\nEvaluating on training set...")
        # trainer.eval(model, trainingset, flag="Train")

        ### Save model
        # model_name = "L4_LOGP_early_stop.pt"
        # save(f"MODELS/{model_name}", model)
        # model = load(f"MODELS/{model_name}", device)

    else:  # Load saved model
        model = load(f"MODELS/{model_name}", device)

    ### Test 
    print(f"\nTest set: {utils.get_name(testset_info['path'])}")
    test_molecules = load_from_csv(testset_info)
    testset = Dataset(test_molecules)
    print("\nTesting...")
    trainer.eval(model, testset, flag="Test")

    ### Explain
    if hasattr(model, "calibration_data") and model.calibration_data is not None:
        utils.print_header("CALIBRATION")
        pprint(model.calibration_data)
        explainer = Explainer(model.calibration_data)
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
    _datasets.append((datasets.muta_train, datasets.muta_test))

    for trainingset_info, testset_info in _datasets:
        # model_name = 'L4_LOGP_10e.pt'
        # model_name = 'L4_MUTA_new_inj.pt'

        ### Reproducibility  (MSELoss => regression is deterministic enough ?)
        if trainingset_info['task'] == 'binary_classification':
            reproducibility.use_deterministic_algorithms(device)

        start_time = time.time()
        # crossvalidation(dataset_info, device)   
        main_loop(trainingset_info, testset_info, device, model_name)
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

