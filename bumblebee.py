import torch

from molecular_data import Dataset, InjectedDataset, load_from_csv
from trainer import Trainer
from model import MAG
from explainer import Explainer
import reproducibility

# CV
from preprocessing import cv_subsets
import statistics

from pprint import pformat
import datasets
from parameters import parameter_summary

from logging_config import getLogger
logger = getLogger(__name__)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save(path, model):
    ckpt = {
        'task': model.task,
        'state_dict': model.state_dict(),
        'calibration': getattr(model, 'calibration_data', None),
    }
    torch.save(ckpt, path)
    logger.info(f"\nModel saved to: {path}")


def load(model_path):
    device = get_device()
    ckpt = torch.load(model_path, map_location=device)
    model = MAG(device)
    model.load_state_dict(ckpt['state_dict'])
    model.task = ckpt['task']
    model.calibration_data = ckpt['calibration']
    logger.info(f"\nLoaded model from {model_path} with task: {model.task}")
    return model


# def crossvalidation(dataset_info, device, folds=5):
#     logger.info(f"\nCross-Validation on: {dataset_info['path']}")
#     graphs = load_from_csv(dataset_info)
#     cv_tracker = statistics.CVTracker()
    
#     for fold, (train_indices, test_indices) in enumerate(cv_subsets(len(graphs), folds), start=1):
#         logger.debug(parameter_summary())
#         reproducibility.set_torch_seed()
#         logger.info(f"Fold {fold}/{folds}", highlight=True)

#         trainingset  = InjectedDataset([graphs[i] for i in train_indices])
#         testset      = Dataset([graphs[i] for i in test_indices])
#         logger.info(f"Train size: {len(trainingset)}, Test size: {len(testset)}")

#         trainer = Trainer(dataset_info['task'], device)
#         trainer.tracker.accumulate = True  # keep all runs
#         trainer.train(MAG(device), trainingset, val_set=testset)
#         cv_tracker.add_fold(trainer.tracker.metrics())    
    
#     logger.info(cv_tracker.summary())  # Print summary


def load_dataset(dataset_info, train=False, flag='Dataset'):
    logger.info(f"\n{flag}: {dataset_info['path']}")
    logger.info("\n Loading...")
    graphs = load_from_csv(dataset_info)
    logger.info(f" Loaded {len(graphs)} structures")
    if train:
        dataset = InjectedDataset(graphs)
    else:
        dataset = Dataset(graphs)
    dataset.task = dataset_info['task']
    return dataset

    
def train(trainingset, validation_set=None):
    ### CPU or GPU
    device = get_device()
    
    ### Reproducibility
    reproducibility.set_torch_seed()
    if trainingset.task == 'binary_classification':
        # (MSELoss => regression is deterministic enough ?)
        reproducibility.use_deterministic_algorithms(device)
        logger.debug("\n(DETERMINISTIC algorithms)")

    ### DEBUG
    logger.debug(f"DEVICE: {device}")
    logger.debug(parameter_summary())

    ### Train
    logger.info("\nTraining...")
    trainer = Trainer(trainingset.task, device)
    model = MAG(device)
    trainer.train(model, trainingset, val_set=validation_set)

    ### Calibration (for Explainer)
    logger.info("\nCalibrating...")
    trainer.calibrate(model, trainingset)
    return model


def test(testset, model):
    logger.info("\nTesting...")
    trainer = Trainer(testset.task, model.device)
    trainer.eval(model, testset, flag="Test")


def main_loop(trainingset_info, testset_info, model_name=None):
    
    if not model_name:  # Train model

        ### Training set
        trainingset = load_dataset(trainingset_info, train=True, flag='Training set')

        ### Load validation set (optional)
        validation_set = None
        validation_set = load_dataset(testset_info, flag='Validation set')

        ### Train model
        model = train(trainingset, validation_set)

        ## Statistics on Training set
        # logger.info("\nEvaluating on training set...")
        # trainer.eval(model, trainingset, flag="Train")

        ### Save model
        # model_name = "L4_LOGP_early_stop.pt"
        # save(f"MODELS/{model_name}", model)
        # model = load(f"MODELS/{model_name}", model.device)

    else:  # Load saved model
        model = load(f"MODELS/{model_name}")

    ### Test 
    testset = load_dataset(testset_info, flag='Test set')
    test(testset, model)

    ### Explain
    if hasattr(model, "calibration_data") and model.calibration_data is not None:
        logger.debug("CALIBRATION", highlight=True)
        logger.debug(pformat(model.calibration_data))
        explainer = Explainer(model.calibration_data)
        return explainer.explain(model, testset)


if __name__ == "__main__":
    import time
    logger.info(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')}")

    trainingset_info, testset_info = datasets.muta_train, datasets.muta_test
    # trainingset_info, testset_info = datasets.logp_train, datasets.logp_test
    # trainingset_info = datasets.logp_split

    model_name = None
    # model_name = 'L4_LOGP_10e.pt'
    # model_name = 'L4_MUTA_new_inj.pt'

    # logger.debug("CV, MUTA, early stop")

    VERBOSE = True
    start_time = time.time()
    # crossvalidation(trainingset_info, device)
    main_loop(trainingset_info, testset_info, model_name)
    logger.info(f"\nTOTAL TIME: {time.time() - start_time:.0f}s")










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

