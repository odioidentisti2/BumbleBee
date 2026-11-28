from typing import Any, Literal

import base64
import torch
import io

from molecular_data import Testset, Trainingset, load_from_csv
from trainer import Trainer
from model import MAG
from explainer_att import AttentionExplainer
from explainer_ig import IGExplainer


# CV
from preprocessing import cv_subsets
import statistics

from pprint import pformat
import datasets
from parameters import parameter_summary, MODEL_BASE_DIR
import reproducibility

from logging_config import getLogger
logger = getLogger(__name__)


# Global
DEVICE = None  # To be initalized by main or GUI
EXPLAINERS = (AttentionExplainer, IGExplainer)

def init_device() -> str:
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return str(DEVICE)

def configure_reproducibility(use_deterministic: bool = True):
    reproducibility.set_torch_seed()
    if use_deterministic:
        # (MSELoss => regression is deterministic enough ?)
        reproducibility.use_deterministic_algorithms(DEVICE)
        logger.debug("\n(DETERMINISTIC algorithms)")


def save(model: MAG, model_name: str,) -> None:
    ckpt = {
        'task': model.task,
        'state_dict': model.state_dict(),
        'calibration': getattr(model, 'calibration_data', None),
    }
    path = f"{MODEL_BASE_DIR}/{model_name}"
    torch.save(ckpt, path)
    logger.info(f"\nModel saved to: {path}")


def load(model_name: str) -> MAG:
    path = f"{MODEL_BASE_DIR}/{model_name}"
    ckpt = torch.load(path, map_location=DEVICE)
    model = MAG(DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    model.task = ckpt['task']
    model.calibration_data = ckpt['calibration']
    logger.info(f"\nLoaded model {model_name} with task: {model.task}")
    return model


# def crossvalidation(dataset_info, folds=5):
#     logger.info(f"\nCross-Validation on: {dataset_info['path']}")
#     graphs = load_from_csv(dataset_info)
#     cv_tracker = statistics.CVTracker()
    
#     for fold, (train_indices, test_indices) in enumerate(cv_subsets(len(graphs), folds), start=1):
#         logger.debug(parameter_summary())
#         reproducibility.set_torch_seed()
#         logger.info(f"Fold {fold}/{folds}", highlight=True)

#         trainingset  = Trainingset([graphs[i] for i in train_indices])
#         testset      = Testset([graphs[i] for i in test_indices])
#         logger.info(f"Train size: {len(trainingset)}, Test size: {len(testset)}")

#         trainer = Trainer(dataset_info['task'], DEVICE)
#         trainer.tracker.accumulate = True  # keep all runs
#         trainer.train(MAG(DEVICE), trainingset, val_set=testset)
#         cv_tracker.add_fold(trainer.tracker.metrics())    
    
#     logger.info(cv_tracker.summary())  # Print summary


def load_dataset(dataset_info: dict, kind: Literal["Training", "Test", "Validation"]):
    logger.info(f"{kind} set: {dataset_info['file_name']}")
    logger.info("\nLoading...")
    graphs = load_from_csv(dataset_info)
    logger.info(f" Loaded {len(graphs)} structures")
    if kind == 'Training':
        dataset = Trainingset(graphs)
    else:
        dataset = Testset(graphs)
    dataset.task = dataset_info['task']
    return dataset

    
def train(trainingset: Trainingset, validation_set: Testset | None = None) -> MAG:    
    ### Reproducibility
    configure_reproducibility(use_deterministic=(trainingset.task == 'binary_classification'))

    ### DEBUG
    logger.debug(parameter_summary())

    ### Train
    logger.info("\nTraining...")
    trainer = Trainer(trainingset.task, DEVICE)
    model = MAG(DEVICE)
    trainer.train(model, trainingset, val_set=validation_set)

    ### Calibration (for Explainer)
    logger.info("\nCalibrating...")
    trainer.calibrate(model, trainingset)

    logger.info("\nTraining completed.")

    # DEBUG
    # save(model, trainingset.task + "_model_RAND15.pt")

    return model


def test(testset: Testset, model: MAG) -> None:
    logger.info("\nTesting...")
    trainer = Trainer(testset.task, model.device)
    trainer.eval(model, testset, flag="Test")


def GUI_explain(model: MAG, testset: Testset):
    explainers = [cls(model.calibration_data) for cls in EXPLAINERS]
    streams = (e.explain(model, testset) for e in explainers)
    for explanations in zip(*streams):
        payload = {}
        for e, (graph, attributions) in zip(explainers, explanations):
            payload[e.name] = {
                "image": "data:image/png;base64," + base64.b64encode(
                    e.depict(graph, attributions)).decode(),
                "prediction": testset.logits[graph.idx],
                **e.extras(),
                "edges": [{"name": f"edge{bond_index}", "weight": float(weight)}
                            for bond_index, weight in enumerate(attributions.tolist())],
            }
        yield payload


def main_loop(trainingset_info: dict, testset_info: dict, model_name: str | None = None):
    
    if not model_name:  # Train model

        ### Training set
        trainingset = load_dataset(trainingset_info, kind='Training')

        ### Load validation set (optional)
        validation_set = None
        validation_set = load_dataset(testset_info, kind='Validation')

        ### Train model
        model = train(trainingset, validation_set)

        ## Statistics on Training set
        # logger.info("\nEvaluating on training set...")
        # trainer.eval(model, trainingset, flag="Train")

        ### Save model
        # model_name = "T4_LOGp_100e.pt"
        # save(model, model_name)
        # model = load(model_name)

    else:  # Load saved model
        model = load(model_name)

    ### Test 
    testset = load_dataset(testset_info, kind='Test')
    test(testset, model)

    ### Explain
    EXPLAIN = False
    if not EXPLAIN:
        return
    if hasattr(model, "calibration_data") and model.calibration_data is not None:
        logger.debug("CALIBRATION DATA", highlight=True)
        logger.debug(pformat(model.calibration_data))
        
        from PIL import Image

        # DEBUG: check if running in Google Colab
        IN_COLAB = False
        try:
            import google.colab
            IN_COLAB = True
            from IPython.display import display
        except ImportError:
            pass

        explainers = [cls(model.calibration_data) for cls in EXPLAINERS]
        streams = (e.explain(model, testset) for e in explainers)
        intensity = 1.0
        for explanations in zip(*streams):
            while True:
                for e, (graph, attributions) in zip(explainers, explanations):
                    e.summary(graph, attributions, testset.logits[graph.idx])
                    png = e.depict(graph, attributions, intensity)
                    image = Image.open(io.BytesIO(png)).convert("RGBA")
                    white_background_image = Image.alpha_composite(
                        Image.new("RGBA", image.size, "white"), image).convert("RGB")
                    if IN_COLAB:
                        display(white_background_image)
                    else:
                        white_background_image.show()
                user_input = input("Enter = next, '+'/'-' = intensity: ")
                plus_count = user_input.count('+')
                minus_count = user_input.count('-')
                if plus_count + minus_count == 0:
                    break
                intensity *= (2 ** plus_count) / (2 ** minus_count)


if __name__ == "__main__":
    import time
    logger.info(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')}")

    device = init_device()
    logger.info(f"DEVICE: {device}")

    trainingset_info, testset_info = datasets.logp_train, datasets.logp_test
    # trainingset_info, testset_info = datasets.logp_train, datasets.logp_test
    # trainingset_info = datasets.logp_split

    model_name = None
    # model_name = 'NEW_L4_LOGP_10e.pt'
    # model_name = 'L4_MUTA_new_inj.pt'

    # logger.debug("CV, MUTA, early stop")

    start_time = time.time()
    # crossvalidation(trainingset_info, DEVICE)
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
