import time
import torch
from torch_geometric.loader import DataLoader
from molecular_data import GraphDataset, ATOM_DIM, BOND_DIM
from model import MAGClassifier
from explainer import *


# Trainer.fit from pytorch_lightning does the same job
def train(model, loader, optimizer, criterion):
    model.train()  # set training mode
    total_loss = 0
    total = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        targets = batch.y.view(-1).to(DEVICE)
        optimizer.zero_grad()  # zero gradients
        logits = model(batch, BATCH_DEBUG)  # forward pass
        loss = criterion(logits, targets)  # calculate loss
        loss.backward()  # backward pass
        optimizer.step()  # update weights
        # statistics
        total_loss += loss.item() * batch.num_graphs
        total += batch.num_graphs
    return total_loss / total

def test(model, loader, criterion):
    model.eval()  # set evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        targets = batch.y.view(-1).to(DEVICE)
        with torch.no_grad():
            logits = model(batch, BATCH_DEBUG)  # forward pass
            loss = criterion(logits, targets)
            total_loss += loss.item() * batch.num_graphs
            total += batch.num_graphs
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == targets).sum().item()
    return total_loss / total, correct / total

def explain(model, single_loader):
    model.eval()
    current_intensity = 1
    for batched_molecule in single_loader:
        batched_molecule = batched_molecule.to(DEVICE)
        repeat = True
        while repeat:
            explain_with_attention(model, batched_molecule, intensity=current_intensity)
            explain_with_gradients(model, batched_molecule, steps=100, intensity=current_intensity)
            explain_with_mlp_integrated_gradients(model, batched_molecule, intensity=current_intensity)
            user_input = input("Press Enter to continue, '-' to halve intensity, '+' to double intensity: ")
            plus_count = user_input.count('+')
            minus_count = user_input.count('-')
            if plus_count + minus_count > 0:
                current_intensity = current_intensity * (2 ** plus_count) / (2 ** minus_count)
            else:
                repeat = False  # Move to next molecule

def training_loop(loader, criterion):
    print("\nTraining...")
    model = MAGClassifier(ATOM_DIM, BOND_DIM, glob['LAYER_TYPES']).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=glob['LR'])
    start_time = time.time()
    for epoch in range(1, glob['NUM_EPOCHS'] + 1):
        loss = train(model, loader, optimizer, criterion)
        print(f"Epoch {epoch}: Loss {loss:.3f}  Time {time.time() - start_time:.0f}s")
    return model

def evaluate(model, loader, criterion, flag):
    print("\nEvaluating...")
    loss, acc = test(model, loader, criterion)
    print(f"{flag}: Loss {loss:.3f}  Acc {acc:.3f}")
    return loss, acc

def save(model):
    model_path = f"model_{time.strftime('%Y%m%d_%H%M')}_{glob['BATCH_SIZE']}_{glob['LR']}_{glob['NUM_EPOCHS']}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

def load(model_path):
    print(f"\nLoading model {model_path}")
    model = MAGClassifier(ATOM_DIM, BOND_DIM, glob['LAYER_TYPES']).to(DEVICE)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def crossvalidation(criterion):
    num_folds = 5
    print(f"\nCross-Validation on: ", DATASET_PATH)
    dataset = GraphDataset(DATASET_PATH)
    dataset_size = len(dataset)
    indices = torch.randperm(dataset_size).tolist()
    fold_size = dataset_size // num_folds
    fold_results = {
        'test_losses': [],
        'test_accs': [],
    }

    start_time = time.time()   
    for fold in range(num_folds):        
        # Create indices
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < num_folds - 1 else dataset_size
        test_indices = indices[test_start:test_end]
        train_indices = indices[:test_start] + indices[test_end:]        
        # Create subsets
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        test_subset = torch.utils.data.Subset(dataset, test_indices)
        train_loader = DataLoader(train_subset, batch_size=glob['BATCH_SIZE'], shuffle=True, drop_last=True)
        test_loader = DataLoader(test_subset, batch_size=glob['BATCH_SIZE'])

        print(f"\n{'='*50}\nFold {fold+1}/{num_folds}\n{'='*50}")
        print(f"Train size: {len(train_indices)}, Test size: {len(test_indices)}")
        model = training_loop(train_loader, criterion)
        loss, acc = evaluate(model, test_loader, criterion, flag=f"Fold {fold+1}")        
        fold_results['test_losses'].append(loss)
        fold_results['test_accs'].append(acc)    

    # Print composite statistics
    print(f"\n{'='*50}\nCROSS-VALIDATION RESULTS\n{'='*50}")    
    mean_loss = sum(fold_results['test_losses']) / num_folds
    std_loss = (sum((x - mean_loss)**2 for x in fold_results['test_losses']) / num_folds) ** 0.5    
    mean_acc = sum(fold_results['test_accs']) / num_folds
    std_acc = (sum((x - mean_acc)**2 for x in fold_results['test_accs']) / num_folds) ** 0.5    
    print(f"Test Loss:  {mean_loss:.3f} ± {std_loss:.3f}")
    print(f"Test Acc:   {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"\nIndividual fold accuracies: {[f'{acc:.3f}' for acc in fold_results['test_accs']]}")
    print(F"\nTOTAL TIME: {time.time() - start_time:.0f}s")
    print(f"{'='*50}\n")


def main():
    criterion = torch.nn.BCEWithLogitsLoss()

    if CROSS_VALIDATION:
        crossvalidation(criterion)
        return
        
    ## Train
    print(f"\nTraining set: {DATASET_PATH} ('Training')")
    trainingset = GraphDataset(DATASET_PATH, split='Training')
    train_loader = DataLoader(trainingset, batch_size=glob['BATCH_SIZE'], shuffle=True, drop_last=True)
    model = training_loop(train_loader, criterion)
    ## Statistics on Training set
    # loader = DataLoader(trainingset, batch_size=glob['BATCH_SIZE'])
    # statistics(model, loader, criterion, flag="Train")

    ## Save model
    # save(model)

    ## Load saved model
    # model = load(MODEL_PATH)

    ## Test
    print(f"\nTest set: {DATASET_PATH} ('Test')")
    testset = GraphDataset(DATASET_PATH, split='Test')
    test_loader = DataLoader(testset, batch_size=glob['BATCH_SIZE'])
    evaluate(model, test_loader, criterion, flag="Test")

    # # Explain
    # single_loader = DataLoader(testset, batch_size=1)
    # explain(model, single_loader)

if __name__ == "__main__":
    ## Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    ## DEBUG
    BATCH_DEBUG = None
    # BATCH_DEBUG =  True  # Debug: use batch Attention even on CPU
    CROSS_VALIDATION = False
    ## GLOBALS
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATASET_PATH = 'DATASETS/MUTA_SARPY_4204.csv'
    MODEL_PATH = 'model_20250822_210138.pt'
    glob = {
        "BATCH_SIZE": 32,  # I should try reducing waste since drop_last=True
        "LR": 1e-4,
        "NUM_EPOCHS": 15,
        "LAYER_TYPES": 'MSMSMP',  # 'MMSP'
    }
    ## Print time and model stamps
    print()
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"DEVICE: {DEVICE}")
    import pprint
    pprint.pprint(glob)
    main() 

    ## ESA repo
    # weight_decay = 1e-10 nel README, 1e-3 come default (AdamW)
    # HIDDEN_DIM = 256  # = MLP_hidden = graph_dim
    # BATCH_SIZE = 128
    # NUM_HEADS = 16
    # LAYER_TYPES = ['MSMSMP']
    # DROPOUT = 0
