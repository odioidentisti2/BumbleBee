import torch
from torch_geometric.data import Batch
from torch_geometric.utils import degree
# import shap
# import numpy as np
import utils
from graphic import *

from molecular_data import ATOM_DIM


class Explainer:

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.intensity = 1
        # Attention
        self.att_shift = -1  # shift so that average attention is at 0
        self.att_factor = 1 / (self.model.att_factor_top + self.att_shift)  # scale so that top attention is at 1
        # Integrated Gradients
        self.ig_factor =  1 / self.model.training_predictions.std().item()  # using PREDICTION std (not actual target)
        self.baseline_pred = None  # DEBUG
        self.predictions = []  # DEBUG
        #SUMMARY
        utils.print_header("CALIBRATION")
        print(f"Prediction distribution mean/std: {self.model.training_predictions.mean():.2f} / {self.model.training_predictions.std():.2f}")
        print(f"Prediction range: {self.model.training_predictions.min():.2f} to {self.model.training_predictions.max():.2f}")
        print(f"IG top: {self.model.training_predictions.std():.2f}")
        print(f"IG factor: {self.ig_factor:.2f}")
        print(f"ATT top: {self.model.att_factor_top:.2f}")
        print(f"ATT factor: {self.att_factor:.2f}")

    def header(self, graph, weights, attention=False, count=None):
        if attention:
            utils.print_header("ATTENTION-BASED EXPLANATION")
            print(f"{graph.y.item():.2f}", graph.smiles)
            print_weights(weights, average=True, title="ATTENTION WEIGHTS:")
        else:
            utils.print_header("INTEGRATED GRADIENTS EXPLANATION")
            print(f"{graph.y.item():.2f}", graph.smiles)

            # if weights.sum().item() != sum_verification[0]:
            #     print("WARNING: Attribution sum does not match verification value!")
            #     print(f"Attribution sum: {weights.sum().item()}, Verification value: {sum_verification[0]}")

            # Verify the sum property (CRITICAL for IG correctness)
            attribution_sum = weights.sum().item()
            baseline_pred = self.baseline_pred
            final_pred = self.predictions[count]
            print(f"\nBaseline prediction: {baseline_pred:.2f}")
            print(f"Attribution sum: {attribution_sum:.2f}")
            print(f"Baseline + Attribution sum: {baseline_pred + attribution_sum:.2f}")    
            print(f"PREDICTION: {final_pred:.2f}")

            print_weights(weights, title="EDGE IMPORTANCE (averaged components):")

    def explain(self, loader):
        self.model.eval()        
        att_list = []
        ig_list = []
        c = 0
        for batch in loader:
            batch = batch.to(self.device)
            att_list.extend(self.attention(batch.clone()))  # why clone()? for random seed consistency?
            ig_list.extend(self.integrated_gradients(batch.clone()))
            for graph in batch.to_data_list():
                repeat = True
                while repeat:
                    self.header(graph, att_list[c], attention=True)
                    depict(graph, att_list[c], factor=self.att_factor * self.intensity, shift=self.att_shift, attention=True)
                    self.header(graph, ig_list[c], count=c)
                    depict(graph, ig_list[c], factor=self.ig_factor * self.intensity)
                    # user_input = ''
                    user_input = input("Press Enter to continue, '-' to halve intensity, '+' to double intensity: ")
                    plus_count = user_input.count('+')
                    minus_count = user_input.count('-')
                    if plus_count + minus_count > 0:
                        self.intensity *= (2 ** plus_count) / (2 ** minus_count)
                    else:
                        repeat = False  # Move to next molecule
                c += 1
            # return att_list, ig_list
        return att_list, ig_list
    
    def attention(self, batch):
        with torch.no_grad():
            weights = self.model(batch, return_attention=True)  # [batch_size, seq_len]
        weight_list = [weights[i, :graph.edge_index.size(1)] for i, graph in enumerate(batch.to_data_list())]  # Remove padding
        return weight_list    

    def integrated_gradients(self, batch, steps=100):
        edge_feat = self.model.get_features(batch)   
        baseline = torch.zeros_like(edge_feat)
        integrated_grads = torch.zeros_like(edge_feat)

        for alpha in torch.linspace(0, 1, steps):  # Interpolate between baseline and input
            alpha = alpha.to(self.device)        
            interp_feat = baseline + alpha * (edge_feat - baseline)
            interp_feat.requires_grad_(True)
            prediction = self.model.batch_forward(interp_feat, batch.edge_index, batch.batch)
            integrated_grads += torch.autograd.grad(
                                outputs=prediction.sum(),
                                inputs=interp_feat,
                                create_graph=False
                            )[0]
            if alpha == 0:
                baseline_pred = prediction.detach()
            elif alpha == 1:
                final_pred = prediction.detach()
        # Average gradients and scale by input difference
        integrated_grads /= steps
        attributions = (edge_feat - baseline) * integrated_grads

        # DEBUG: Sanity checks
        assert baseline_pred.std() < 1e-6, "Baseline predictions are not constant!"
        if self.baseline_pred is not None:
            assert torch.abs(baseline_pred.mean() - self.baseline_pred) < 1e-6, "Baseline predictions differ from previous graphs!"
        else:
            self.baseline_pred = baseline_pred.mean().item()
        self.predictions.extend(final_pred.tolist())  # DEBUG

        # Process each graph in the batch
        feat_importance_list = []
        atom_bond_importance_list = []
        edge_importance_list = []

        # Split attributions by graph
        edge_count = 0
        for graph in batch.to_data_list():
            # Slice per-graph attributions
            num_edges = graph.edge_index.size(1)
            graph_attributions = attributions[edge_count:edge_count + num_edges]
            edge_count += num_edges
            
            # Aggregate for this graph
            atom_feat_importance, bond_feat_importance = self.aggregate_per_feature(
                graph_attributions.detach(), graph
            )
            feat_importance_list.append((atom_feat_importance, bond_feat_importance))

            atom_importance, bond_importance = self.aggregate_per_atom_bond(
                atom_feat_importance, bond_feat_importance
            )
            atom_bond_importance_list.append((atom_importance, bond_importance))

            edge_importance = self.aggregate_per_edge(atom_importance, bond_importance, graph)
            edge_importance_list.append(edge_importance)
        
        return edge_importance_list
 

    def aggregate_per_feature(self, attributions, graph):
        # print("\n\nFEATURE IMPORTANCE:")
        
        # Split attributions by type
        src_attr = attributions[:, :ATOM_DIM]              # [num_edges, num_atom_features]
        dst_attr = attributions[:, ATOM_DIM:2*ATOM_DIM]    # [num_edges, num_atom_features]
        edge_attr = attributions[:, 2*ATOM_DIM:]           # [num_edges, num_bond_features]
        
        num_atoms = graph.x.shape[0]
        num_bonds = graph.mol.GetNumBonds()
        num_atom_features = ATOM_DIM
        num_bond_features = edge_attr.shape[1]
        
        # Initialize feature-level importance tensors
        atom_feat_importance = torch.zeros(num_atoms, num_atom_features)
        bond_feat_importance = torch.zeros(num_bonds, num_bond_features)
        
        src_idx = graph.edge_index[0].cpu()
        dst_idx = graph.edge_index[1].cpu()
        
        # Accumulate feature vectors for each unique atom/bond
        for i in range(attributions.shape[0]):
            src = int(src_idx[i])
            dst = int(dst_idx[i])
            
            # Accumulate atom feature attributions
            atom_feat_importance[src] += src_attr[i]
            atom_feat_importance[dst] += dst_attr[i]
            
            # Accumulate bond feature attributions
            bond = graph.mol.GetBondBetweenAtoms(src, dst)
            bond_idx = bond.GetIdx()
            bond_feat_importance[bond_idx] += edge_attr[i]
        
        # print(f"Atom feature importance shape: {atom_feat_importance.shape}")
        # print(f"Bond feature importance shape: {bond_feat_importance.shape}")
        # print(f"Example atom features (atom 0): {atom_feat_importance[0]}")
        # print(f"Example bond features (bond 0): {bond_feat_importance[0]}")
        
        return atom_feat_importance, bond_feat_importance
    
    def aggregate_per_atom_bond(self, atom_feat_importance, bond_feat_importance):
        # print("\n\nATOM/BOND IMPORTANCE:")
        
        # Sum across feature dimension
        atom_importance = atom_feat_importance.sum(dim=1)  # [num_atoms]
        bond_importance = bond_feat_importance.sum(dim=1)  # [num_bonds]
        
        # print_weights(atom_importance, 
        # title="BOND IMPORTANCE:")
        
        return atom_importance, bond_importance
  
    def aggregate_per_edge(self, atom_importance, bond_importance, graph):
        # print("\n\nEDGE IMPORTANCE:")
        
        src_idx = graph.edge_index[0].cpu()
        dst_idx = graph.edge_index[1].cpu()
        num_edges = src_idx.shape[0]
        
        # Compute atom appearance counts from degree
        # Each atom appears in (degree × 2) edges (bidirectional)
        atom_count = degree(src_idx, num_nodes=graph.x.shape[0]) + \
                    degree(dst_idx, num_nodes=graph.x.shape[0])
        
        # All bonds appear exactly twice (bidirectional)
        bond_count = 2
        
        # Build edge importance with averaged components (single pass)
        edge_importance = torch.zeros(num_edges)
        
        for i in range(num_edges):
            src = int(src_idx[i])
            dst = int(dst_idx[i])
            bond = graph.mol.GetBondBetweenAtoms(src, dst)
            bond_idx = bond.GetIdx()
            
            # Each component divided by its count
            edge_importance[i] = (
                atom_importance[src] / atom_count[src] + 
                atom_importance[dst] / atom_count[dst] + 
                bond_importance[bond_idx] / bond_count
            )
        
        
        return edge_importance
    


    
    # def _shap(self, graph, background_size=1):  # Changed to 1
    #     """
    #     SHAP explanation using KernelExplainer (model-agnostic).
    #     Uses same aggregation logic as IG for consistency.
    #     """
    #     import numpy as np
    #     import shap
        
    #     utils.print_header("SHAP EXPLANATION")
    #     graph = graph.cpu()
    #     batched_graph = Batch.from_data_list([graph])
        
    #     # Get edge features
    #     edge_feat = self.model.get_features(batched_graph)
    #     num_edges = edge_feat.shape[0]
        
    #     # Use zero baseline (like your IG) - single sample
    #     background = torch.zeros_like(edge_feat).unsqueeze(0).numpy()  # [1, num_edges, num_features]
    #     background = background.reshape(1, -1)  # [1, num_edges*num_features]
        
    #     # Wrap model for SHAP
    #     def model_fn(x):
    #         """Takes numpy array, returns numpy array"""
    #         x_tensor = torch.tensor(x, dtype=torch.float32, device=edge_feat.device)
            
    #         # Reshape from flat to [num_coalitions, num_edges, num_features]
    #         num_coalitions = x_tensor.shape[0]
    #         x_tensor = x_tensor.reshape(num_coalitions, num_edges, -1)
            
    #         predictions = []
    #         for i in range(num_coalitions):
    #             coalition_feat = x_tensor[i]  # [num_edges, num_features]
                
    #             with torch.no_grad():
    #                 output = self.model.single_forward(
    #                     coalition_feat,
    #                     batched_graph.edge_index, 
    #                     batched_graph.batch
    #                 )
    #             predictions.append(output.cpu().item())
            
    #         return np.array(predictions)
    
    #     # Create SHAP explainer
    #     explainer = shap.KernelExplainer(
    #         model_fn,
    #         background
    #     )
        
    #     # Flatten edge_feat for SHAP
    #     edge_feat_flat = edge_feat.cpu().numpy().reshape(1, -1)  # [1, num_edges*num_features]
        
    #     # Compute SHAP values
    #     shap_values = explainer.shap_values(edge_feat_flat, nsamples=1000)
        
    #     # Reshape back to [num_edges, num_features]
    #     attributions = torch.tensor(shap_values, dtype=torch.float32).reshape(num_edges, -1).cpu().detach()
        
    #     if self.count == 1:
    #         baseline_pred = self.model.single_forward(
    #             torch.zeros_like(edge_feat), 
    #             batched_graph.edge_index, 
    #             batched_graph.batch
    #         ).item()
    #         final_pred = self.model.single_forward(
    #             edge_feat, 
    #             batched_graph.edge_index, 
    #             batched_graph.batch
    #         ).item()
    #         _summary(graph, attributions, baseline_pred, final_pred)
        
    #     factor = self.intensity
    #     if not hasattr(graph, 'label'):  # regression
    #         factor *= 1 / self.model.training_predictions.std().item()
        
    #     graph.prediction = final_pred
        
    #     # First: atom/bond visualization (raw sums)
    #     atom_importance, bond_importance = self._atom_bond_importance(attributions, graph)
    #     depict_feat(graph, atom_importance, bond_importance, factor=factor)
        
    #     # Second: edge visualization (with averaging to handle duplication)
    #     edge_importance = self.aggregate_per_edge(atom_importance, bond_importance, graph)
    #     depict(graph, edge_importance, factor=factor)
        
    #     return attributions
