import torch
from torch_geometric.utils import degree

from graphic import depict as draw_molecule, weight_summary

from logging_config import getLogger
logger = getLogger(__name__)


class IGExplainer:

    name = "Integrated Gradients"

    def __init__(self, calibration_data):
        top = calibration_data['logit_std']  # using PREDICTION std (not actual target)
        self.shift = 0
        self.factor = 1 / (top + self.shift)
        self.baseline_logits = None  # DEBUG: for logging (see depict)
        self.logits = []  # DEBUG: for logging (see depict)
        self.last_batch_logits = None  # DEBUG: checked in an assert in Explainer base class

        logger.debug(f"IG top: {top:.2f}")
        logger.debug(f"IG factor: {self.factor:.2f}")

    def attributions(self, model, batch, steps=100):
        edge_feat = model.get_features(batch)   
        baseline = torch.zeros_like(edge_feat)
        integrated_grads = torch.zeros_like(edge_feat)

        for alpha in torch.linspace(0, 1, steps):  # Interpolate between baseline and input
            alpha = alpha.to(edge_feat.device)        
            interp_feat = baseline + alpha * (edge_feat - baseline)
            interp_feat.requires_grad_(True)
            logits = model.batch_forward(interp_feat, batch.edge_index, batch.batch)
            integrated_grads += torch.autograd.grad(
                                outputs=logits.sum(),
                                inputs=interp_feat,
                                create_graph=False
                            )[0]
            if alpha == 0:
                baseline_logits = logits.detach()
            elif alpha == 1:
                self.last_batch_logits = logits.detach().cpu()  # DEBUG
        # Average gradients and scale by input difference
        integrated_grads /= steps
        attributions = (edge_feat - baseline) * integrated_grads

        # DEBUG: Sanity checks
        logger.debug(f"\nBATCH BASELINES: {baseline_logits}") 
        if baseline_logits.shape[0] > 1:
            assert baseline_logits.std() < 1e-4, "Baseline predictions are not constant!"
        if self.baseline_logits is not None:
            assert torch.abs(baseline_logits.mean() - self.baseline_logits) < 1e-4, "Baseline predictions differ from previous graphs!"
        else:
            self.baseline_logits = baseline_logits.mean().item()
        self.logits.extend(self.last_batch_logits.tolist())

        # Process each graph in the batch
        # feat_importance_list = []
        # atom_bond_importance_list = []
        edge_importance_list = []
        edge_count = 0
        for graph in batch.to_data_list():
            # Slice per-graph attributions
            num_edges = graph.edge_index.size(1)
            graph_attributions = attributions[edge_count:edge_count + num_edges]
            edge_count += num_edges
            
            # Aggregate
            atom_feat_importance, bond_feat_importance = _aggregate_per_feature(
                graph_attributions.detach(), graph
            )
            atom_importance, bond_importance = _aggregate_per_atom_bond(
                atom_feat_importance, bond_feat_importance
            )
            edge_importance = _aggregate_per_edge(atom_importance, bond_importance, graph)
            edge_importance_list.append(edge_importance)

            # feat_importance_list.append(torch.cat([atom_feat_importance.flatten(), bond_feat_importance.flatten()]))
            # atom_bond_importance_list.append(torch.cat([atom_importance, bond_importance]))
        return edge_importance_list  #, feat_importance_list, atom_bond_importance_list

    def depict(self, graph, weights, intensity=1):
        return draw_molecule(graph, weights, factor=self.factor * intensity, shift=self.shift)

    def summary(self, graph, weights, index):
        # Verify the sum property (CRITICAL for IG correctness)
        attribution_sum = weights.sum().item()
        logger.debug("INTEGRATED GRADIENTS EXPLANATION", highlight=True)
        logger.debug(f"{graph.y.item():.2f} {graph.smiles}")
        logger.debug(f"\nBaseline prediction: {self.baseline_logits:.2f}")
        logger.debug(f"Attribution sum: {attribution_sum:.2f}")
        logger.debug(f"Baseline + Attribution sum: {self.baseline_logits + attribution_sum:.2f}")
        logger.debug(f"PREDICTION: {self.logits[index]:.2f}")
        logger.debug(weight_summary(weights, title="EDGE IMPORTANCE (averaged components):"))



# IG AGGREGATION:
# IG computes attributions per (arc, feature), not per arc as attention does. But the
# features of a chemical entity are duplicated across arcs (arcs incident to the same
# atom share it, and atoms and bonds are further duplicated because of bidirectional edges).
# Across different runs the "importance" of an atom/bond can be "spread" very inhomogeneously 
# across arcs. This 3-step aggregation process removes that degree of freedom:
# 1) attributions are gathered around each chemical entity (atom/bond), but each feature of
#    the entity is still visible: per-entity feature vectors [num_atoms, ATOM_DIM],
#    [num_bonds, BOND_DIM]
# 2) the feature dimension is collapsed: we have a scalar for each entity
# 3) the attributions are gathered back to the arc dimension, but in a uniform way (each
#    entity's scalar is split evenly over the arcs touching it; the total is preserved,
#    arc-level evidence is lost)

def _aggregate_per_feature(attributions, graph):
    """Returns atom_feat_importance [num_atoms, atom_dim] and bond_feat_importance [num_bonds, bond_dim]."""

    # Feature dimensions
    atom_dim = graph.x.shape[1]
    bond_dim = graph.edge_attr.shape[1]
    assert attributions.shape[1] == 2 * atom_dim + bond_dim, \
        f"Attribution width {attributions.shape[1]} != 2*{atom_dim} + {bond_dim}"  # DEBUG
    
    # Split attributions by type
    src_attr = attributions[:, :atom_dim]            # [num_edges, num_atom_features]
    dst_attr = attributions[:, atom_dim:2*atom_dim]  # [num_edges, num_atom_features]
    edge_attr = attributions[:, 2*atom_dim:]         # [num_edges, num_bond_features]
        
    # Initialize feature-level importance tensors
    num_atoms = graph.x.shape[0]
    num_bonds = graph.mol.GetNumBonds()
    atom_feat_importance = torch.zeros(num_atoms, atom_dim, device=attributions.device)
    bond_feat_importance = torch.zeros(num_bonds, bond_dim, device=attributions.device)
     
    # Accumulate feature vectors for each unique atom/bond
    src_idx = graph.edge_index[0]
    dst_idx = graph.edge_index[1]   
    for i in range(attributions.shape[0]):
        src = src_idx[i].item()
        dst = dst_idx[i].item()        
        # Atoms
        atom_feat_importance[src] += src_attr[i]
        atom_feat_importance[dst] += dst_attr[i]        
        # Bonds
        bond = graph.mol.GetBondBetweenAtoms(src, dst)
        bond_idx = bond.GetIdx()
        bond_feat_importance[bond_idx] += edge_attr[i]

    # print("\n\nFEATURE IMPORTANCE:")
    # print(f"Atom feature importance shape: {atom_feat_importance.shape}")
    # print(f"Bond feature importance shape: {bond_feat_importance.shape}")
    # print(f"Example atom features (atom 0): {atom_feat_importance[0]}")
    # print(f"Example bond features (bond 0): {bond_feat_importance[0]}")
    
    return atom_feat_importance, bond_feat_importance

def _aggregate_per_atom_bond(atom_feat_importance, bond_feat_importance):
    """Returns atom_importance [num_atoms] and bond_importance [num_bonds]."""
    # Sum across feature dimension
    atom_importance = atom_feat_importance.sum(dim=1)
    bond_importance = bond_feat_importance.sum(dim=1)
    # print("\n\nATOM/BOND IMPORTANCE:")        
    # print_weights(atom_importance, title="ATOM IMPORTANCE:")       
    # print_weights(bond_importance, title="BOND IMPORTANCE:")
    return atom_importance, bond_importance

def _aggregate_per_edge(atom_importance, bond_importance, graph):
    """Returns edge_importance [num_edges]."""
    
    src_idx = graph.edge_index[0]
    dst_idx = graph.edge_index[1]
    num_edges = src_idx.shape[0]
    
    # Compute atom appearance counts from degree
    # Each atom appears in (degree × 2) edges (bidirectional)
    atom_count = degree(src_idx, num_nodes=graph.x.shape[0]) + \
                degree(dst_idx, num_nodes=graph.x.shape[0])
        
    # Build edge importance with averaged components (single pass)
    edge_importance = torch.zeros(num_edges)
    
    for i in range(num_edges):
        src = src_idx[i].item()
        dst = dst_idx[i].item()
        bond = graph.mol.GetBondBetweenAtoms(src, dst)
        bond_idx = bond.GetIdx()
        
        # Each component divided by its count
        edge_importance[i] = (
            atom_importance[src] / atom_count[src] + 
            atom_importance[dst] / atom_count[dst] + 
            bond_importance[bond_idx] / 2  # Bonds appear exactly twice (bidirectional)
        )
    
    return edge_importance



    # SHAP code
    #
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