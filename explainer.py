import torch
from torch_geometric.data import Batch
import utils
from graphic import *

from molecular_data import ATOM_DIM



# DEBUG
def print_weights(weights, average=False, title="WEIGHTS:"):
    print(f"\n{title}")
    print(weights)
    print(f"Weights range: {weights.min():.4f} - {weights.max():.4f}")
    if average: print("Weights Average: ", f"{weights.mean().item():.4f}")
    print(f"Weight sum: {weights.sum():.2f}")
    print()


class Explainer:

    def __init__(self, model):
        self.model = model.cpu()
        self.intensity = 1
        self.count = 0  # DEBUG

    def explain(self, dataset):
        self.model.eval()
        utils.print_header("CALIBRATION")
        print(f"Prediction distribution mean/std: {self.model.training_predictions.mean():.2f} / {self.model.training_predictions.std():.2f}")
        print(f"Prediction range: {self.model.training_predictions.min():.2f} to {self.model.training_predictions.max():.2f}")
        print(f"IG top: {self.model.training_predictions.std():.2f}")
        print(f"ATT top: {self.model.att_factor_top:.2f}")
        aw = []
        ig = []
        for graph in dataset:
            self.count = 1  # DEBUG
            repeat = True
            while repeat:
                aw.append(self._attention(graph.clone()))  # why clone()?
                ig.append(self._integrated_gradients(graph.clone()))
                # self.explain_with_mlp_IG(graph.clone())
                # user_input = ''
                user_input = input("Press Enter to continue, '-' to halve intensity, '+' to double intensity: ")
                plus_count = user_input.count('+')
                minus_count = user_input.count('-')
                if plus_count + minus_count > 0:
                    self.intensity *= (2 ** plus_count) / (2 ** minus_count)
                else:
                    repeat = False  # Move to next molecule
            # return aw, ig
        return aw, ig

    def _attention(self, graph):
        graph = graph.cpu()
        batched_graph = Batch.from_data_list([graph])
        # edge_feat = model.get_features(batched_graph)
        with torch.no_grad():
            _, weights = self.model(batched_graph, return_attention=True)
            # _, weights = model.single_forward(edge_feat, batched_graph.edge_index, batched_graph.batch, return_attention=True)[0]  # remove batch
        weights = weights[0].detach()  # remove batch

        if self.count == 1:
            utils.print_header("ATTENTION-BASED EXPLANATION")
            print(f"{graph.y.item():.2f}", graph.smiles)
            print_weights(weights, average=True)

        # Weights come after softmax (they add up to 1): 
        # => weights.mean() = 1 / len(weights)
        # Therefore:
        #   weight * len(weights) == 1  means "average attention"
        scores = weights * len(weights)  # visualize the proportion to average attention
        shift = -1  # shift so that average attention is at 0
        factor = self.intensity / (self.model.att_factor_top + shift)  # scale so that top attention is at 1
        depict(graph, scores, factor=factor, shift=shift, positive_only=True)
        return weights


    def _integrated_gradients(self, graph, steps=100):
        graph = graph.cpu()
        batched_graph = Batch.from_data_list([graph])
        edge_feat = self.model.get_features(batched_graph)   
        baseline = torch.zeros_like(edge_feat)
        integrated_grads = torch.zeros_like(edge_feat)

        for alpha in torch.linspace(0, 1, steps):  # Interpolate between baseline and input            
            interp_feat = baseline + alpha * (edge_feat - baseline)
            interp_feat.requires_grad_(True)
            prediction = self.model.single_forward(interp_feat, batched_graph.edge_index, batched_graph.batch)
            integrated_grads += torch.autograd.grad(
                                outputs=prediction,
                                inputs=interp_feat,
                                create_graph=False
                            )[0]
            if alpha == 0:
                baseline_pred = prediction.item()
            elif alpha == 1:
                final_pred = prediction.item()
        # Average gradients and scale by input difference
        integrated_grads /= steps
        attributions = (edge_feat - baseline) * integrated_grads

        if self.count == 1:
            _ig_summary(graph, attributions, baseline_pred, final_pred)

        factor = self.intensity
        if not hasattr(graph, 'label'):  # regression
            factor *= 1 / self.model.training_predictions.std().item()

        graph.prediction = final_pred  # DEBUG

        edge_importance = self._edge_importance(attributions.detach(), graph)
        depict(graph, edge_importance, factor=factor)

        atom_importance, bond_importance = self._atom_bond_importance(attributions.detach(), graph)
        depict_feat(graph, atom_importance, bond_importance, factor=factor)
        # aggregated_importance = torch.cat([atom_importance, bond_importance])
        # return aggregated_importance


    def _edge_importance(self, attributions, graph):
        print("\n\nEDGE INTEGRATED GRADIENTS")
        edge_importance = attributions.sum(dim=1)  # Sum across feature dimensions
        print_weights(edge_importance)
        return edge_importance


    def _atom_bond_importance(self, attributions, graph):
        print("\n\nATOM/BOND INTEGRATED GRADIENTS")
        feature_importance = attributions  # shape: [num_edges, num_features_per_edge]
        # print("\nPER-FEATURE INTEGRATED GRADIENTS")
        # print(f"Shape: {feature_importance.shape}")
        # print(f"Example (first edge): {feature_importance[0]}")
        src_attr = feature_importance[:, :ATOM_DIM]
        dst_attr = feature_importance[:, ATOM_DIM:2*ATOM_DIM]
        edge_attr = feature_importance[:, 2*ATOM_DIM:]
        # Per-edge scalars
        edge_bond_scalar = edge_attr.sum(dim=1)        # [num_edges]
        edge_atom_src = src_attr.sum(dim=1)            # [num_edges]
        edge_atom_dst = dst_attr.sum(dim=1)            # [num_edges]

        # Aggregate to bond instances (one value per RDKit bond)
        num_bonds = graph.mol.GetNumBonds()
        bond_importance = torch.zeros(num_bonds)
        # Aggregate to atom instances (one value per atom index)
        num_atoms = graph.x.shape[0]
        atom_importance = torch.zeros(num_atoms)

        src_idx = graph.edge_index[0].cpu()
        dst_idx = graph.edge_index[1].cpu()
        for i in range(edge_bond_scalar.shape[0]):
            src = int(src_idx[i])
            dst = int(dst_idx[i])
            # accumulate atom contributions for the specific atom indices
            atom_importance[src] += edge_atom_src[i].item()
            atom_importance[dst] += edge_atom_dst[i].item()
            # find bond instance and accumulate bond contribution
            bond = graph.mol.GetBondBetweenAtoms(src, dst)
            bond_importance[bond.GetIdx()] += edge_bond_scalar[i].item()

        print_weights(atom_importance, title="ATOM IMPORTANCE:")
        print_weights(bond_importance, title="BOND IMPORTANCE:")
 
        # print("\nSPLIT FEATURE ATTRIBUTIONS")
        # print(f"Source node attribution shape: {src_attr.shape}")
        # print(f"Destination node attribution shape: {dst_attr.shape}")
        # print(f"Edge (bond) attribution shape: {edge_attr.shape}")
        # print(f"First edge src: {src_attr[0]}")
        # print(f"First edge dst: {dst_attr[0]}")
        # print(f"First edge bond: {edge_attr[0]}")

        return atom_importance, bond_importance
    

def _ig_summary(graph, attributions, baseline_pred, final_pred):
    utils.print_header("INTEGRATED GRADIENTS EXPLAINATION   ")
    print(f" {graph.y.item():.2f}", graph.smiles)
    # Verify the sum property (CRITICAL for IG correctness)
    attribution_sum = attributions.sum().item()
    # expected_sum = (final_pred - baseline_pred).item()   
    print(f"\nBaseline prediction: {baseline_pred:.2f}")
    print(f"Attribution sum: {attribution_sum:.2f}")
    print(f"Baseline + Attribution sum: {baseline_pred + attribution_sum:.2f}")    
    print(f"PREDICTION: {final_pred:.2f}")





    # def explain_with_mlp_IG(self, graph, steps=50):
    #     """
    #     Integrated gradients for the output of input_mlp (edge embeddings).
    #     """
    #     graph = graph.to('cpu')

    #     # Get edge features and baseline
    #     batched_graph = Batch.from_data_list([graph])
    #     edge_feat = self.model.get_features(batched_graph)
    #     baseline_emb = torch.zeros_like(self.model.input_mlp(edge_feat))
    #     edge_emb = self.model.input_mlp(edge_feat).detach()

    #     integrated_grads = torch.zeros_like(edge_emb)

    #     for alpha in torch.linspace(0, 1, steps):
    #         interp_emb = baseline_emb + alpha * (edge_emb - baseline_emb)
    #         interp_emb = interp_emb.detach().clone().requires_grad_(True)

    #         # --- Forward hook trick: temporarily replace input_mlp's forward to return interp_emb ---
    #         orig_forward = self.model.input_mlp.forward
    #         self.model.input_mlp.forward = lambda *_: interp_emb

    #         # Forward pass
    #         pred = self.model(batched_graph)
    #         pred = torch.flatten(pred)[0]

    #         # Backward pass
    #         self.model.zero_grad()
    #         pred.backward()
    #         grad = interp_emb.grad.detach()
    #         integrated_grads += grad

    #         # Restore original forward
    #         self.model.input_mlp.forward = orig_forward

    #     integrated_grads /= steps
    #     attributions = (edge_emb - baseline_emb) * integrated_grads
    #     edge_importance = attributions.sum(dim=1)  # Sum across embedding dimensions

    #     # After computing attributions
    #     pred_real = float(self.model(batched_graph).detach().cpu().numpy())
    #     # Temporarily override input_mlp to always return baseline_emb
    #     orig_forward = self.model.input_mlp.forward
    #     self.model.input_mlp.forward = lambda *_: baseline_emb
    #     pred_base = float(self.model(batched_graph).detach().cpu().numpy())
    #     self.model.input_mlp.forward = orig_forward

    #     weights = edge_importance.detach().cpu()


    #     print("\nDEPICT MLP INTEGRATED GRADIENTS")
        # print(f"{graph.y.item():.2f}", graph.smiles)
    #     print(f"Prediction (baseline): {pred_base:.2f}")
    #     print(f"Sum of attributions: {edge_importance.sum():.2f}")
    #     print(f"Difference: {pred_real - pred_base:.2f}")
    #     print_weights(weights)
    #     depict(graph, weights.numpy() * self.intensity/ 10, attention=False)
    #     return edge_importance

