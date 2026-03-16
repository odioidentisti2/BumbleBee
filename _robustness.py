from bumblebee import *
import torch.nn.functional as F
from scipy.stats import pearsonr
import numpy as np
import utils
import time

def robustness(list1, list2):
    """Cosine similarity between two lists of tensors."""
    assert len(list1) == len(list2), "Lists must have same length"
    
    per_sample_robustness = []
    for t1, t2 in zip(list1, list2):
        assert len(t1) == len(t2), f"Length mismatch: {len(t1)} vs {len(t2)}"
        sim = F.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0)).item()
        per_sample_robustness.append(sim)
    
    mean_robustness = np.mean(per_sample_robustness)
    std_robustness = np.std(per_sample_robustness)
    return per_sample_robustness, mean_robustness, std_robustness


def robustness_pearson(list1, list2):
    per_sample_robustness = []
    for t1, t2 in zip(list1, list2):
        assert len(t1) == len(t2), f"Length mismatch: {len(t1)} vs {len(t2)}"
        corr, _ = pearsonr(t1.cpu().numpy(), t2.cpu().numpy())
        per_sample_robustness.append(corr)
    
    mean_robustness = np.mean(per_sample_robustness)
    std_robustness = np.std(per_sample_robustness)
    return per_sample_robustness, mean_robustness, std_robustness


def robustness_mae(list1, list2):
    per_sample_robustness = []
    for t1, t2 in zip(list1, list2):
        assert len(t1) == len(t2), f"Length mismatch: {len(t1)} vs {len(t2)}"
        mae = torch.abs(t1 - t2).mean().item()
        per_sample_robustness.append(mae)
    
    mean_robustness = np.mean(per_sample_robustness)
    std_robustness = np.std(per_sample_robustness)
    return per_sample_robustness, mean_robustness, std_robustness

print("\nEvaluating IG robustness aggregating attrubutions over atoms and bonds.\n")


def compare(sample1, sample2):
    per_sample_cos, mean_cos, std_cos = robustness(sample1, sample2)
    per_sample_pearson, mean_pearson, std_pearson = robustness_pearson(sample1, sample2)
    per_sample_mae, mean_mae, std_mae = robustness_mae(sample1, sample2)
    print(f"Robustness (42 vs 15):")
    print(f"  Cosine Similarity:  {mean_cos:.4f} ± {std_cos:.4f}")
    print(f"  Pearson Correlation: {mean_pearson:.4f} ± {std_pearson:.4f}")
    print(f"  MAE:                {mean_mae:.4f} ± {std_mae:.4f}")

if __name__ == "__main__":
    start = time.time()
    aw42, ig42 = main_loop(datasets.logp_split, device, 'logp_rand42_inj.pt')
    aw15, ig15 = main_loop(datasets.logp_split, device, 'logp_rand15_inj.pt')
    utils.print_header("ATTENTION")
    compare(aw42, aw15)
    utils.print_header("IG FEATURES")
    compare(ig42[0], ig15[0])
    utils.print_header("IG ATOMS & BONDS")
    compare(ig42[1], ig15[1])
    utils.print_header("IG EDGES")
    compare(ig42[2], ig15[2])
    print(f"\nTOTAL TIME FOR ROBUSTNESS: {time.time() - start:.0f}s")

# ig42_inj = main(device, datasets.logp_split, 'logp_rand42_inj.pt')
# ig15_inj = main(device, datasets.logp_split, 'logp_rand15_inj.pt')
# per_sample_cos_inj, mean_cos_inj, std_cos_inj = robustness(ig42_inj, ig15_inj)
# per_sample_pearson_inj, mean_pearson_inj, std_pearson_inj = robustness_pearson(ig42_inj, ig15_inj)
# per_sample_mae_inj, mean_mae_inj, std_mae_inj = robustness_mae(ig42_inj, ig15_inj)
# print(f"\nIG robustness with injection (ig42_inj vs ig15_inj):")
# print(f"  Cosine Similarity:  {mean_cos_inj:.4f} ± {std_cos_inj:.4f}")
# print(f"  Pearson Correlation: {mean_pearson_inj:.4f} ± {std_pearson_inj:.4f}")
# print(f"  MAE:                {mean_mae_inj:.4f} ± {std_mae_inj:.4f}")




    # m1 = main(device, datasets.muta, 'muta_benchmark.pt')
    # l1 = torch.cat(m1.stats[-1]['logits'])

    # m2 = main(device, datasets.muta, 'muta_RAND30.pt')
    # l2 = torch.cat(m2.stats[-1]['logits'])

    # logits_close = torch.allclose(l1, l2, rtol=1e-5, atol=1e-8)
    # logit_diff = (l1 - l2).abs()
    # print(f"Logits close: {logits_close}")
    # print(f"Logits difference: min={logit_diff.min().item():.6f}, max={logit_diff.max().item():.6f}, mean={logit_diff.mean().item():.6f}, std={logit_diff.std().item():.6f}   \n")
    
    # # Binary predictions
    # pred1 = torch.cat(m1.stats[-1]['predictions'])
    # pred2 = torch.cat(m2.stats[-1]['predictions'])
    # agree = sum(p1 == p2 for p1, p2 in zip(pred1, pred2))
    # print(f"\nModels agreement: {agree} / {len(pred1)}")
    # diff_idx = (pred1 != pred2).nonzero(as_tuple=True)[0]
    # print(f"Discordant: {diff_idx.numel()}") or (print("\n".join(
    #     f"[{i}] p1={int(pred1[i])} p2={int(pred2[i])} l1={l1[i].tolist()} l2={l2[i].tolist()}"
    #     for i in diff_idx[:20].tolist())))