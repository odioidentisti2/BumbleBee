import torch

def test_centered_attributions():
    """Test centered attribution calculation in various scenarios"""
    
    neutral_point = 0.5
    
    test_cases = [
        # (baseline_pred, edge_importance, description)
        # final_pred will be calculated as: baseline + sum(edge_importance)
        (-2.0, torch.tensor([2.0, 0.8]), "Case 1: baseline < neutral < final"),
        (-2.0, torch.tensor([2.0, 0.0]), "Case 2: baseline < neutral < final, one zero attr"),
        (-2.0, torch.tensor([1.5, -0.8]), "Case 3: baseline < final < neutral, mixed signs"),
        (-2.0, torch.tensor([0.5, 0.3]), "Case 4: baseline < final < neutral, same sign"),
        (-2.0, torch.tensor([-0.5, -0.3]), "Case 5: final < baseline < neutral, both negative"),
        (0.8, torch.tensor([-0.5, -0.1]), "Case 6: final < neutral < baseline"),
        (0.2, torch.tensor([0.4, 0.1]), "Case 7: baseline < neutral < final"),
        (0.6, torch.tensor([-0.2, -0.05]), "Case 8: final < baseline < neutral"),
    ]
    
    for baseline_pred, edge_importance, description in test_cases:
        final_pred = baseline_pred + edge_importance.sum().item()
        
        print(f"\n{'='*80}")
        print(f"{description}")
        print(f"{'='*80}")
        
        # Calculate centered attributions
        offset = neutral_point - baseline_pred
        num_edges = edge_importance.shape[0]
        shift_per_edge = offset / num_edges
        centered_importance = edge_importance - shift_per_edge
        
        # Verify
        reconstructed_pred = neutral_point + centered_importance.sum().item()
        
        print(f"Baseline prediction:      {baseline_pred:.4f}")
        print(f"Neutral point:            {neutral_point:.4f}")
        print(f"Final prediction:         {final_pred:.4f}")
        print(f"Order: ", end="")
        values = sorted([(baseline_pred, "baseline"), (neutral_point, "neutral"), (final_pred, "final")])
        print(" < ".join([name for _, name in values]))
        
        print(f"\nOffset (neutral - base):  {offset:.4f}")
        print(f"Shift per edge:           {shift_per_edge:.4f}")
        print(f"\nOriginal attributions:    {edge_importance.numpy()}")
        print(f"Centered attributions:    {centered_importance.numpy()}")
        print(f"\nOriginal sum:             {edge_importance.sum().item():.4f}")
        print(f"Centered sum:             {centered_importance.sum().item():.4f}")
        print(f"\nVerification:")
        print(f"  baseline + original_sum = {baseline_pred:.4f} + {edge_importance.sum().item():.4f} = {final_pred:.4f}")
        print(f"  neutral + centered_sum  = {neutral_point:.4f} + {centered_importance.sum().item():.4f} = {reconstructed_pred:.4f}")
        print(f"  Match: {abs(reconstructed_pred - final_pred) < 1e-6} ✓" if abs(reconstructed_pred - final_pred) < 1e-6 else f"  Match: False ✗")
        
        print(f"\nInterpretation:")
        for i, (orig, centered) in enumerate(zip(edge_importance.numpy(), centered_importance.numpy())):
            direction = "→ class 1 (toxic)" if centered > 0 else "→ class 0 (stable)" if centered < 0 else "neutral"
            print(f"  Edge {i}: {orig:+.2f} → {centered:+.2f}  {direction}")

# Run tests
test_centered_attributions()