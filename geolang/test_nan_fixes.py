#!/usr/bin/env python3
"""
Quick test script to verify NaN fixes in DGGM and loss computation.
Run this before training to confirm the fixes are working.
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/tejass/Downloads/TUDELFT_ROBOTICS/Robotics_Q3/CV/HAGRID/geolang')

from model.DGGM import DGGM
from types import SimpleNamespace


def test_dggm_stability():
    """Test that DGGM doesn't produce NaN with extreme geometry values."""
    print("=" * 80)
    print("Testing DGGM Numerical Stability")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create DGGM module
    dggm = DGGM(dim=512).to(device)
    dggm.eval()
    
    # Test case 1: Normal inputs
    print("\nTest 1: Normal input shapes")
    B, H, W, C = 2, 26, 26, 512
    x = torch.randn(B, H, W, C, device=device)
    depth = torch.randn(B, 1, H, W, device=device)
    
    with torch.no_grad():
        try:
            out = dggm(x, depth)
            has_nan = torch.isnan(out).any()
            print(f"  Output shape: {out.shape}")
            print(f"  Contains NaN: {has_nan}")
            print(f"  Output range: [{out[~torch.isnan(out)].min().item():.6f}, {out[~torch.isnan(out)].max().item():.6f}]")
            print("  ✓ PASS" if not has_nan else "  ✗ FAIL")
        except Exception as e:
            print(f"  ✗ FAIL: {e}")
    
    # Test case 2: Extreme geometry values
    print("\nTest 2: Extreme geometry values (large depth differences)")
    x = torch.randn(B, H, W, C, device=device)
    depth = torch.zeros(B, 1, H, W, device=device)
    # Create sharp depth discontinuities
    depth[:, :, :H//2, :] = 0
    depth[:, :, H//2:, :] = 100  # Large jump
    
    with torch.no_grad():
        try:
            out = dggm(x, depth)
            has_nan = torch.isnan(out).any()
            print(f"  Output shape: {out.shape}")
            print(f"  Contains NaN: {has_nan}")
            print(f"  Output range: [{out[~torch.isnan(out)].min().item():.6f}, {out[~torch.isnan(out)].max().item():.6f}]")
            print("  ✓ PASS" if not has_nan else "  ✗ FAIL")
        except Exception as e:
            print(f"  ✗ FAIL: {e}")
    
    # Test case 3: Uniform depth (minimal geometry contribution)
    print("\nTest 3: Uniform depth (minimal geometry effect)")
    x = torch.randn(B, H, W, C, device=device)
    depth = torch.ones(B, 1, H, W, device=device) * 5.0
    
    with torch.no_grad():
        try:
            out = dggm(x, depth)
            has_nan = torch.isnan(out).any()
            print(f"  Output shape: {out.shape}")
            print(f"  Contains NaN: {has_nan}")
            print(f"  Output range: [{out[~torch.isnan(out)].min().item():.6f}, {out[~torch.isnan(out)].max().item():.6f}]")
            print("  ✓ PASS" if not has_nan else "  ✗ FAIL")
        except Exception as e:
            print(f"  ✗ FAIL: {e}")
    
    print("\n" + "=" * 80)


def test_loss_stability():
    """Test that loss computation doesn't produce NaN."""
    print("\n" + "=" * 80)
    print("Testing Loss Computation Stability")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test case 1: Normal loss values
    print("\nTest 1: Normal prediction and target values")
    B, H, W = 2, 104, 104
    pred = torch.randn(B, 1, H, W, device=device)
    mask = torch.rand(B, 1, H, W, device=device) > 0.5
    mask = mask.float()
    
    try:
        weight = mask * 0.5 + 1
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, mask, weight=weight)
        has_nan = torch.isnan(loss)
        print(f"  Loss value: {loss.item():.6f}")
        print(f"  Contains NaN: {has_nan}")
        print("  ✓ PASS" if not has_nan else "  ✗ FAIL")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
    
    # Test case 2: Extreme prediction values
    print("\nTest 2: Extreme prediction values")
    pred = torch.randn(B, 1, H, W, device=device) * 100  # Very large values
    mask = torch.rand(B, 1, H, W, device=device) > 0.5
    mask = mask.float()
    
    try:
        weight = mask * 0.5 + 1
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, mask, weight=weight)
        has_nan = torch.isnan(loss)
        print(f"  Loss value: {loss.item():.6f}")
        print(f"  Contains NaN: {has_nan}")
        print("  ✓ PASS" if not has_nan else "  ✗ FAIL")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
    
    # Test case 3: smooth_l1_loss
    print("\nTest 3: smooth_l1_loss computation")
    grasp_qua_pred = torch.randn(B, 1, H, W, device=device)
    grasp_qua_mask = torch.rand(B, 1, H, W, device=device)
    
    try:
        loss = torch.nn.functional.smooth_l1_loss(grasp_qua_pred, grasp_qua_mask)
        has_nan = torch.isnan(loss)
        print(f"  Loss value: {loss.item():.6f}")
        print(f"  Contains NaN: {has_nan}")
        print("  ✓ PASS" if not has_nan else "  ✗ FAIL")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
    
    print("\n" + "=" * 80)


def test_dggm_parameters():
    """Verify DGGM parameter initialization."""
    print("\n" + "=" * 80)
    print("Testing DGGM Parameter Initialization")
    print("=" * 80)
    
    dggm = DGGM(dim=512)
    
    print(f"\nParameter ranges:")
    print(f"  lambda1 (depth weight): {dggm.lambda1.item():.6f}")
    print(f"  lambda2 (spatial weight): {dggm.lambda2.item():.6f}")
    print(f"  eta (decay factor): {dggm.eta.item():.6f}")
    
    # Verify eta is in valid range
    if 0 < dggm.eta.item() < 1:
        print("  ✓ eta is in valid range (0, 1)")
    else:
        print(f"  ⚠️  eta={dggm.eta.item()} is outside (0, 1) - may cause issues!")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("\n🔍 Running NaN Fix Verification Tests\n")
    
    try:
        test_dggm_parameters()
        test_dggm_stability()
        test_loss_stability()
        
        print("\n✅ All tests completed!")
        print("\nNext steps:")
        print("  1. If all tests PASS, you can proceed with training")
        print("  2. If any test FAIL, check the error messages above")
        print("  3. Run training with: python train_geolang.py --config config/OCID-VLG/geolang_multiple_r50.yaml")
        print("  4. If NaN still appears, check NAN_DEBUG_GUIDE.md for additional fixes")
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
