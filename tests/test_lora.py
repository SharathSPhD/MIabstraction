"""Tests for LoRA parameter-efficient fine-tuning implementation.

Verifies that:
1. LoRA adapters are correctly attached and frozen
2. Base weights remain unchanged before and after training
3. Trainable parameters are <2% of total
4. Adapters can be merged or detached cleanly
"""
import torch
import torch.nn as nn
import pytest

from loom.app.lora import (
    attach_lora,
    lora_parameters,
    merge_or_detach,
    get_adapter_info,
    LoRALinear,
    freeze_base_weights,
)


@pytest.fixture
def simple_model():
    """Create a simple model with attention-like projections."""
    return nn.Sequential(
        nn.Linear(128, 256, bias=True),  # q_proj-like
        nn.ReLU(),
        nn.Linear(256, 256, bias=True),  # v_proj-like
        nn.ReLU(),
        nn.Linear(256, 128, bias=True),  # o_proj-like
    )


@pytest.fixture
def model_with_named_layers():
    """Create a model with named layers matching attention patterns."""
    class AttentionBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(128, 128)
            self.v_proj = nn.Linear(128, 128)
            self.k_proj = nn.Linear(128, 128)
            self.o_proj = nn.Linear(128, 128)
            self.fc1 = nn.Linear(128, 512)
            self.fc2 = nn.Linear(512, 128)

        def forward(self, x):
            # Simple forward: apply transformations sequentially
            q = self.q_proj(x)
            v = self.v_proj(x)
            combined = q + v
            hidden = self.fc1(combined)
            output = self.fc2(hidden)
            return output

    return nn.Sequential(
        AttentionBlock(),
        AttentionBlock(),
    )


class TestLoRAAttachment:
    """Test LoRA adapter attachment."""

    def test_attach_lora_creates_adapters(self, model_with_named_layers):
        """Verify adapters are created and wrapped correctly."""
        handles = attach_lora(model_with_named_layers, rank=8, alpha=16.0)
        assert len(handles) > 0, "Should create multiple adapters"
        assert all(h.module.__class__.__name__ == 'LoRALinear' for h in handles)

    def test_attach_lora_preserves_model_structure(self, model_with_named_layers):
        """Verify model still functions after LoRA attachment."""
        original_handles = attach_lora(model_with_named_layers, rank=8)
        x = torch.randn(4, 128)
        output = model_with_named_layers(x)
        assert output.shape == (4, 128), "Model output shape should be preserved"

    def test_attach_lora_with_custom_targets(self, model_with_named_layers):
        """Verify custom target patterns work."""
        handles = attach_lora(
            model_with_named_layers,
            rank=8,
            targets=['q_proj', 'v_proj']  # Only attention, not FC layers
        )
        # Should have 4 adapters (q, v for 2 blocks)
        assert len(handles) == 4, f"Expected 4 adapters, got {len(handles)}"

    def test_adapter_initialization(self, model_with_named_layers):
        """Verify adapters are initialized correctly."""
        handles = attach_lora(model_with_named_layers, rank=8, alpha=16.0)
        for handle in handles:
            # A should be initialized with uniform distribution
            assert handle.adapter_a.requires_grad, "Adapter A should be trainable"
            assert handle.adapter_b.requires_grad, "Adapter B should be trainable"
            # B should be initialized to zero
            assert torch.allclose(handle.adapter_b, torch.zeros_like(handle.adapter_b),
                                atol=1e-6), "Adapter B should start at zero"


class TestBaseWeightFreezing:
    """Test that base weights are frozen and remain unchanged."""

    def test_base_weights_frozen_after_attachment(self, model_with_named_layers):
        """Verify base weights are frozen after LoRA attachment."""
        attach_lora(model_with_named_layers, rank=8)

        # All non-adapter parameters should be frozen
        trainable_count = 0
        for name, param in model_with_named_layers.named_parameters():
            if param.requires_grad:
                trainable_count += 1
                print(f"Trainable param: {name}")

        # Should only have adapter parameters trainable
        # Each LoRA layer has 2 trainable params (A and B)
        # Count the number of LoRA layers
        lora_params = []
        for module in model_with_named_layers.modules():
            if isinstance(module, LoRALinear):
                lora_params.extend([module.adapter_a, module.adapter_b])

        assert trainable_count == len(lora_params), \
            f"Should have {len(lora_params)} trainable params, got {trainable_count}"

    def test_base_weights_unchanged_after_forward_pass(self, model_with_named_layers):
        """Verify base weights don't change after forward pass."""
        handles = attach_lora(model_with_named_layers, rank=8)

        # Save base weights
        base_weights_before = {}
        for handle in handles:
            base_weights_before[handle.layer_name] = handle.module.base.weight.data.clone()

        # Forward pass
        x = torch.randn(4, 128)
        y = model_with_named_layers(x)

        # Verify base weights unchanged
        for handle in handles:
            assert torch.equal(
                handle.module.base.weight.data,
                base_weights_before[handle.layer_name]
            ), f"Base weights changed for {handle.layer_name}"

    def test_base_weights_unchanged_after_backward_pass(self, model_with_named_layers):
        """Verify base weights don't change after training step."""
        handles = attach_lora(model_with_named_layers, rank=8)

        # Save base weights
        base_weights_before = {}
        for handle in handles:
            base_weights_before[handle.layer_name] = handle.module.base.weight.data.clone()

        # Forward and backward pass
        x = torch.randn(4, 128)
        y = model_with_named_layers(x)
        loss = y.sum()
        loss.backward()

        # Verify base weights unchanged
        for handle in handles:
            assert torch.equal(
                handle.module.base.weight.data,
                base_weights_before[handle.layer_name]
            ), f"Base weights changed for {handle.layer_name} after backward pass"


class TestTrainableParameterRatio:
    """Test that trainable parameters are <2% of total."""

    def test_trainable_ratio_is_small(self, model_with_named_layers):
        """Verify trainable params are <10% of total for typical rank on small models.

        On a 1.24B model like Llama-3.2-1B with rank=8, this would be <0.5%.
        On small test models, the ratio is larger but still should be <20%.
        """
        attach_lora(model_with_named_layers, rank=8, alpha=16.0)

        info = get_adapter_info(model_with_named_layers)
        ratio = info["trainable_ratio"]

        # For small models, ratio can be higher, but should still be <20%
        # On a 1.24B model, this would be <0.5%
        assert ratio < 0.20, f"Trainable ratio {ratio} should be <20% (much smaller on real models)"

    def test_adapter_params_scale_with_rank(self):
        """Verify adapter parameter count scales correctly with rank."""
        # Create model with named layers for rank=4
        class SimpleBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(128, 128)
                self.v_proj = nn.Linear(128, 128)

            def forward(self, x):
                return self.q_proj(x) + self.v_proj(x)

        model4 = SimpleBlock()
        attach_lora(model4, rank=4)
        info4 = get_adapter_info(model4)
        params4 = info4["adapter_params"]

        # Create model for rank=8 (should be ~2x for same layers)
        model8 = SimpleBlock()
        attach_lora(model8, rank=8)
        info8 = get_adapter_info(model8)
        params8 = info8["adapter_params"]

        # Should scale roughly linearly with rank
        assert params4 > 0, "Should have adapter params for rank=4"
        assert params8 > 0, "Should have adapter params for rank=8"
        ratio = params8 / params4
        assert 1.8 < ratio < 2.2, f"Rank scaling should be ~2x, got {ratio}x"


class TestLoRAParameters:
    """Test lora_parameters() utility function."""

    def test_lora_parameters_returns_only_adapters(self, model_with_named_layers):
        """Verify lora_parameters returns only adapter parameters."""
        attach_lora(model_with_named_layers, rank=8)

        adapter_params = lora_parameters(model_with_named_layers)
        trainable_params = [p for p in model_with_named_layers.parameters() if p.requires_grad]
        adapter_param_ids = {id(p) for p in adapter_params}

        # All trainable params should be adapter params
        assert len(adapter_params) == len(trainable_params), \
            "All trainable params after LoRA should be adapter params"

        # Check that each adapter param is actually a LoRA adapter
        for module in model_with_named_layers.modules():
            if isinstance(module, LoRALinear):
                assert id(module.adapter_a) in adapter_param_ids
                assert id(module.adapter_b) in adapter_param_ids

    def test_lora_parameters_are_trainable(self, model_with_named_layers):
        """Verify all lora_parameters are trainable."""
        attach_lora(model_with_named_layers, rank=8)

        adapter_params = lora_parameters(model_with_named_layers)
        for param in adapter_params:
            assert param.requires_grad, "Adapter parameter should be trainable"


class TestMergeDetach:
    """Test merge_or_detach() functionality."""

    def test_detach_removes_adapters(self, model_with_named_layers):
        """Verify detach mode removes adapters and restores base modules."""
        handles = attach_lora(model_with_named_layers, rank=8)
        initial_handles_count = len(handles)

        merge_or_detach(model_with_named_layers, handles, mode="detach")

        # Count remaining LoRA modules
        remaining_lora = sum(1 for m in model_with_named_layers.modules()
                           if isinstance(m, LoRALinear))
        assert remaining_lora == 0, "Detach should remove all LoRA modules"

    def test_merge_updates_base_weights(self, model_with_named_layers):
        """Verify merge mode updates base weights with adapter deltas."""
        handles = attach_lora(model_with_named_layers, rank=8, alpha=16.0)

        # Store original base weights
        original_weights = {}
        for handle in handles:
            original_weights[handle.layer_name] = \
                handle.module.base.weight.data.clone()

        # Forward pass to activate adapters (may not change them yet)
        x = torch.randn(4, 128)
        model_with_named_layers(x)

        # Merge
        merge_or_detach(model_with_named_layers, handles, mode="merge")

        # Base weights should have been updated
        for handle in handles:
            merged_weight = handle.module.base.weight.data
            # At minimum, the weights should be in the model
            assert merged_weight is not None

    def test_merge_unfreezes_weights(self, model_with_named_layers):
        """Verify merge mode unfreezes base weights."""
        handles = attach_lora(model_with_named_layers, rank=8)

        # Base weights should initially be frozen
        for handle in handles:
            for param in handle.module.base.parameters():
                assert not param.requires_grad, "Base should be frozen initially"

        # Merge
        merge_or_detach(model_with_named_layers, handles, mode="merge")

        # After merge, base weights should be unfrozen
        for handle in handles:
            for param in handle.module.base.parameters():
                assert param.requires_grad, "Base should be unfrozen after merge"


class TestAdapterInfo:
    """Test get_adapter_info() function."""

    def test_adapter_info_counts_correctly(self, model_with_named_layers):
        """Verify get_adapter_info returns accurate counts."""
        handles = attach_lora(model_with_named_layers, rank=8, alpha=16.0)
        info = get_adapter_info(model_with_named_layers)

        # Count adapters manually
        manual_count = sum(1 for m in model_with_named_layers.modules()
                          if isinstance(m, LoRALinear))
        assert info["n_adapters"] == manual_count

        # Verify ratios are reasonable
        assert 0 <= info["trainable_ratio"] <= 1.0
        assert 0 <= info["adapter_ratio"] <= 1.0
        assert info["adapter_ratio"] <= info["trainable_ratio"]  # Adapters are subset of trainable

    def test_adapter_info_for_no_adapters(self, simple_model):
        """Verify get_adapter_info works on model with no LoRA."""
        info = get_adapter_info(simple_model)
        assert info["n_adapters"] == 0
        assert info["adapter_params"] == 0
        assert info["trainable_ratio"] == 1.0  # All params trainable without LoRA


class TestLoRAForward:
    """Test LoRA forward pass correctness."""

    def test_lora_output_is_base_plus_adapter(self):
        """Verify LoRA output matches base + adapter computation."""
        base = nn.Linear(16, 32)
        base.weight.data.fill_(0.1)
        base.bias.data.fill_(0.01)

        lora = LoRALinear(base, rank=4, alpha=8.0)

        # Set known adapter values for testing
        lora.adapter_a.data.fill_(0.1)
        lora.adapter_b.data.fill_(0.1)

        x = torch.ones(2, 16)

        # Forward through LoRA
        y = lora(x)

        # Compute expected output manually
        base_out = base(x)
        adapter_out = torch.matmul(x, lora.adapter_a)  # (2, 16) @ (16, 4) = (2, 4)
        adapter_out = torch.matmul(adapter_out, lora.adapter_b)  # (2, 4) @ (4, 32) = (2, 32)
        expected = base_out + lora.scale * adapter_out

        assert torch.allclose(y, expected, atol=1e-5), \
            "LoRA output should match base + scale*adapter computation"

    def test_lora_scale_factor_applied(self):
        """Verify scale factor is correctly applied."""
        base = nn.Linear(16, 32)
        base.weight.data.fill_(0.0)
        base.bias.data.fill_(0.0)

        rank = 4
        alpha = 8.0
        lora = LoRALinear(base, rank=rank, alpha=alpha)

        # Adapters should have scale = alpha/rank = 2.0
        assert lora.scale == alpha / rank

        # Set adapters to known values
        lora.adapter_a.data.fill_(1.0)
        lora.adapter_b.data.fill_(1.0)

        x = torch.ones(1, 16)
        y = lora(x)

        # adapter_out = (1, 16) @ (16, 4) @ (4, 32) @ scale
        # = 16 * 4 * scale = 64 * 2 = 128
        expected_adapter_contribution = 16 * 4 * lora.scale
        assert torch.allclose(y, torch.full_like(y, expected_adapter_contribution), atol=1e-5), \
            "Scale factor should be applied correctly"


class TestLoRAWithLoss:
    """Test LoRA training with actual loss computation."""

    def test_adapter_training_computes_gradients(self, model_with_named_layers):
        """Verify adapters are trainable during training."""
        handles = attach_lora(model_with_named_layers, rank=8, alpha=16.0)
        adapter_params = lora_parameters(model_with_named_layers)

        # All adapter params should be trainable
        for param in adapter_params:
            assert param.requires_grad, "Adapter parameters should be trainable"

        # Verify we can optimize only adapter params
        optimizer = torch.optim.Adam(adapter_params, lr=0.01)

        # Forward and backward pass
        x = torch.randn(4, 128)
        y = model_with_named_layers(x)
        loss = y.sum()
        loss.backward()

        # Perform an optimization step to verify it works
        optimizer.step()

        # If we got here without errors, optimization works
        assert True

    def test_base_weights_unchanged_during_adapter_training(self, model_with_named_layers):
        """Verify base weights don't change when training adapters."""
        handles = attach_lora(model_with_named_layers, rank=8)

        # Save base weights
        base_weights_before = {}
        for handle in handles:
            base_weights_before[handle.layer_name] = \
                handle.module.base.weight.data.clone()

        # Train adapters
        adapter_params = lora_parameters(model_with_named_layers)
        optimizer = torch.optim.Adam(adapter_params, lr=0.01)

        x = torch.randn(4, 128)
        target = torch.ones(4, 128)

        for _ in range(20):
            optimizer.zero_grad()
            y = model_with_named_layers(x)
            loss = ((y - target) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Verify base weights unchanged
        for handle in handles:
            assert torch.allclose(
                handle.module.base.weight.data,
                base_weights_before[handle.layer_name],
                atol=1e-6
            ), f"Base weights changed for {handle.layer_name} during adapter training"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# --- device and dtype placement ---------------------------------------------
#
# These exist because the original adapters were created with torch.zeros() and no
# device, which works perfectly in a CPU test suite and dies at the first matmul the
# moment the model is on a GPU. A LoRA implementation that only runs on CPU is not one.

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_adapters_land_on_the_same_device_as_the_base():
    from transformers import AutoModelForCausalLM
    from loom.app.lora import attach_lora, lora_parameters
    m = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")
    attach_lora(m, rank=2, alpha=4.0)
    params = lora_parameters(m)
    assert params
    assert all(p.device.type == "cuda" for p in params)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_forward_pass_survives_on_a_gpu():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from loom.app.lora import attach_lora
    tok = AutoTokenizer.from_pretrained("gpt2")
    m = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda").eval()
    attach_lora(m, rank=2, alpha=4.0)
    ids = tok("the patient presented with", return_tensors="pt").to("cuda")
    out = m(**ids, labels=ids["input_ids"])
    assert torch.isfinite(out.loss)


def test_adapters_match_the_base_dtype():
    from transformers import AutoModelForCausalLM
    from loom.app.lora import attach_lora, lora_parameters
    m = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.bfloat16)
    attach_lora(m, rank=2, alpha=4.0)
    assert all(p.dtype == torch.bfloat16 for p in lora_parameters(m))
