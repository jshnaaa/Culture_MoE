# src/llamafactory/model/milora.py
"""
MiLoRA (Matrix-informed Low-Rank Adaptation) Implementation

MiLoRA improves upon LoRA by using SVD-based initialization instead of random initialization.
The key innovation is decomposing the original weight matrix W into:
- Principal matrix W_p (top m-r singular values) - frozen
- Minor matrix W_m (bottom r singular values) - used to initialize LoRA matrices

Reference: Based on the MiLoRA paper methodology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import logging
import numpy as np


class MiLoRALinear(nn.Module):
    """
    MiLoRA Linear layer that replaces standard Linear layers in the model.

    Unlike standard LoRA which uses random initialization, MiLoRA initializes
    the low-rank matrices using SVD decomposition of the original weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        dropout: float = 0.05,
        original_weight: Optional[torch.Tensor] = None,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.dropout_rate = dropout

        # Store original weight for SVD decomposition
        if original_weight is None:
            raise ValueError("MiLoRA requires original_weight for SVD initialization")

        # Validate rank
        max_rank = min(in_features, out_features)
        if rank >= max_rank:
            raise ValueError(f"Rank {rank} must be less than min(in_features, out_features) = {max_rank}")

        # Perform SVD decomposition and initialize MiLoRA components
        self._initialize_milora(original_weight, device, dtype)

        # Dropout layer
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Optional bias (usually False for most LLM layers)
        self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype)) if bias else None

        logging.info(f"MiLoRA layer initialized: {in_features}x{out_features}, rank={rank}")

    def _initialize_milora(self, original_weight: torch.Tensor, device: Optional[torch.device], dtype: Optional[torch.dtype]):
        """
        Initialize MiLoRA using SVD decomposition of original weights.

        Steps:
        1. Perform SVD: W = U Σ V^T
        2. Split into principal (W_p) and minor (W_m) matrices
        3. Initialize A_m and B_m from W_m
        4. Freeze W_p
        """
        W = original_weight.detach().clone().float()  # Ensure float for SVD

        # Step 1: SVD decomposition
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)

        # Step 2: Split singular values and vectors
        # Principal components (top m-r singular values) - frozen
        principal_rank = len(S) - self.rank

        U_p = U[:, :principal_rank]  # [out_features, principal_rank]
        S_p = S[:principal_rank]     # [principal_rank]
        Vt_p = Vt[:principal_rank, :] # [principal_rank, in_features]

        # Minor components (bottom r singular values) - for LoRA initialization
        U_m = U[:, principal_rank:]   # [out_features, rank]
        S_m = S[principal_rank:]      # [rank]
        Vt_m = Vt[principal_rank:, :] # [rank, in_features]

        # Step 3: Create principal matrix W_p (frozen)
        W_p = U_p @ torch.diag(S_p) @ Vt_p  # [out_features, in_features]
        self.register_buffer('W_p', W_p.to(device=device, dtype=dtype))

        # Step 4: Initialize LoRA matrices from minor components
        # W_m = B_m @ A_m where:
        # B_m = U_m @ sqrt(S_m)  [out_features, rank]
        # A_m = sqrt(S_m) @ Vt_m  [rank, in_features]
        sqrt_S_m = torch.sqrt(torch.clamp(S_m, min=1e-8))  # Avoid numerical issues

        B_m_init = U_m @ torch.diag(sqrt_S_m)  # [out_features, rank]
        A_m_init = torch.diag(sqrt_S_m) @ Vt_m  # [rank, in_features]

        # Create trainable parameters
        self.A_m = nn.Parameter(A_m_init.to(device=device, dtype=dtype))
        self.B_m = nn.Parameter(B_m_init.to(device=device, dtype=dtype))

        # Step 5: Verification (optional but recommended)
        self._verify_initialization(original_weight)

        logging.info(f"  Principal matrix W_p: {W_p.shape} (frozen)")
        logging.info(f"  LoRA matrix A_m: {self.A_m.shape} (trainable)")
        logging.info(f"  LoRA matrix B_m: {self.B_m.shape} (trainable)")

    def _verify_initialization(self, original_weight: torch.Tensor):
        """
        Verify that W_p + B_m @ A_m ≈ W (within numerical precision)
        """
        with torch.no_grad():
            reconstructed = self.W_p + self.B_m @ self.A_m
            reconstruction_error = torch.norm(reconstructed - original_weight.to(reconstructed.device))
            relative_error = reconstruction_error / torch.norm(original_weight.to(reconstructed.device))

            if relative_error > 1e-3:  # Threshold for acceptable error
                logging.warning(f"MiLoRA initialization verification failed! Relative error: {relative_error:.6f}")
            else:
                logging.info(f"  ✅ Initialization verified. Reconstruction error: {relative_error:.8f}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: output = (W_p + B_m @ A_m) @ x + bias

        This is mathematically equivalent to:
        output = W_p @ x + B_m @ A_m @ x + bias
        """
        # Principal component (frozen)
        output = F.linear(x, self.W_p.T)  # W_p @ x

        # LoRA component (trainable)
        lora_output = F.linear(x, self.A_m)  # A_m @ x
        lora_output = self.dropout(lora_output)  # Apply dropout
        lora_output = F.linear(lora_output, self.B_m.T)  # B_m @ (A_m @ x)

        output = output + lora_output

        if self.bias is not None:
            output = output + self.bias

        return output

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, dropout={self.dropout_rate}'


def apply_milora_to_model(
    model: nn.Module,
    rank: int = 8,
    target_modules: list = None,
    dropout: float = 0.05
) -> Dict[str, Any]:
    """
    Apply MiLoRA to specified modules in the model.

    Args:
        model: The model to apply MiLoRA to
        rank: LoRA rank
        target_modules: List of module names to apply MiLoRA to
        dropout: Dropout rate

    Returns:
        Dict containing information about the conversion
    """
    if target_modules is None:
        # Default target modules for LLaMA/Qwen models
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    converted_modules = []
    total_params = 0
    trainable_params = 0

    def _replace_module(parent_module, child_name, child_module):
        """Replace a module with MiLoRA version"""
        if hasattr(child_module, 'weight') and isinstance(child_module, nn.Linear):
            # Get original weight
            original_weight = child_module.weight.data.clone()

            # Create MiLoRA replacement
            milora_module = MiLoRALinear(
                in_features=child_module.in_features,
                out_features=child_module.out_features,
                rank=rank,
                dropout=dropout,
                original_weight=original_weight,
                bias=child_module.bias is not None,
                device=original_weight.device,
                dtype=original_weight.dtype
            )

            # Replace the module
            setattr(parent_module, child_name, milora_module)

            # Count parameters
            nonlocal total_params, trainable_params
            module_total = sum(p.numel() for p in milora_module.parameters())
            module_trainable = sum(p.numel() for p in milora_module.parameters() if p.requires_grad)

            total_params += module_total
            trainable_params += module_trainable

            converted_modules.append(f"{parent_module.__class__.__name__}.{child_name}")

            logging.info(f"  Converted {child_name}: {child_module.weight.shape} -> MiLoRA(rank={rank})")
            logging.info(f"    Parameters: {module_total:,} total, {module_trainable:,} trainable")

    # Traverse the model and replace target modules
    for name, module in model.named_modules():
        for child_name, child_module in module.named_children():
            if any(target in child_name for target in target_modules):
                _replace_module(module, child_name, child_module)

    logging.info(f"MiLoRA conversion completed:")
    logging.info(f"  Converted modules: {len(converted_modules)}")
    logging.info(f"  Total parameters: {total_params:,}")
    logging.info(f"  Trainable parameters: {trainable_params:,}")

    return {
        "converted_modules": converted_modules,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "rank": rank,
        "target_modules": target_modules
    }


def get_milora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract MiLoRA parameters from the model.

    Returns:
        Dictionary containing only MiLoRA parameters (A_m and B_m matrices)
    """
    milora_state_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, MiLoRALinear):
            milora_state_dict[f"{name}.A_m"] = module.A_m
            milora_state_dict[f"{name}.B_m"] = module.B_m

    return milora_state_dict


def load_milora_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    """
    Load MiLoRA parameters into the model.

    Args:
        model: Model with MiLoRA layers
        state_dict: Dictionary containing MiLoRA parameters
    """
    for name, module in model.named_modules():
        if isinstance(module, MiLoRALinear):
            if f"{name}.A_m" in state_dict:
                module.A_m.data = state_dict[f"{name}.A_m"]
            if f"{name}.B_m" in state_dict:
                module.B_m.data = state_dict[f"{name}.B_m"]

    logging.info("MiLoRA state dict loaded successfully")