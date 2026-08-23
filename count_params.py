"""
Count trainable parameters for each method (DC-Finetune, Vanilla MoE, MixLoRA, CulDPD)
under unified LoRA rank=16, alpha=32 configuration, on Llama-3.1-8B and Qwen2.5-7B backbones.

Architecture facts (verified from src/llamafactory/model/):
- CulDPD: base LoRA on q/k/v/o + 4 FFN experts (full nn.Linear gate/up/down, NOT LoRA)
          + 1 shared FFN expert + router + fusion gate.
          MoE experts are full FFN modules, not LoRA-decomposed.
- Vanilla MoE: 4 LoRA experts on q/k/v/o + router (no shared, no masking, no CSL).
- MixLoRA: per ft_mixlora.py defaults — LoRA experts on FFN (gate/up/down) +
           shared LoRA adapters on attention (q/v). Default num_experts=6, rank=64 in code,
           but for fair comparison we use rank=16, 4 experts to match CulDPD.
- DC-Finetune: single LoRA on q/k/v/o.

Backbone hidden dims:
- Llama-3.1-8B-Instruct: hidden=4096, intermediate=14336, layers=32, attn heads=32
- Qwen2.5-7B-Instruct:   hidden=3584, intermediate=18944, layers=28, attn heads=28
"""

# Backbone configs (hardcoded to avoid network access)
BACKBONES = {
    "Llama-3.1-8B-Instruct": {
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
    },
    "Qwen2.5-7B-Instruct": {
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_hidden_layers": 28,
    },
}

# ---- LoRA config (per paper) ----
# CulDPD / Vanilla MoE / DC-Finetune: LoRA rank=16, alpha=32 (paper Section 4.2 / Appendix D)
# CulDPD and Vanilla MoE use LoRA experts (per paper Section 3.2: "A and B are low-rank
# decomposition matrices of LoRA"), injected per transformer layer.
LORA_RANK = 16
LORA_ALPHA = 32
NUM_EXPERTS = 4

# MixLoRA uses its own config (per run_ft_mixlora.sh): rank=32, alpha=16, 4 experts
MIXLORA_RANK = 32
MIXLORA_ALPHA = 16
MIXLORA_EXPERTS = 4

# Module sets
ATTN_QKVO = ["q_proj", "k_proj", "v_proj", "o_proj"]
ATTN_QV   = ["q_proj", "v_proj"]
FFN_GUD   = ["gate_proj", "up_proj", "down_proj"]


def lora_params(d_in, d_out, r):
    """Trainable params of one LoRA adapter: A(d_in, r) + B(r, d_out)."""
    return d_in * r + r * d_out


def attn_layer_params(hidden, r, modules):
    """LoRA params on attention modules of one transformer layer."""
    # q/k/v/o all map hidden -> hidden; modules is a list of names
    return len(modules) * lora_params(hidden, hidden, r)


def ffn_layer_params(hidden, inter, r, modules):
    """LoRA params on FFN modules of one transformer layer."""
    total = 0
    if "gate_proj" in modules: total += lora_params(hidden, inter, r)
    if "up_proj"   in modules: total += lora_params(hidden, inter, r)
    if "down_proj" in modules: total += lora_params(inter, hidden, r)
    return total


def full_ffn_params(hidden, inter):
    """Params of one full FFN module (gate + up + down, with bias)."""
    # gate: hidden->inter, up: hidden->inter, down: inter->hidden
    return (hidden * inter + inter) * 2 + (inter * hidden + hidden)


def router_params(hidden, n_experts):
    """Router linear: hidden -> n_experts, with bias."""
    return hidden * n_experts + n_experts


def hidden_proj_params(d_in, d_out):
    """Projection layer backbone_hidden -> moe_hidden (and back), with bias."""
    return d_in * d_out + d_out


def fusion_gate_params(moe_hidden):
    """Fusion gate: nn.Linear(moe_hidden, 2), with bias.
    Note: in code gate_network = nn.Linear(moe_hidden_dim, 2), operates on moe_hidden-dim input."""
    return moe_hidden * 2 + 2


def count_method(model_name, method):
    cfg = BACKBONES[model_name]
    H = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    L = cfg["num_hidden_layers"]
    r = LORA_RANK
    n = NUM_EXPERTS

    if method == "DC-Finetune":
        # single LoRA on q/k/v/o, per-layer
        per_layer = attn_layer_params(H, r, ATTN_QKVO)
        return per_layer * L

    if method == "Vanilla MoE":
        # Vanilla MoE (Shen et al. 2023, per-layer MoE-LoRA):
        # 4 LoRA experts on FFN (gate/up/down) + router, injected per transformer layer.
        # No shared expert, no masking, no fusion gate.
        per_layer_experts = n * ffn_layer_params(H, I, r, FFN_GUD)
        router = router_params(H, n)
        return (per_layer_experts + router) * L

    if method == "MixLoRA":
        # MixLoRA (Li et al. 2024b): per-layer LoRA experts on FFN (gate/up/down)
        # + shared LoRA on attn (q/v) + router. Config from run_ft_mixlora.sh.
        r_m = MIXLORA_RANK
        n_m = MIXLORA_EXPERTS
        per_layer_experts = n_m * ffn_layer_params(H, I, r_m, FFN_GUD)
        shared_attn = attn_layer_params(H, r_m, ATTN_QV)
        router = router_params(H, n_m)
        return (per_layer_experts + shared_attn + router) * L

    if method == "CulDPD":
        # CulDPD (per paper Section 3.2): per-layer design with two pathways.
        # Culture-shared pathway: 1 shared LoRA expert on attention q/k/v/o.
        # Culture-specific pathway: 4 LoRA experts on FFN (gate/up/down) + router.
        # Plus fusion gate (per layer) for combining shared and specific outputs.
        shared_lora = attn_layer_params(H, r, ATTN_QKVO)
        spec_experts = n * ffn_layer_params(H, I, r, FFN_GUD)
        router = router_params(H, n)
        gate = fusion_gate_params(H)
        return (shared_lora + spec_experts + router + gate) * L

    raise ValueError(method)


def fmt(n):
    return f"{n:,} ({n/1e6:.2f}M)"


def main():
    models = {
        "Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
        "Qwen2.5-7B-Instruct":   "Qwen2.5-7B-Instruct",
    }
    methods = ["DC-Finetune", "Vanilla MoE", "MixLoRA", "CulDPD"]

    print(f"Config: LoRA rank={LORA_RANK}, alpha={LORA_ALPHA}, experts={NUM_EXPERTS}")
    print(f"{'Method':<16} | {'Llama-3.1-8B':>22} | {'Qwen2.5-7B':>22}")
    print("-" * 70)
    results = {}
    for m in methods:
        row = []
        for disp, mid in models.items():
            n = count_method(mid, m)
            row.append(n)
        results[m] = row
        print(f"{m:<16} | {fmt(row[0]):>22} | {fmt(row[1]):>22}")

    # also print breakdown for CulDPD on Llama
    print("\n--- CulDPD breakdown (Llama-3.1-8B, per layer × 32 layers) ---")
    cfg = BACKBONES["Llama-3.1-8B-Instruct"]
    H, I, L = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
    r, n = LORA_RANK, NUM_EXPERTS
    shared_lora = attn_layer_params(H, r, ATTN_QKVO)
    spec_expert = ffn_layer_params(H, I, r, FFN_GUD)
    spec_experts = n * spec_expert
    router = router_params(H, n)
    gate = fusion_gate_params(H)
    per_layer = shared_lora + spec_experts + router + gate
    print(f"  shared LoRA (q/k/v/o):           {shared_lora:>12,}")
    print(f"  1 specific LoRA expert (g/u/d):  {spec_expert:>12,}")
    print(f"  4 specific LoRA experts:         {spec_experts:>12,}")
    print(f"  router:                          {router:>12,}")
    print(f"  fusion gate:                     {gate:>12,}")
    print(f"  per-layer total:                 {per_layer:>12,}")
    print(f"  × {L} layers = {per_layer * L:,}")

    # write JSON for downstream use
    import json
    out = {
        "config": {"lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA, "num_experts": NUM_EXPERTS},
        "results": {m: {disp: n for disp, n in zip(models.keys(), results[m])} for m in methods},
    }
    with open("/tmp/param_counts.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nJSON written to /tmp/param_counts.json")


if __name__ == "__main__":
    main()
