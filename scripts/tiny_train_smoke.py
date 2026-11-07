"""Test unmodified official model code; this is not a formal training run."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

# Upstream tracks historical .pyc files; never replace them during a diagnostic.
sys.dont_write_bytecode = True


from roe_repro.common import UPSTREAM
checkout = UPSTREAM
report_path = Path(sys.argv[1])
report_path.parent.mkdir(parents=True,exist_ok=True)
report = {"checkout": str(checkout), "formal_training": False}
try:
    import torch
    from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
    from llava.train.roe_adapter_train import set_Adapter
    import llava.train.roe_adapter_train as adapter_module

    report["commit"] = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    report["tracked_changes"] = subprocess.check_output(
        ["git", "-C", str(checkout), "diff", "--name-only"], text=True
    ).splitlines()
    report["adapter_source"] = adapter_module.__file__
    assert Path(adapter_module.__file__).resolve().is_relative_to(checkout)
    report["versions"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "accelerate", "deepspeed")
    }
    torch.set_num_threads(4)
    # Tiny CPU smoke input only. No source functions or parameter policies patched.
    config = LlavaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64, pad_token_id=0, use_cache=False,
    )
    model = LlavaLlamaForCausalLM(config)
    set_Adapter(model, dim=8, skip_ratio=0.3)
    model.half().train()
    model.gradient_checkpointing_enable()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    masks = torch.tensor([[[1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 1, 1, 1]]], dtype=torch.float16)
    report["stages"] = []
    for stage in range(3):
        model.zero_grad(set_to_none=True)
        for layer in model.model.layers:
            layer.adapter_MHA_router.training_stage = stage
        output = model(
            inputs_embeds=model.get_input_embeddings()(tokens),
            attention_mask=torch.ones_like(tokens), labels=tokens,
            question_mask=masks, weight_mask=masks, use_cache=False,
        )
        output.loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        finite = bool(torch.isfinite(output.loss)) and all(
            bool(torch.isfinite(g).all()) for g in gradients
        )
        report["stages"].append({
            "stage": stage, "loss": float(output.loss.detach()),
            "gradient_tensors": len(gradients), "finite": finite,
        })
        assert finite, f"Non-finite loss/gradient at stage {stage}"
    report["status"] = "passed_tiny_cpu_forward_backward_only"
except Exception:
    report["status"] = "failed"
    report["traceback"] = traceback.format_exc()
finally:
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
if report["status"] == "failed":
    sys.exit(1)
