# MLLM Efficiency: Dynamic Layer Routing

当前分支为 **PAR**：[复现代码与新服务器使用说明](PAR/README.md)。覆盖Layer-only和Joint的POPE/SEED-I，明确记录未公开参数、公开代码冲突及残余数值差距。

Private research reproduction archive. Each paper is kept on a separate branch.

| Paper | Branch | Status |
|---|---|---|
| PAR (ICCV 2025), LLaVA-v1.5-7B | [`PAR`](../../tree/PAR) | Portable public-settings reconstruction; Layer-only close, Joint SEED close, POPE gap disclosed |
| AdaLLaVA (ICCV 2025), AdaLLaVA-L / LLaVA-1.5-7B | [`adallava`](../../tree/adallava) | Full training completed; four benchmarks evaluated; known MME gap documented |

```bash
git clone --branch adallava --single-branch https://github.com/YSChen123/mllm-efficiency-dynamic-layer-routing.git
```

Branches preserve code, fixed source/dependency versions, settings and measured results. Download datasets and pretrained base/vision weights separately. Training checkpoints and credentials are not versioned. Future papers start from `main` and get their own branch.
