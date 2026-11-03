# AdaLLaVA-L 正式训练交接

本轮目标是启动完整训练并确认稳定；训练结束后再继续正式评测和补充分析。以最后一次状态快照为准，不能把本文件当作实时监控。

追加复核：2026-10-28用户要求严格核对settings，详见[STRICT_ALIGNMENT_REVIEW.md](STRICT_ALIGNMENT_REVIEW.md)。读取当前checkpoint确认核心配置、global128及无跳步；文档的有效gradient clipping说明已纠正为0，与官方ZeRO默认一致。运行中的训练参数未改变。

交接快照：**2026-10-28 00:54:21 CST，RUNNING，双rank均完成5/5197步**。前5步loss为3.1362、2.1741、3.3426、2.0567、3.0797，均为有限值。此时GPU0/1分别67443/67431MiB、利用率92%/93%；rank0峰值allocated62,674,448,896 bytes、reserved69,730,304,000 bytes，开头累计吞吐约3.27 samples/s。上述为开头快照，不是全程峰值/平均值。已停止主动轮询，tmux训练继续。

本地证据：[formal_status.json](evidence/formal_status.json)、[stability_checks.json](evidence/stability_checks.json)、[完整小型归档](evidence/20261028_005421.tar.gz)。归档100个来源文件逐一SHA校验通过；归档SHA-256：`613c93cec6cfdb3f54b23f19eeef07d232d5a6e025afeaae04920098f1777d3e`。对应服务器归档为`training_repro/handoff/20261028_005421.tar.gz`。不包含大权重、优化器或图像。

服务器项目根目录：`/home/SigConv/cys/AdaLLaVA`。tmux：`ada_train_formal`。正式 fresh 启动时间：2026-10-28 00:49:41 CST。

| 项目 | 实际配置 |
|---|---|
| GPU | 0、1；2 × RTX PRO 6000 Blackwell Server，单卡约96GB |
| 起点 | 原始 `liuhaotian/llava-v1.5-7b`，随机新建 Scheduler；不承接 smoke 权重 |
| 更新参数 | Backbone、projector、Scheduler；CLIP冻结；6,777,295,376可训练参数 |
| 方法 | AdaLLaVA-L；prefix16；token_selecting=none；BF16；SDPA；ZeRO-3；gradient checkpointing |
| Batch | 每卡2 × 2卡 × accumulation32 = global128 |
| 时长 | 原官方 Trainer 的1 epoch，实际总更新数5,197 |
| 优化 | AdamW；LR1e-5；cosine；warmup0.03；weight_decay0；max_length2048；seed42 |
| 保存 | 每500步，保留1个已完成状态；新 checkpoint完整写入后才轮换；终点补存并导出HF权重 |

数据已经全量校验：665,298条指令、349,034张引用图像，全部解码通过。

| 文件 | 服务器路径（项目根目录下） |
|---|---|
| 全量JSON | `training_repro/data/llava_v1_5_mix665k.json` |
| 图像统一入口 | `training_repro/data/images/` |
| 数据校验证据 | `training_repro/data/DATA_READY.json`、`instruction_manifest.json` |
| 原始底座 | `checkpoints/llava-v1.5-7b/`；离线配置视图为`training_repro/models/llava-v1.5-7b/` |
| 正式 checkpoint及最终权重 | `training_repro/runs/adallava-L-20261027/` |
| 当前日志 | `training_repro/logs/formal.log` → `formal_20261028_004941.log` |
| 每步指标 | 正式run下`metrics_rank0.jsonl`、`metrics_rank1.jsonl` |
| GPU采样 | `training_repro/logs/gpu_*.csv`，30秒间隔 |
| 命令、环境、脚本SHA | 正式run下`provenance/`，每次启动独立快照 |
| Smoke证据 | `training_repro/smoke/SMOKE_PASSED.json`、`EXPORT_RELOAD_PASSED.json`、`evidence/` |

状态检查、日志查看（先连接服务器）：

```bash
cd /home/SigConv/cys/AdaLLaVA
envs/adallava-train/bin/python training_repro/training_status.py
tail -n 40 training_repro/logs/formal.log
tmux attach -t ada_train_formal
```

只查看时建议使用前两条。tmux中按`Ctrl-b`再按`d`分离，训练继续运行。

进程已退出且存在完整 checkpoint 时恢复：

```bash
cd /home/SigConv/cys/AdaLLaVA
bash training_repro/start_formal.sh resume
```

恢复脚本选择带有`COMPLETE.json`且step匹配的最新 checkpoint，恢复模型、Adam、LR、各rank RNG及Trainer step，拒绝残缺的新目录。两张卡须重新空闲；若被他人使用，检查会拒绝启动。第500步前尚无正式 checkpoint，中断只能以`bash training_repro/start_formal.sh fresh`从原始底座重启。正式训练运行时不要重复启动或运行双卡NCCL测试。

真实 smoke 已覆盖：micro2/accum4/world2，两步后保存81,936,189,661 bytes完整ZeRO checkpoint；实际从step2恢复至3；模型FP32样本、Adam状态指纹、LR和Python/NumPy/CPU/CUDA RNG均匹配；Scheduler和backbone有有限非零梯度及更新；最终HF导出独立重载权重匹配。smoke大权重验证后已删除，保留小型证据。首次正式 checkpoint仍须等到500步，不提前宣称已经保存。

主盘软配额270GiB、硬配额300GiB；正式启动前使用约103.25GiB。实测完整checkpoint约76.31GiB；准入包含双份瞬时峰值及10GiB预留，每次保存前另检查配额。若其他文件增长导致容量不足，保存会失败并保留已有完整 checkpoint，不承诺能抵御配额之外的外部变化。

配置差异及证据边界：

- 官方脚本没有固定GPU数量。global128由上游Stage-2说明及论文约5200步曲线支持，未获作者明确确认；本次两卡将accum4调整为32。
- 保留官方旧Trainer的一epoch更新规则：5197×128=665216次样本呈现，尾部82条未进入更新。完整JSON未筛减，不能声称每一行均更新过。
- 新训练venv使用DeepSpeed0.16.9；仅训练进程预加载NCCL2.31.2处理Blackwell大消息故障；回补Accelerate对ZeRO分片的no_sync排除条件；仅自有RNG文件显式允许旧pickle恢复。实际NCCL runtime由ncclGetVersion记录为23102，Torch编译期版本另列。
- ZeRO prefetch_bucket_size=0采纳作者issue建议；中间不重复gather HF权重，完整恢复状态仍保存；最终再导出。
- Seed42提前到Scheduler构造前；官方Trainer置seed较晚。图像镜像固定revision及归档SHA，不resize、不重编码。
- Training FLOPs未可靠实测，标为未测。完整耗时/GPU-hours/峰值及吞吐待训练结束汇总，不能用短期开头推断全程结果。

原评测环境、作者checkpoint和20组正式评测结果均保留。后续P1评测Dense/官方/自训练/论文，包含本次重新列入的MMBench-EN Dev；再推进真实延迟、路由分布和Fig.4 attention。评测前还需核实judge配置。完整审计见[CONFIG_AUDIT.md](CONFIG_AUDIT.md)，修复记录见[REVIEW.md](REVIEW.md)，本地脚本副本见[reproduction/training](../../reproduction/training)。
