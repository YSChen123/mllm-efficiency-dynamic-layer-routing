# AdaLLaVA-L 完整训练配置审计

**训练完成更新（2026-10-31核验）：** 正式训练于2026-10-30 08:49 CST正常结束，完成5197更新、两rank退出0；最后ZeRO checkpoint与三份portable shards已验收，导出样本匹配，实际完整评测已启动。后文“训练尚未完成”等均为当时审计快照，不代表当前状态。结果见[评测交接](../evaluation_repro/HANDOFF_20261031.md)。

**2026-10-28 22:34 CST 严格复核：** 已重新对照当前官方脚本、论文附录图、运行入口、checkpoint内TrainingArguments/模型config/已解析DeepSpeed config及保存的engine元数据。核心训练方法与可验证超参数一致，但global128属于有证据支持的解释，硬件/部分依赖和初始化seed时机存在已知差异，不能表述为作者环境逐项完全相同。详见[STRICT_ALIGNMENT_REVIEW.md](STRICT_ALIGNMENT_REVIEW.md)。第2000步完整checkpoint记录global_samples=256000、skipped_steps=0，真实更新数与global128匹配。

审计日期：2026-10-27—28。状态：底座、完整665298指令及349034图像已全部校验；启用梯度累积的clean smoke保存、实际恢复、portable导出及独立重载均通过。正式训练于2026-10-28 00:49:41 CST重新fresh启动，实际Trainer报告5197更新、world2/micro2/accum32/global128。训练尚未完成；最终交接状态见TRAINING_HANDOFF.md。

交接验证（2026-10-28 00:54:21 CST）：两个rank均完成5步、连续5个有限loss，GPU0/1利用率92%/93%，tmux持续运行。状态/配置/日志证据已归档到本地evidence/，停止主动监控。尚未到500步首次正式checkpoint；仅smoke已真实保存与恢复，不将两者混同。

## 来源与版本

- [论文 v3](https://arxiv.org/html/2503.10905v3)，尤其 §4.1 与附录 A。
- [官方训练脚本](https://github.com/zhuoyan-xu/AdaLLaVA/blob/2a020d90973b1a387c75a042e68a1194e32757d2/scripts/train_script.sh)。AdaLLaVA commit `2a020d90973b1a387c75a042e68a1194e32757d2`。
- LLaVA commit `c121f0432da27facab705978f83c4ada465e46fd`。
- 原始底座 `liuhaotian/llava-v1.5-7b` revision `4481d270cc22fd5c4d1bb5df129622006ccd9234`；两份原始权重 shard 均通过 Hub LFS SHA-256 校验。
- CLIP 复用既有 `openai/clip-vit-large-patch14-336`，revision `ce19dc912ca5cd21c8a653c79e251e808ccabcd1`。本地底座视图只修改 config 中 CLIP 路径，权重软链接指向已校验原始文件。
- 训练 worktree：`/home/SigConv/cys/AdaLLaVA/training_repro/source/AdaLLaVA`；分支 `repro/adallava-l-20261027`。旧评测代码和官方 Ada checkpoint 保留。

## 参数对照

| 项目 | 官方可验证实现 | 本次正式配置 |
|---|---|---|
| 初始权重 | 原始 LLaVA-1.5-7B | 相同；不能使用官方 Ada checkpoint |
| Scheduler / prefix / token selecting | L / 16 / none | 相同 |
| Backbone / vision | Backbone 联合训练，CLIP 冻结 | 相同；smoke 检查梯度及 FP32 master 更新 |
| epoch / LR / weight decay | 1 / 1e-5 / 0 | 相同 |
| LR schedule / warmup | cosine / 0.03 | 相同 |
| max length / precision | 2048 / BF16 | 相同 |
| checkpointing / TF32 | True / True | 相同 |
| attention | `train_mem.py` 明确 SDPA | SDPA；README 的 FlashAttention 安装步骤不是入口实际使用的实现 |
| micro batch / accumulation | 2 / 4；脚本未固定 world size | 2 / 32，world size 2，global batch 128；见下文不确定性 |
| optimizer | `adamw_torch`，β=(0.9,0.999)，eps=1e-8 | 相同；非 fused Adam、非 LoRA |
| gradient clipping | HF字段max_grad_norm=1，但官方zero3未设gradient_clipping，DS默认0；Accelerate将裁剪交给DS | 相同：有效gradient_clipping=0，未启用；不能把HF字段值1写成实际裁剪阈值 |
| seed / data seed | HF Trainer 默认 42 / None | 42 / None；额外在模型构造前显式 set_seed(42) |
| batch shuffle | LLaVA LengthGroupedSampler，按模态长度分组并 torch.randperm | 原实现；不预先重排 JSON |
| 保存 | steps=500、save_total_limit=1 | 相同；完整 ZeRO 状态；最后导出通用权重 |
| dataloader | 4 workers、pin memory=True、drop_last=False | 相同 |
| prompt / image | v1→vicuna_v1；pad；CLIP336；feature layer -2；mlp2x_gelu | 原实现，不修改预处理 |

### Global batch 的证据边界

官方 Ada 脚本没有 GPU 数，不能从 micro=2、accum=4 唯一推出 global batch。上游 LLaVA Stage-2 公布 global batch 128；本轮直接渲染并查看论文PDF第13页，Fig. A横轴global_step曲线确实到约5,200步，665,298/128≈5,198，与此一致。因此本次选global batch128，两张干净GPU上增加accumulation至32。它是有来源支持的复现选择，**不能声称已获作者明确确认**。不同world size/累积安排可能因浮点归约、各rank RNG等产生差异，不能声称逐次梯度完全相同。在两张卡上照抄accum4会变成global16并将更新数扩大约8倍，并非更接近论文。实际Trainer报告5197更新；保留上游一epoch尾部规则。

### 随机初始化与训练行为

底座不存在 Scheduler 权重，`from_pretrained` 会报告新参数；Scheduler 由 Ada 类新建，并使用 HF `post_init`。线性层默认 normal(std=initializer_range，通常0.02)、bias=0，LayerNorm weight=1/bias=0。官方 Trainer 在模型构造后设置 seed，因此新 Scheduler 初始化未被该晚置 seed 完整控制；入口前置 seed 是本次的可重复性调整。

实际新增结构：`mlp_head: 4096→16`；latency up projection 为 LayerNorm(256)→Linear(256,4096)→GELU→Linear(4096,4096)。smoke 按 `ds_numel` 实测：总参数7,080,802,832；可训练6,777,295,376；Scheduler新增17,900,048；冻结CLIP303,507,456。底座加载仅报告8个 Scheduler tensor新初始化。

`get_random_latency` 为 `randint(16,33)/32`，即 **17 个离散均匀预算**，不是连续均匀。量化动态开关数 `k=max(floor(32*l)-16,0)`；编码值 `k/17` 经 128 对 sin/cos 后投影到 4096。Latency token 插入在当前有效序列最后一个 IGNORE_INDEX label 的后方，不能笼统解释成总是 prepend。前 16 层处理该 token 后，Scheduler 预测后 16 层的执行 mask。

训练使用温度 5、无放回 Gumbel-Softmax、hard straight-through 梯度。每个样本前16层固定执行，后16层恰选 k 层，attention/MLP 共用层开关。**训练会计算各层再施加 gate；不能拿推理 mask 减少比例推算训练 FLOPs。** 0.5 端点不激活动态层，单个端点样本 Scheduler head 无有效梯度不自动意味着故障；smoke 会检查多个实际样本的非零梯度和更新。

## 环境和必要工程差异

原评测 conda 环境复用为基础：Python3.10.22、torch2.7.1+cu128、Transformers4.37.2、Accelerate0.34.2、PEFT0.13.2。新建 `envs/adallava-train` venv（system-site-packages），仅给训练安装 DeepSpeed0.16.9 等依赖。老 DeepSpeed0.12.6 与 Blackwell 所需新 torch 的兼容性存在风险，使用新版本并通过实际 smoke 验证，不能声称环境逐项等同原文。

- [官方 Ada issue #3](https://github.com/zhuoyan-xu/AdaLLaVA/issues/3) 作者建议将 ZeRO-3 `stage3_prefetch_bucket_size` 改为 0，规避 inflight parameters 问题；本次采用独立 zero3.json，不改原上游文件。
- 中间 `stage3_gather_16bit_weights_on_model_save=false`，保留完整模型分片/FP32 master/Adam moments/LR/RNG/Trainer step，减少重复保存 HF 权重。最终导出时开启 gather；smoke 必须验证真实恢复。
- 正式首次启动在首个更新前触发 Accelerate0.34.2 / DeepSpeed0.16.9 的 no_sync 冲突。[Accelerate issue #3481](https://github.com/huggingface/accelerate/issues/3481) 解释 ZeRO2/3 的梯度分片需要reduce，不能用DDP no_sync；本次只回补该排除条件，保留DeepSpeed自己的梯度累积，不改optimizer step或有效batch。11项本地回归通过；追加32条真实样本、micro2/accum4/world2的保存/恢复smoke，避免仅用accum1覆盖不足。
- 首次双卡初始化失败；最小 NCCL 测试证实 2.26.2 在大 tensor collective 上 illegal memory access，禁用 P2P 仍失败。[NVIDIA issue #2418](https://github.com/NVIDIA/nccl/issues/2418) 记录同一 Blackwell/NCCL2.26.2 故障，并报告 2.31.2 解决。本次仅训练进程 LD_PRELOAD 独立安装的 NCCL2.31.2；验证结果另存日志，不调整系统驱动或其他任务。

## 完整数据与空间

官方 JSON revision `9d451dc7629cfe0469f6ae4432b765cd603d5fcb`；大小 1,029,887,963 bytes；SHA-256 `ce959ce6e23073ee1cd1a8a2ef1c633768c10d4174327b8b2dc7113b91af6cf8`。

| source | 指令样本 | 唯一引用图像 |
|---|---:|---:|
| COCO | 364100 | 88524 |
| VG | 86417 | 86417 |
| GQA | 72140 | 72140 |
| OCR-VQA | 80000 | 80000 |
| TextVQA | 21953 | 21953 |
| text only | 40688 | 0 |
| 合计 | 665298 | 349034 |

为处理原 VG 源不可达，使用[研究项目原始图像归档镜像](https://huggingface.co/datasets/kaiyuyue/llava-1.5-665k-instructions) revision `f321f47bf67bf43969756ef22650e04ab367bcaa`，不使用它的 shuffled JSON。所有归档固定 LFS SHA-256；仅提取原官方 JSON 引用的全部图像，直接复制字节、不 resize、不重编码；全量 PIL verify+RGB decode 后才写 DATA_READY。

COCO 路径别名 `coco/images/train2017`→`coco/train2017`，OCR `ocr-vqa`→`ocr_vqa`，用于与官方 JSON 匹配。其余保持原布局。COCO 放主目录，其余放 `/home/SigConv/backup2/adallava-training-20261027/images`，通过软链接统一访问。仅在单个归档完成校验、所有引用图像解码通过后删除本次下载的归档，保留来源/哈希清单。

主盘 soft/hard=270/300 GiB，审计初始使用约75.2 GiB；backup2 soft/hard=150/200 GiB，初始使用约85.1 GiB；runs/backup1 共用另一盘且已超过软配额，未将它作为备用容量。Checkpoint 峰值必须包括前一个完整 checkpoint 与正在写入的新 checkpoint，不能只算单份权重。实际 checkpoint bytes 将在 smoke 后更新，并在每次保存前检查软配额和 reserve。

## 实测证据

2026-10-28 更新：NCCL2.31.2 已通过16/4096/16777216元素 BF16 all-reduce一致性检查；wheel SHA-256 `f9b1dc3c2a7e20176054144ebb3b32fea83b40402ee5d7ac7045cd11ecc956c0`。Torch2.7默认 torch.load 限制与旧 Trainer RNG pickle冲突已复现，使用仅针对自有RNG的显式 weights_only=False 精确恢复Python/NumPy/CPU/CUDA状态，CPU回归已通过。

NCCL版本测量修正：torch.cuda.nccl.version()在本build返回编译期2.26.2，不能作为预加载库的runtime版本。入口改用预加载lib的ncclGetVersion读取真实runtime整数，同时单独记录torch_build_nccl_version；之前首次失败formal ENTRY中旧命名字段不用于判断真实运行库。最小NCCL日志已明确运行库2.31.2。

最终clean Smoke实测通过：micro2/accum4/world2，两rank从step2恢复至3；FP32模型样本、Adam moments/step指纹、Trainer持有的LR scheduler、Python/NumPy/CPU/CUDA RNG均一致；更新后仍有非零梯度及参数变化。完整checkpoint为81,936,189,661 bytes（约76.31GiB），中间保存不额外重复HF权重；双份瞬时约152.62GiB，正式启动另留10GiB。gather导出3份safetensors后独立from_pretrained无missing/mismatched trainable keys，Scheduler/backbone参数与导出前权重一致；391个冻结CLIP keys因官方vision tower lazy-load被忽略，配置仍指向已复用的同一CLIP。已存EXPORT_RELOAD_PASSED.json和SMOKE_PASSED.json。本轮无审计sidecar修补；此前accum1调试记录另存。

调试权重和大optimizer分片已在验证通过后删除，仅保留全套元数据、rank proof、RNG、JSONL、失败/成功日志和文件大小清单；不会把smoke权重作为正式起点。审查和运行问题处理详见REVIEW.md。

续训后的旧HF汇总train_loss会按累计global step归一化，即使本次只执行剩余步骤；正式loss曲线和跨resume平均值必须从逐step LOG合并，不能直接把该汇总字段作为全程平均。

**上游epoch尾部行为：** 原Trainer4.37.2一epoch按floor(len(dataloader)/accum)决定更新数。本配置预计5197更新、665216样本呈现，尾部82条不进入更新（分布式loader自身还可能补齐极少样本）。本次保留全665298 JSON与349034图像，遵循官方依赖与原一epoch训练循环；不能报告为“每一行都产生了梯度更新”。不改旧HF梯度累积循环或偷偷进入第二epoch制造整除。实际step/epoch/呈现量在正式日志核实。

- 底座各文件 SHA：`training_repro/logs/download_inputs.log`。
- 原始环境审计：`training_repro/audit/environment_and_sources.txt`。
- 默认值/保存实现：`training_repro/audit/runtime_defaults_and_checkpoint.txt`。
- NCCL 最小复现：`training_repro/logs/nccl_default.log`、`nccl_no_p2p.log`；新版测试 `nccl_2312.log`。
- 数据完整性：`training_repro/data/instruction_manifest.json`、每源 `*_verified.json`、最终 `DATA_READY.json`。
- 训练 FLOPs 暂未实测；HF 默认估计不会直接报告成动态路径的真实训练 FLOPs。原文未报告的 training time/GPU-hours/peak memory 标记 Not Reported。
