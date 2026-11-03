# AdaLLaVA-L 当前训练与官方设置严格复核

复核时间：2026-10-28 22:27—22:34 CST。结论：**核心方法和可验证训练超参数对齐；存在明确记录的batch实现、初始化、环境及保存工程差异，不能称为作者运行环境逐项完全相同。当前没有证据支持中断训练或改成两卡accum4。** 本轮仅做只读检查、补充证据和纠正文档，没有改正在运行的训练参数、模型、数据或优化器状态。

## 核对了哪些实际证据

- 官方AdaLLaVA commit `2a020d90973b1a387c75a042e68a1194e32757d2`、LLaVA commit `c121f0432da27facab705978f83c4ada465e46fd`；两个工作目录`git diff HEAD --stat`均为空，源文件未改。
- 当前官方main训练脚本与固定commit脚本的关键参数一致；逐项对照运行ENTRY中实际argv。37个官方选项中29个值相同（包括数值0与0.），其余8项为本地路径、run名称、独立ZeRO配置及accumulation差异，逐项记录在[CLI对照](evidence/strict_cli_comparison.json)。新增显式seed42与原Trainer默认seed42一致，但提前设置时机另见下文。
- 直接读取本任务已保存的`checkpoint-1500/training_args.bin`，同时读取模型config、Trainer state、LR state及已解析ZeRO配置。详见[实际设置](evidence/strict_actual_settings_20261028_222704.json)。该args中`max_steps=-1`表示未手动覆盖epoch训练，不代表没有训练；实际Trainer state的max_steps为5197。
- 以CPU mmap仅读取`checkpoint-2000`保存的DeepSpeed metadata：global_steps=2000，global_samples=256000，skipped_steps=0，dp_world_size=2。即实际global batch128，且到此没有溢出跳过optimizer更新。详见[engine元数据](evidence/strict_engine_metadata_20261028_223404.json)。不加载大权重到GPU，不干扰训练。
- 当前本地交付的6个关键入口/配置/兼容文件SHA与服务器读取结果一致；完整启动快照另存run/provenance。
- 直接查看论文PDF第13页附录训练图，曲线终点约5200步；[渲染页](evidence/paper_training_curve.png)保留。

## 已确认对齐的训练设置

| 项目 | 官方可验证设置 | 当前实际生效 |
|---|---|---|
| 起点 | 原始liuhaotian/llava-v1.5-7b，随机新Scheduler | 相同；不是作者Ada权重，也不承接smoke |
| 架构 | L、prefix16、token_selecting=none | 相同；32层、4096 hidden；Scheduler默认tau5 |
| 训练范围 | LLM/projector/Scheduler联合，vision冻结 | 相同；可训练6,777,295,376，新Scheduler17,900,048 |
| LoRA/量化/adapter-only | 不启用 | lora_enable=False，bits=16，tune_mm_mlp_adapter=False，freeze_mm_mlp_adapter=False |
| 训练集 | llava_v1_5_mix665k.json完整指令及引用图像 | 官方JSON哈希一致，665298行、349034引用图像全部可读，无手动筛样本 |
| Epoch/长度 | 1 / 2048 | 相同；Trainer总5197更新 |
| Optimizer | adamw_torch，beta0.9/0.999，eps1e-8 | 相同；没有换fused Adam或8bit Adam |
| LR与分组 | 1e-5，backbone/Scheduler同率 | 相同；mm_projector_lr=None，不另设projector学习率；保存LR state的两组base_lrs均1e-5 |
| Schedule | cosine，warmup_ratio0.03，weight_decay0 | 相同；warmup_steps显式默认0，按ratio推导156步 |
| Batch | micro2，accum4，卡数未给出 | micro2，world2，accum32，global128；差异说明见下 |
| 精度/attention | BF16、TF32、train_mem.py明确SDPA | 相同；checkpoint的torch_dtype=bfloat16、运行TRAIN_BEGIN记录sdpa |
| Checkpointing | gradient_checkpointing=True | 相同；未更改checkpoint重算机制 |
| 图像与prompt | CLIP-L/14-336、layer-2、patch、mlp2x_gelu、pad、v1→vicuna_v1 | 相同，源码未改；tokenizer右padding、max2048 |
| 数据组织 | modality length grouping、lazy preprocess、4workers | 相同；pin_memory=True、drop_last=False、data_seed=None |
| Budget/路由 | 官方L源码的0.5—1.0离散预算、Gumbel无放回、hard straight-through | 原源码未改；17个预算为16/32…32/32，前16层固定、后16层按k选择 |
| 保存频率 | 每500步、save_total_limit1 | 相同；新checkpoint完整后再轮换，最终补存并导出 |

注意模型config里的`max_position_embeddings=4096`和生成配置`max_length=4096`不能作为训练长度；实际TrainingArguments.model_max_length和config.tokenizer_model_max_length都是2048。

## 最重要的差异：global batch的证据与界限

1. [Ada官方脚本](https://github.com/zhuoyan-xu/AdaLLaVA/blob/2a020d90973b1a387c75a042e68a1194e32757d2/scripts/train_script.sh)只写micro2、accum4，没有world size。不能据此确定作者global batch，更不能直接说作者用了16张卡。
2. [论文§4.1](https://arxiv.org/html/2503.10905v3#S4.SS1)说其余设置遵循LLaVA Stage-2；[上游LLaVA说明](https://github.com/haotian-liu/LLaVA#train)给出Stage-2 global128，并明确卡数变化时调整micro/accum以维持global batch。上游表格展示13B配置，不能单靠该表格证明Ada7B实际运行细节。
3. 论文自身AdaLLaVA-L-7B附录Fig.A曲线在约5200步结束。完整665298条、1epoch、global128对应约5198步，与本次旧Trainer实际5197步相符。这是独立支持，仍是图像与协议推断，不是作者提供的原始launch log。
4. 本次2×2×32=128；两卡照抄accum4会变成global16，更新约41581步，warmup长度和LR相对于样本数的进程也随之变化。因此本次保留128更有依据。
5. 相同global batch不保证逐次更新完全相同：rank数、预算/Gumbel随机流、浮点求和、不同硬件kernel均可能改变训练轨迹。报告应写“global128的有据复现”，不能写“已确认作者相同GPU/累积布局”。

## 梯度累积与裁剪的实际语义

本次特别检查了安装环境源码，见[Trainer/Accelerate/optimizer源码](evidence/strict_gradient_optimizer_sources.txt)和[有效梯度路径源码](evidence/strict_effective_gradient_sources.txt)。

- Trainer把原loss传入Accelerator.backward；DeepSpeed路径中Accelerate不额外除以累积数；DS在backward prologue除以32。Trainer返回的detach loss再除32只用于日志累加，不参与反向。未发现重复除以32或漏除的路径。
- Accelerate的DS wrapper每micro调用engine.step，由DS检查累计边界；engine checkpoint确认2000更新/256000 samples，未多走optimizer step。no_sync补丁仅对ZeRO分片绕开不兼容DDP上下文，保留DS自己的归约及累积边界。
- **纠正文档：HF的max_grad_norm=1不等于实际启用了阈值1裁剪。** HF配置填充只替换已有`auto`字段，不会补写缺失字段；官方zero3及本次zero3都没有gradient_clipping，Accelerate在DS模式下不执行外部clip，DS默认为0（关闭）。[官方DS0.12.6常数](https://github.com/microsoft/DeepSpeed/blob/v0.12.6/deepspeed/runtime/constants.py)与当前DS0.16.9同为0。这是先前审计描述不严谨，本次已修正；运行中行为一直与可验证官方配置一致。为了“更严格”额外启用clip1反而会引入差异。

## 其余公开记录的差异

| 差异 | 当前处理及影响边界 |
|---|---|
| 原始seed时机 | 本次在Scheduler构造前set_seed42，官方只在模型构造后的Trainer置seed。随机初始化分布相同，具体初值不保证相同；作者原始Scheduler初值未公开，不能靠现在删seed恢复 |
| 软件版本 | LLaVA依赖torch2.1.2/torchvision0.16.2/Accelerate0.21.0/DS0.12.6；当前torch2.7.1+cu128/torchvision0.22.1/Accelerate0.34.2/DS0.16.9；Transformers4.37.2相同。Blackwell兼容需求及旧评测环境复用已有记录；不能声称数值轨迹完全相同 |
| NCCL | 训练进程单独预加载2.31.2，修复Blackwell大消息collective错误；不改系统或其他任务 |
| ZeRO prefetch | 设0，采纳[作者issue #3建议](https://github.com/zhuoyan-xu/AdaLLaVA/issues/3)；不改路由/损失函数 |
| checkpoint格式 | 中间不重复gather HF权重，保存完整ZeRO模型/FP32 master/Adam/LR/RNG；最终gather导出。实际保存恢复已测，不更改训练数学 |
| 输入镜像 | 官方JSON直接固定revision与SHA；图像使用固定revision/归档SHA镜像，全量引用图像可解码，提取过程不重编码。6张COCO与原始官网下载字节哈希一致；其余图像未逐一与不可达原服务器做跨源哈希，因此不应称为全量跨源逐字节认证 |
| epoch尾部 | 沿用官方旧Trainer循环，5197×128=665216次呈现、尾部82条未进入更新；不是手工缩减JSON，不额外补第二epoch制造新差异 |

决策：继续当前训练。发现并修正的是审计文档的裁剪说明；未发现需要回滚当前权重的训练方法错误。若将来拿到作者原始world size、Scheduler seed、launch/env日志，才能进一步判定剩余不可验证项。保持当前run配置固定，避免在训练中途换batch或学习率导致混合协议。
