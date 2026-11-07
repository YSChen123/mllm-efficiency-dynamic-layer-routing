# RoE-30% release 复现与动态跳层验证

归档 ICLR 2025 **Routing Experts: Learning to Route Dynamic Experts in Existing Multi-modal Large Language Models** 的 LLaVA-1.5-7B 实验，训练及评测完成于 **2026-11-07**。

**复现对象是固定版本的官方 release，不宣称论文全部设置已严格对齐。** 训练源码及超参数保留原样；外部动态入口恢复官方硬路由，并包含明确记录的缓存兼容和数值恢复。没有调整测试阈值、固定跳过层数或强制首层执行。

- [论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/49a4829a2ccaaa4706cc82e68c39a9c6-Paper-Conference.pdf)、[官方仓库](https://github.com/DoubtedSteam/RoE)
- 固定 commit：`4f529e468d396246dd96f44fb7e13bd3cba61261`，以 Git 子模块保存在 `upstream/RoE`。
- 本分支只保存代码、配置、版本/哈希和聚合结果，**不包含环境、数据标注、图片、权重或凭据**。

## 本次结果

Full 使用同一份完成 RoE 训练的 checkpoint，执行全部 Transformer 层；不是未经 RoE 训练的原始 LLaVA 基线。

| 数据集 | 样本数 | Full 准确率 | 动态准确率 | 动态实际跳过率 | FP32 恢复数 |
|---|---:|---:|---:|---:|---:|
| ScienceQA-IMG | 2,017 | 69.3604% | **69.7571%** | **13.5768%** | 1 |
| GQA | 12,578 | 61.1067% | **60.8046%** | **13.7977%** | 8 |
| SEED image | 14,233 | 65.2006% | **65.1093%** | **33.9343%** | 0 |

三项全量完成，无空答案、未跳过异常样本；路由记录覆盖每个样本一次，每个样本统计 32 次层决策。SEED 只评图像，排除 3,757 条视频。MMBench 未运行；复现本表不需要 GPT 评分密钥。

与论文表 1 的 RoE-LLaVA-30% 对照：

| 数据集 | 论文准确率 | 本次准确率 | 论文实际跳过率 | 本次实际跳过率 |
|---|---:|---:|---:|---:|
| ScienceQA-IMG | 68.40% | 69.7571% | 23.03% | 13.5768% |
| GQA | 61.40% | 60.8046% | 8.81% | 13.7977% |
| SEED（口径未对齐） | 58.20% | 65.1093%（image） | 30.43% | 33.9343%（image） |

**准确率接近，跳过率未整体复现论文。** ScienceQA 少跳约 9.45 个百分点，GQA 多跳约 4.99 个百分点；SEED 不作直接横比。没有统一数值策略和观察开销的 Full 速度对照，不宣称复现论文加速比。机器可读成绩见 [results/measured_20261107.json](results/measured_20261107.json)。

## 设置与差异

| 项目 | 本分支的实际行为 |
|---|---|
| 目标跳过率 | `--skip_ratio 0.3`；实际路径由模型学习决定，不保证每项任务等于 30% |
| 训练数据 | 原 665,298 条，经 release 的 `idx % 10 < 6` 得到 **399,180 条（60%）**；文件名仍含 `_50` |
| 三阶段 | 原 Adapter warmup、Router warmup、联合 tuning、取整及温度/偏置日程不改 |
| 损失 | 原样 `Lt + exp(-5 * Lt).detach() * Ls`；论文文字为 `Lt + 0.5 * exp(-Lt) * Ls` |
| 学习率 | 名义参数 backbone `2e-6`、RoE `4e-4`、通用 LR `1e-5`；保留原参数组及按编号冻结行为，不声称每组每步的有效 LR 都等于名义值 |
| 其他训练参数 | Adapter 1024，BF16，ZeRO-3，长度 2048，1 epoch；两卡各 batch 4、累积 4，全局 batch 32 |
| early stopping | release 未配置验证驱动的 early-stopping callback；本次跑满 **12,474 步**，不称为实现论文描述的 early stopping |
| 保存 | 原 `save_steps=50000`、`save_total_limit=4`；本次不足 50000 步，只写一份最终模型，不是定期完整优化器 checkpoint 策略 |
| 训练环境 | Python 3.10，Torch 2.7.1+cu128，Transformers 4.31.0，Accelerate 0.34.2，DeepSpeed 0.9.5；作者 Torch 2.4.1 无法运行本次 Blackwell GPU |
| 评测环境 | 独立 Python 3.10，Torch 2.7.1+cu128，Transformers 4.39.2；每任务单卡，batch 1、beam 1 |
| 动态入口 | release 的 lmms-eval 硬路由被注释；外部封装复用官方 `RoE/llava/train/roe_adapter_eval.py` 的硬路由函数 |
| 缓存兼容 | 跳过层返回带长度的零步长缓存视图，不计算该层 Attention/KV，不修改路由判定 |
| 数值兼容 | ScienceQA/GQA 的非有限 FP16 请求以相同权重 FP32 重算一次，之后恢复参数 dtype、buffers、缓存长度及默认 dtype，重算不重复计数 |
| 数据和路径 | 固定公开数据 revision，鉴权设置为 `token: false`；仅图像筛选、任务别名及 checkpoint 视觉塔路径迁移另行记录，不改提示词/计分算法 |

应描述为 **“官方 release 训练与动态路由验证，附环境和推理兼容差异”**，不能称为“论文所有 setting 完全一致”或“官方评测脚本逐字不变”。

## 1. 获取代码

运行平台：**Linux + NVIDIA GPU + Python 3.10**。本次使用两张 96GB Blackwell GPU，训练约 81 小时。SEED 原版上下文准备占较多主机内存，建议至少 96–128GB RAM；更小显存需要另测。

```bash
git clone --branch ROE --single-branch https://github.com/YSChen123/mllm-efficiency-dynamic-layer-routing.git
cd mllm-efficiency-dynamic-layer-routing
python3 scripts/bootstrap_upstream.py
cp config.example.json config.local.json
```

Linux 也可使用 `git submodule update --init`。上游有带冒号的历史日志，Windows 不支持直接完整 checkout；bootstrap 排除这些日志及缓存文件，不修改算法源码。训练/评测仍要求 Linux。

编辑 `config.local.json`。相对路径以**配置文件所在目录**为基准；runtime、模型、图片可分别放在有配额的磁盘上。个人配置不入库。

## 2. 创建环境

不要直接安装作者完整 freeze：其中含作者机器的私有目录及大量未用于本实验的包。这里分别固定实际使用的训练/评测核心依赖；不声称与作者完整 freeze 一致。

```bash
# 先确保有 Python 3.10 及支持 CUDA 12.8 的 NVIDIA 驱动。
bash scripts/setup_env.sh train
bash scripts/setup_env.sh eval
source .venvs/train/bin/activate
python -m roe_repro.check --phase train
deactivate
source .venvs/eval/bin/activate
python -m roe_repro.check --phase eval --tests
```

用 `ROE_PYTHON=/path/to/python3.10` 指定解释器，脚本第二个参数可指定新 venv 目录。脚本拒绝覆盖已有环境，不改其他项目环境。

## 3. 准备模型和数据

激活 eval 环境，明确调用下载命令：

```bash
source .venvs/eval/bin/activate
python -m roe_repro.assets --config config.local.json models
python -m roe_repro.assets --config config.local.json verify-models
python -m roe_repro.assets --config config.local.json annotation
```

固定模型版本和逐文件哈希见 [results/assets.json](results/assets.json)。下载同时准备专用 Hub cache：原版训练初始化仍用基础配置中的 CLIP Hub ID，只有一个本地 CLIP 目录不足以离线运行。专用 cache 的 main ref 绑定到固定 commit；已有不同 ref 时拒绝覆盖。

标注下载后必须匹配 [results/training_data_hashes.json](results/training_data_hashes.json)。URL 使用上游 main，但字节 SHA256 固定，上游内容变化时会拒绝静默继续。

另行下载并解压 LLaVA 混合集图片，放在 `images_dir`：

```text
images/
├── coco/train2017/
├── gqa/images/
├── ocr_vqa/images/       # LLaVA 使用 .jpg
├── textvqa/train_images/
└── vg/
    ├── VG_100K/
    └── VG_100K_2/
```

来源：[COCO train2017](http://images.cocodataset.org/zips/train2017.zip)、[GQA](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)、[OCR-VQA](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_)、[TextVQA](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)、[VG part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip)、[VG part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)。另见 [作者的数据文档](upstream/RoE/readme.md)。

可逐个下载、校验后删除压缩包，但混合集训练前须保留全部引用图片，不能分别训练各来源数据集来代替本实验。

```bash
python -m roe_repro.prepare_data --config config.local.json --check-images
python -m roe_repro.assets --config config.local.json eval-data
python -m roe_repro.check_tasks --config config.local.json
```

子集生成精确复现 release 的选择与序列化，校验源/子集 SHA256，并检查图片路径。`check_tasks` 对比固定版本别名与原版任务的全部文本上下文和目标答案；SEED 仅取 image，不解码全量图片做检查。

## 4. 训练

```bash
source .venvs/train/bin/activate
python -m roe_repro.train --config config.local.json --dry-run
cat runtime/roe/commands/finetune_RoE.paths_only.sh
mkdir -p runtime/roe/logs
# 建议在 tmux 内执行。自定义 runtime 时同步修改下面的日志路径。
python -m roe_repro.train --config config.local.json > runtime/roe/logs/train.log 2>&1
```

`--dry-run` 只生成命令。正式入口检查两张物理 GPU 空闲，拒绝已有 checkpoint 目录；只替换原脚本的基础模型、视觉塔、子集、图片根目录和输出路径，保留原 19999 端口及其他超参数。不自动恢复训练、跳样本、改学习率或裁剪策略。

```bash
tail -n 30 runtime/roe/logs/train.log
# 可选：tiny CPU 三阶段前后向检查，不等同于 7B 正式训练。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/upstream/RoE/RoE:$PWD" python scripts/tiny_train_smoke.py runtime/roe/tiny_train_smoke.json
```

不同 GPU 数量、batch、样本或参数属于新实验条件，不能继续声称复现本次 12,474 步运行。

## 5. 评测最终权重

激活 eval 环境，配置 checkpoint。每项单卡；GPU 编号按空闲情况修改，以下顺序执行会逐项完成。短测成功后自动进入全量：

```bash
source .venvs/eval/bin/activate
python -m roe_repro.evaluate --task scienceqa_img --mode dynamic --gpu 0 --smoke-first
python -m roe_repro.evaluate --task gqa --mode dynamic --gpu 0 --smoke-first
python -m roe_repro.evaluate --task seedbench_image --mode dynamic-fp16 --gpu 0 --smoke-first
```

这对应本次 ScienceQA/GQA 的选择性 FP32 恢复和 SEED 的原样 FP16 路由。`dynamic-fp16` 遇非有限值就停止；若需恢复，保留失败目录，明确改用 `dynamic` 并记录精度差异。本次 SEED 无重试。

Full 对照：

```bash
python -m roe_repro.evaluate --task scienceqa_img --mode full --gpu 0 --sync-debug
python -m roe_repro.evaluate --task gqa --mode full --gpu 0 --sync-debug
python -m roe_repro.evaluate --task seedbench_image --mode full --gpu 0
```

`--sync-debug` 对应本次 ScienceQA/GQA Full 保留的同步开关，会拖慢速度，不是官方原脚本要求。公平测速应统一开关、观察开销和数值策略。

```bash
python -m roe_repro.status --config config.local.json
tail -c 2000 runtime/roe/evaluation/reproduction/dynamic/gqa/formal/run.log
```

只有退出/结果文件/有效样本数/limit/路由计数检查通过后才记 completed，冒烟与正式结果分开。已有目录不会覆盖；重复验证请指定新的 `--run-name`。

## 目录与适用范围

2026-11-08 发布前在本次实际运行环境重新验证：9 项回归测试通过，训练导入和路径渲染通过，tiny 模型三阶段均有有限 loss/梯度；三项全部文本上下文和目标答案与固定版本原任务一致。真实 7B 权重搬迁视觉塔路径后，ScienceQA 16 条、GQA/SEED 各 8 条短测完成生成、计分、保存。399,180 条训练子集的 SHA256 匹配，240,041 个引用图片路径全部存在。离线 CLIP 标准缓存检查通过，测试后官方 tracked 源码保持干净。详见 [验证记录](results/validation_20261108.json)。这些检查没有重新训练 81 小时，也没有在全新机器重装全部环境，不能代替新硬件的短测。

`upstream/` 固定官方来源；`roe_repro/` 提供入口；`scripts/` 准备环境及源码；`requirements/` 固定核心依赖；`tests/` 验证路径渲染、真实 decoder 缓存和精度恢复；`results/` 保存聚合成绩与 SHA256。

不依赖旧项目 editable 安装、私有服务器路径或旧 venv。checkpoint 搬迁时只生成替换 `mm_vision_tower` 路径的临时配置视图，权重通过链接复用，不修改原 checkpoint，该变化记入运行元数据。

本分支提供可重跑流程与版本证据，不保证不同硬件、驱动、CUDA 内核或随机实现下逐位相同的权重/分数。论文与 release 的已知差异被明确保留；若要严格对齐论文文字设置，应另做实验，不能据此断言论文结果偏高。
