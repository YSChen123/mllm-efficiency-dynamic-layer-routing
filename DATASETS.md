# 数据与模型下载清单

数据暂不清理。路径保留原位，便于其他复现复用。2026-11-03 盘点，GiB=2^30 bytes。训练图像是 JSON 实际引用子集，不是每个数据集的完整发布。

## 训练输入

`liuhaotian/LLaVA-Instruct-150K` 的 `llava_v1_5_mix665k.json`，revision `9d451dc7629cfe0469f6ae4432b765cd603d5fcb`，665,298 条，1,029,887,963 bytes；SHA-256 `ce959ce6e23073ee1cd1a8a2ef1c633768c10d4174327b8b2dc7113b91af6cf8`。

JSON、manifest、READY、smoke JSON 留在 `/home/SigConv/cys/AdaLLaVA/training_repro/data/`。统一图像入口 `.../data/images/` 的目录/符号链接保留。

| 来源 | 不同图像数 | GiB | 实际路径 |
|---|---:|---:|---|
| COCO train2017 | 88,524 | 13.66 | `/home/SigConv/cys/AdaLLaVA/training_repro/data/images/coco` |
| GQA | 72,140 | 9.64 | `/home/SigConv/backup2/adallava-training-20261027/images/gqa` |
| OCR-VQA | 80,000 | 3.37 | `/home/SigConv/backup2/adallava-training-20261027/images/ocr_vqa` |
| TextVQA train images | 21,953 | 5.80 | `/home/SigConv/backup2/adallava-training-20261027/images/textvqa` |
| Visual Genome | 86,417 | 11.72 | `/home/SigConv/backup2/adallava-training-20261027/images/vg` |

训练图像约 44.18 GiB，JSON 约 0.96 GiB，text-only 样本 40,688 条。图像镜像 `kaiyuyue/llava-1.5-665k-instructions`，revision `f321f47bf67bf43969756ef22650e04ab367bcaa`，`train_split/{coco,ocr_vqa,textvqa,vg,gqa}.tar.gz`。校验 archive LFS SHA、保存原字节；详细 metadata 见 [`provenance/image_mirrors.json`](provenance/image_mirrors.json)。6 张 COCO 与原站交叉核对过，未声称全部图像都已和原站核对。

## 评测数据

| 基准 | HF repo / config | revision | split / 样本数 | 保留目录（前缀 `/home/SigConv/cys/AdaLLaVA/datasets/`） | GiB |
|---|---|---|---|---|---:|
| ScienceQA-IMG | `lmms-lab/ScienceQA` / `ScienceQA-IMG` | `69dd4d6b67373d38f96a5badd5d24d0eb5bcdc50` | test / 2,017 | `scienceqa_img` | 0.50 |
| POPE | `lmms-lab/POPE` | `4db1276663dfa5eb8ad16a52d24c31a09e470896` | test / 9,000 | `pope` | 1.61 |
| MME | `lmms-lab/MME` | `d6c9023f017b564f7b3ccccf5348166bce8fdbcd` | test / 2,374 | `mme` | 2.42 |
| TextVQA | `lmms-lab/textvqa` | `9c0699cd19768ac5ab97568f6b3cbac4c0062884` | validation / 5,000 | `textvqa_val` | 5.52 |

评测缓存约 10.05 GiB，包括 Arrow 和下载缓存，可能含副本。服务器 manifest 保留于 `/home/SigConv/cys/AdaLLaVA/configs/datasets/`，Git 副本见 [`provenance/dataset-manifests/`](provenance/dataset-manifests/)。TextVQA OCR 开启；MMBench 未完成，本清单没有可复用的 MMBench 缓存。

## 预训练权重（Git 不保存）

| 用途 | 仓库 | revision |
|---|---|---|
| 重训底座 | `liuhaotian/llava-v1.5-7b` | `4481d270cc22fd5c4d1bb5df129622006ccd9234` |
| 冻结视觉编码器 | `openai/clip-vit-large-patch14-336` | `ce19dc912ca5cd21c8a653c79e251e808ccabcd1` |
| 可选官方对比 | `zhuoyanxu/ada-llava-L-v1.5-7b` | `5b3e371bdcd16e597b35a60f5e5d9f66c64beff6` |

ROE 正在使用 `/home/SigConv/cys/AdaLLaVA/checkpoints/llava-v1.5-7b` 和 `.../clip-vit-large-patch14-336`，两份共享底座暂留，保护在跑实验。AdaLLaVA 产出的中间/final checkpoint 和官方 Ada checkpoint 已在 GitHub 校验后清理。数据及共享底座保留文件身份清单前后一致。最终回执见 [`ARCHIVE_STATUS.md`](docs/ARCHIVE_STATUS.md)。
