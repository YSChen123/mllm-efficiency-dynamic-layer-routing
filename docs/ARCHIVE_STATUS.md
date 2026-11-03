# 归档与清理状态

用户要求日期：2026-11-03（Asia/Shanghai）。本地整理、从本地 push；一篇论文一个分支；数据暂留；不保存训练 checkpoint，不重训或重跑完整评测。

完整源码/实验记录已下载到上传端本地归档。SHA-256 `3b287110bc9e1af208eb46e44268786898618ba633afb5e14026ff7291aeb03a`，123,175,667 bytes，1,145 个执行文件逐项核对。三套官方源码另附固定 revision。

已完成从本地推送：[私有仓库](https://github.com/YSChen123/mllm-efficiency-dynamic-layer-routing)，分支 `main`（论文索引）和 `adallava`（本次完整代码/settings/结果）。GitHub API 确认 private=true；清理前两分支提交均与本地一致。独立从 GitHub 克隆，1,672 个源码/脚本/测试文件逐字节一致；27 项 CPU 测试与 git fsck 通过。未从服务器执行 Git push。

服务器清理于 2026-11-03 23:49（北京时间）完成。删除 72 个 AdaLLaVA 专用目标，包括正式/中间 checkpoint、官方 Ada checkpoint、环境、源码、脚本、日志、评测输出和训练辅助文件；原始小型实验记录已在本地归档。

项目目录减少 120,056,283,136 bytes（111.81 GiB）磁盘占用；临时打包/部署预演目录也已删除，合计减少 112.16 GiB。这里是目录磁盘占用差值，不保证与共享账户 quota 或其他项目同时变化后的文件系统空闲增量相同。

保留全部训练/评测数据及 manifest，图像链接保留原位。ROE 正在引用的 LLaVA/CLIP 预训练底座也保留原位，没有修改 ROE 进程或项目文件。清理后仍观察到 ROE 的训练/数据加载进程；未把这一点表述成完整训练健康检查。

349,215 个保留文件、73,774,524,180 logical bytes 的路径、解析目标、大小、mtime、inode 清单 SHA-256 在清理前后相同。此项是文件身份核验，不是重新读取所有数据内容做全量 SHA。

回执：[删除回执](../provenance/server-cleanup-receipt.json)、[独立验收](../provenance/server-cleanup-verified.json)。服务器保留目录：`datasets/`、`training_repro/data/`、`configs/datasets/`、`checkpoints/{llava-v1.5-7b,clip-vit-large-patch14-336}/`，以及两份小型清理回执。数据逐项路径见 [DATASETS.md](../DATASETS.md)。

上传端本地：`C:\Users\chenys\Desktop\AdaLLAVA\mllm-efficiency-dynamic-layer-routing\`；完整实验记录归档：`C:\Users\chenys\Desktop\AdaLLAVA\archives\adallava-20261103\adallava-code-and-evidence-20261103.tar.gz`。
