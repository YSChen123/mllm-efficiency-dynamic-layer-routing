# 本次 AdaLLaVA-L 训练脚本

这是远程 `/home/SigConv/cys/AdaLLaVA/training_repro/` 实际脚本的本地副本；不包含模型和数据图像。源码/环境/数据来源及所有差异见 `docs/training_repro/CONFIG_AUDIT.md`，执行记录见 `PROGRESS.md`。

交接说明：[`TRAINING_HANDOFF.md`](../../docs/training_repro/TRAINING_HANDOFF.md)。当前脚本固定了本次服务器路径，需要先读交接说明；它们不是脱离已有环境即可运行的通用安装包。凭据不包含在此目录或证据归档中。

在服务器项目根目录检查状态：

```bash
envs/adallava-train/bin/python training_repro/training_status.py
tail -n 40 training_repro/logs/formal.log
```

已有完整 checkpoint 且训练已退出时恢复：

```bash
bash training_repro/start_formal.sh resume
```

首次 checkpoint 前中断，需从原底座重新 fresh；`resume` 会拒绝没有完整checkpoint的目录。不要把smoke权重作为正式起点。训练入口会检查全量数据READY和smoke证据；中间保存完整ZeRO状态，成功写完新checkpoint后才轮换，最终导出通用权重。

本地数据/守卫/no_sync回归：

```powershell
python -m unittest discover -s reproduction/training -p "test_*.py" -v
```

服务器依赖回归：

```bash
envs/adallava-train/bin/python training_repro/check_rng_compat.py
envs/adallava-train/bin/python training_repro/check_optimizer_audit.py
LD_PRELOAD=/home/SigConv/cys/AdaLLaVA/envs/adallava-train/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2 CUDA_VISIBLE_DEVICES=0,1 envs/adallava-train/bin/python -m torch.distributed.run --nproc_per_node=2 --master_port=29728 training_repro/check_nccl.py
```

NCCL测试需要两张空闲卡，正式训练运行期间不重复执行。
