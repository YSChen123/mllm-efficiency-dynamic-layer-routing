# 自主模型评测脚本副本

对应服务器`/home/SigConv/cys/AdaLLaVA/evaluation_repro/scripts/`。运行依赖已有`envs/adallava`、`scripts/pipeline.py`及固定版本AdaLLaVA/LLaVA/lmms-eval和已验证Arrow数据；不是独立安装包，不包含凭据、模型或图像。

最终结果与离线分析见[RESULTS_20261031.md](../../docs/evaluation_repro/RESULTS_20261031.md)。`analyze_results.py`重新验证48新+20旧结果，`diagnose_mme.py`复用官方聚合函数。绘图使用独立`evaluation_repro/plot_lib`内的Matplotlib3.8.4等固定依赖；服务器执行绘图时将该目录放入PYTHONPATH，不修改原评测环境。

状态、日志、预算、恢复方式及限制见[评测交接](../../docs/evaluation_repro/HANDOFF_20261031.md)。smoke与正式结果分开，完成验证才标记成功。

本地纯CPU回归：

```powershell
python -m unittest discover -s reproduction/evaluation -p "test_*.py" -v
```
