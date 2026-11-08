# PAR（ICCV 2025）公开设置复现

本分支提供 LLaVA-v1.5-7B 的 Layer-only K4/M0 与 Joint K4/M400，从官方原始输入重新构建偏好、训练 router、独立验证选优并评测 POPE/SEED-I。它是公开材料基础上的重建实现，**不是作者完整训练代码，也不声称所有作者 settings 已知**。

当前已验收的 Layer-only 为 POPE **86.4938**、SEED-I **65.5308**，接近论文 Table3 DPO 的85.8/65.1；Joint 验证选中模型为 **85.9743/63.1841**，对照论文76.1/62.7。Joint 的 POPE 高9.8743个百分点仍未得到唯一解释。完整复核与本轮mask/batch实验见 [结果](docs/RESULTS.md) 和 [设置](docs/SETTINGS.md)。

## 1. 安装独立环境

在新的 Linux/NVIDIA 服务器准备 Python3.10、git，以及支持 CUDA12.8 的驱动。参考环境为 Python3.10.22、Torch2.7.1+cu128、TF4.37.2、Pillow10.2.0；实测 GPU 为 RTX PRO6000 96GB。单GPU可串行运行，4GPU可并行准备数据和训练四条轨迹。较小GPU未做显存保证。

```bash
git clone --branch PAR --single-branch https://github.com/YSChen123/mllm-efficiency-dynamic-layer-routing.git
cd mllm-efficiency-dynamic-layer-routing/PAR
bash scripts/setup.sh
source .venv/bin/activate
```

若解释器命令不同：`PAR_PYTHON=/path/to/python3.10 bash scripts/setup.sh`。安装脚本创建独立 venv，不读取或共享其他复现工程的 site-packages，不安装上游过时的 Torch 依赖。数值参考中的Torch/backend相对作者仓库有差异，见设置说明。

## 2. 原始数据与模型

选择一个有足够空间的工作目录，例如 `/data/par_joint`。下载并解压标准 [COCO train2017](https://images.cocodataset.org/zips/train2017.zip) 和 [COCO val2014](https://images.cocodataset.org/zips/val2014.zip)，分别放在：

```text
/data/par_joint/datasets/coco/train2017/*.jpg
/data/par_joint/datasets/coco/val2014/*.jpg
```

然后执行：

```bash
python -m par_repro --work-dir /data/par_joint prepare --download-models --download-seed-images
```

该命令下载并校验固定revision的 LLaVA-v1.5-7B、CLIP、原始 LLaVA80k 标注、POPE 历史三类标注、SEED 官方元数据和 **官方 SEED-Bench-image.zip**，并检出固定baseline源码。模型文件约14GB，COCO与工作产物还需要额外空间。不会下载SEED视频。

已有权重/图像可复用原始资源：复制 `configs/joint.json` 为 `configs/site.local.json`，将 `model.backbone`、`model.vision`、`data.coco_train`、`data.coco_val`、`data.seed_images` 改为你的目录。所有命令加 `--config configs/site.local.json`；已有权重时可省略 `--download-models`，已有SEED官方图像时可省略 `--download-seed-images`。相对路径按工作目录解析。`model.llava_source` 默认为工作目录内的源码checkout，保持默认即可。

SEED图像允许外层目录嵌套，按data_id精确匹配文件stem；重复stem会报错。固定5256条训练/验证样本ID与解码图像指纹随代码提供；偏好生成时验证实际图像，防止换数据后仍沿用旧结果。**没有旧Arrow缓存、旧环境或旧验收状态文件的依赖。**

## 3. 先做完整小测

用独立工作目录，例如 `/data/par_smoke`，完成同样的prepare；可用site.local.json让它共享原始模型和数据，避免复制大文件。

```bash
python -m par_repro --work-dir /data/par_smoke --config configs/site.local.json prepare
python -m par_repro --work-dir /data/par_smoke --config configs/site.local.json run --gpus 0 --smoke
```

小测使用4训练/4验证输入、4次更新，执行偏好构建、checkpoint保存/恢复所需产物、POPE6条与SEED4条推理、实际预算和cached/uncached检查。所有小测产物明确为SMOKE_ONLY，不能作为正式分数或混入正式目录。恢复等价与数值核对的额外检查见 [迁移验收](docs/REPRODUCIBILITY.md)。

## 4. 完整 Joint 训练、选优、测试

```bash
python -m par_repro --work-dir /data/par_joint run --gpus 0,1,2,3
```

它会依次构建5000训练/256验证输入各5动作，运行已注册的因果/双向mask × 全局batch4/16四条轨迹，各3epochs，保留epoch1/3的8个候选；只按256张验证图像的实际路由Full-KL选择，然后自动测试完整POPE8910和SEED-I14233。不使用benchmark分数选参。`--gpus 0` 以单卡分批串行执行同一流程；每个worker只使用一张暴露的GPU。

```bash
python -m par_repro --work-dir /data/par_joint status
tail -n 10 /data/par_joint/logs/train_bidirectional_b4.log
```

中断后重跑同一个 `run` 命令：已验证偏好逐条复用，训练从latest恢复，已完整的评测重验后跳过。训练最多重做尚未保存的250步；未保存日志尾部会另存保留。不得修改配置后续接旧输出目录，身份变化会停止。设备有其他进程占用时worker会停止，程序不会终止别人的任务。

## 5. Layer-only

使用新的工作目录，以 `configs/layer_only.json` 或其本地路径配置准备原始资源：

```bash
python -m par_repro --work-dir /data/par_layer --config configs/layer_only.json prepare --download-models --download-seed-images
python -m par_repro --work-dir /data/par_layer --config configs/layer_only.json run --gpus 0,1,2,3
```

Layer-only保持5000输入、5动作、M0、均衡2正×2负、global4、lr1e-4、1epoch/1250步，之后自动评测两项完整benchmark。其配置与Joint分开，不能复用不同模式的router。

## 6. 单独操作和产物

```bash
CUDA_VISIBLE_DEVICES=0 python -m par_repro --work-dir /data/par_joint train --name bidirectional_b4 --resume
CUDA_VISIBLE_DEVICES=0 python -m par_repro --work-dir /data/par_joint evaluate --benchmark pope --checkpoint /data/par_joint/checkpoints/bidirectional_b4/epoch3.pt
python -m par_repro --work-dir /data/par_joint verify --result-dir /data/par_joint/results/selected/pope
```

完整运行保存有效配置、输入/实现/权重/预测哈希、逐条动作/生成文本、评分和独立verification。不同候选必须使用对应的 `--config /data/par_joint/configs/<trial>.json`；否则拒绝评测。结果与权重不提交Git。训练权重可在完成后自行复制到新服务器；从头复现无需旧checkpoint。`convert-checkpoint` 只用于本项目历史正式5000样本权重的显式转换，禁止将smoke权重提升为正式。

目前覆盖这两项benchmark与固定预算。论文其余benchmark、多预算、Joint-then-layer、实测速度/FLOPs不在已完成范围。更高POPE不证明设置一致；未知作者参数、公开论文/代码冲突及实验负结果全部保留。
