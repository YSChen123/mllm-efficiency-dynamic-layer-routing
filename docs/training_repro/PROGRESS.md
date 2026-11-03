# Execution ledger — plan: docs/superpowers/plans/2026-10-27-adallava-full-training.md

2026-10-31 结果复核：两条评测队列均COMPLETED、失败0；48组新测试+20组旧官方结果共68组重新验证，220692条新逐样本测量完整。报告docs/evaluation_repro/RESULTS_20261031.md与预算/实测prefill曲线已生成。满预算自训ScienceQA67.4269、POPE F185.4447、MME Perception1463.2141、TextVQA56.2760；相对官方差−0.1487pp/−0.5757pp/−57.5594分/−0.0980pp。MME类别损失主要OCR−30、celebrity−15.2941、posters−13.2653，官方聚合函数复算一致；不能称为全部严格复现。MMBench judge、同卡受控profiling及P2深度分析仍未完成，未重训/改评分。绘图依赖只装到evaluation_repro/plot_lib，未改评测或训练环境。

2026-10-31：用户要求确认结束并立即测试。训练正常于2026-10-30 08:49 CST结束，5197更新/两rank exit0/最后checkpoint及portable导出已验收；实际约55.99h、111.97GPU-hours。自主模型1.0/.5、Dense1.0、官方.85真实smoke均通过，最终分片SHA及checkpoint权重样本一致。独立审查及修复后8项CPU回归通过。正式P1两队列2026-10-31 10:34启动，GPU0自主32组、GPU3 Dense/官方补充16组，旧官方20组准确率重新校验复用；交接时ScienceQA自主886/2017、Dense1043/2017。MMBench因缺judge凭据明确blocked，其余继续。详情见../evaluation_repro/HANDOFF_20261031.md。

2026-10-28 22:34 CST 用户要求更严格对齐后的追加审计：读取正式checkpoint内实际TrainingArguments/模型config/已解析DS config，核对37个官方CLI选项、原源码clean状态及梯度累积实现；第2000步engine记录256000samples、skipped_steps0，global128已实证。直接查看附录训练图约5200步，保留global128的有据选择，未改运行中配置。修正文档中“HF max_grad_norm1即实际clip1”的不严谨表述：官方及当前DS有效clip均为0。版本/seed时机/图像跨源验证边界如实记录。结论及证据见STRICT_ALIGNMENT_REVIEW.md，不宣称逐项完全等同作者环境。

最新状态（2026-10-28 00:54:21 CST）：Task1—4的正式训练启动与交接工作完成；训练本身仍RUNNING，尚未结束。双rank实际完成5/5197步，连续5个有限loss，GPU0/1各约65.9GiB、利用率92%/93%。停止主动监控，按用户要求由tmux继续。证据归档20261028_005421.tar.gz已下载本地，100个文件逐一SHA通过，实际脚本副本位于reproduction/training。P1/P2待用户训练结束后继续。此前clean accum4 smoke fresh/resume均exit0，CP81,936,189,661 bytes，模型/Adam/LR/RNG/portable导出与重载均通过。下文为按时间保存的过程记录。

2026-10-27: 用户提供完整规格并授权直接执行。采用当前会话内执行，仅在最终需独立审查时使用执行技能指定的 reviewer。

Task 1: in_progress。原 AdaLLaVA commit 2a020d90973b1a387c75a042e68a1194e32757d2、LLaVA commit c121f0432da27facab705978f83c4ada465e46fd；20 组旧结果全 COMPLETED。GPU 0/1/2 空闲，GPU 3 有其他任务。主盘用户软配额 270 GiB，已用约75.2 GiB；sdc1 150 GiB 软配额已用约85.1 GiB；sdb1 超过150 GiB软配额。

Task 2: 官方底座与1.03GB完整注释后台下载已启动，tmux ada_train_inputs。CLIP 直接复用。尚未 READY，不可正式训练。

Ruling: 规格已明确“无需再确认”，不重复请求计划/设计或开始训练批准；所有资源和算法决策留痕。

Ruling: 训练使用独立源码与轻量环境，避免改变已有正式评测；不复用官方 AdaLLaVA 权重作为训练起点。

Pre-flight interfaces: 任务2数据manifest是任务3/4的准入条件；任务3保存/恢复证据是任务4准入条件。训练 READY 不能由下载进程存在推断。

Known upstream issue: AdaLLaVA #3 作者建议 ZeRO-3 stage3_prefetch_bucket_size=0 处理 inflight params；计划记录并在 smoke 核实。

2026-10-27 23:49 CST: 原始底座及全量 JSON 全部 SHA 校验完成。665298 样本、349034 唯一图像引用，不能缩水。训练 venv + worktree 已创建；DS0.16.9 导入通过。图像 raw tar 镜像固定 revision，tmux ada_train_images 顺序哈希/解码，ada_train_prefetch 两个有界 worker 共用缓存预取，不重复归档。
Ruling: 拟 global batch=128（2 GPU × micro2 × accum32）— 上游 Stage-2 batch128 与附录约5200steps支持，但 Ada 脚本没有GPU数量，不能声称作者明确确认 — 如果推断不符原实验，globalbatch/浮点/采样差异必须计入结果解释。
Ruling: checkpoint 保留完整 ZeRO 状态而中间不额外 gather HF 权重；最终再导出 — 降低双 checkpoint 瞬时峰值，保持恢复状态完整 — smoke 必须实测保存/恢复和大小。
Task 3: 单元测试8项红→绿；首次双卡 smoke 在模型加载前 NCCL illegal memory access。最小 all-reduce 证实大消息失败；NCCL_P2P_DISABLE=1 仍失败。上游 NVIDIA #2418 对应2.26.2/Blackwell故障，训练环境单独安装2.31.2验证中；未动原评测环境、驱动或其他任务。

2026-10-28 00:06 CST: NCCL2.31.2 大消息测试通过；smoke 两步 loss=5.3345、2.1278，Scheduler head 与 backbone norm 均有有限非零梯度，step2 FP32 master变化约1e-5。实测可训练6,777,295,376，Scheduler17,900,048；单rank峰值allocated62.67GB、reserved69.88GB。正在真实保存 checkpoint，尚未宣称恢复通过。
Review: fresh reviewer training_final_review 全套代码审查完成，9项本地测试通过。确认 RNG旧Trainer/Torch2.7兼容错误：已新增复现→窄修复→CPU回归通过；新增最终HF导出/重载验证；LR state恢复显式对比已补充。
Ruling: 保留官方Trainer4.37.2的一epoch协议及完整665298 JSON，不另改梯度累积算法或偷偷重复第二epoch — 预计5197steps/665216呈现、尾部82是原loop行为，不是手动筛除数据 — 最终必须如实说明，不能声称每行都更新。
Data: COCO88524张原图14,485,550,009bytes已全部读取通过；OCR、TextVQA下载已完成并按序校验；VG/GQA仍后台进行。尝试通过原GQA ZIP中央目录验证VG复用，但官方Range请求503且只有57350/72140文件名重叠，未未经字节校验擅自复用。

2026-10-28 00:24 CST: smoke实际续训成功，两个rank进程exit0，从step2恢复至3；模型FP32 master、Adam state指纹、LR scheduler state、Python/NumPy/CPU/CUDA RNG全部断言一致；step3仍有非零梯度和约5e-6更新。最终gather HF权重导出完成，独立from_pretrained重载进行中。
Runtime correction: LR scheduler实际由Trainer持有，DeepSpeedEngine.lr_scheduler=None；改审计读Trainer属性。CPU回归2/2通过。避免第三次重写77GiB：官方zero_to_fp32 lazy独立读取两rank已完整落盘分片，校验globalstep2/worldsize2，补充审计sidecars；实际resume完成后重新执行live on_save回调成功。未修改任何磁盘权重/Adam/RNG来通过验证。

2026-10-28 00:28 CST: Task3 complete。SMOKE_PASSED.json与EXPORT_RELOAD_PASSED.json均passed，真实portable加载无missing/mismatched trainable keys；CP81936189747bytes（76.31GiB）。保存小证据后删除本次已验证的smoke大权重/optimizer，释放约90GiB，保留所有日志。6张官方下载COCO与镜像原图SHA一致。VG86417张全部读通过；GQA剩余下载约1.4GB时最后检查，准备脚本后台继续，未高频查询。

2026-10-28 00:33 CST: Task1 complete；Task2 complete。DATA_READY.json正式生成，665298指令、349034图像全量decode通过，5源归档均SHA验证，下载归档已删除。图像总原始字节46,715,299,500（约43.51GiB），其中COCO主盘13.49GiB，其他backup2约30GiB。Task4正式preflight/launch进行中，不等待用户再次确认。

2026-10-28 00:35 CST: formal首次attempt在第一个更新前因旧Accelerate调用DS0.16新增no_sync断言失败；无formal checkpoint，无参数更新，未宣称稳定。Task3重新打开以补测accum>1。按上游Accelerate #3481回补ZeRO梯度分片不进入DDP no_sync，DeepSpeed累积本身不改；失败→修复→本地11测试通过，32条真实smoke数据/micro2/accum4开始，两步更新已通过并正在保存。
Ruling: Torch NCCL版本函数返回编译期版本，入口nccl_runtime字段改用ctypes调用预加载库ncclGetVersion，另标torch_build版本；不篡改首次失败尝试历史日志。新attempt将保留新provenance快照。
