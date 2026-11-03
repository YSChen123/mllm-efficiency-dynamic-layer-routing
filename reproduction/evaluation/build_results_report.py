"""Build a source-linked Chinese report from already validated local evidence."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'docs/evaluation_repro'
d=json.loads((DEST/'evidence/validated_results_68.json').read_text(encoding='utf-8'))
m=json.loads((DEST/'evidence/mme_diagnosis.json').read_text(encoding='utf-8'))
paper=json.loads((ROOT/'results/2026-10-24/paper_comparison/paper_reference.json').read_text(encoding='utf-8'))
tasks=['scienceqa_img','pope','mme','textvqa_val']
names={'scienceqa_img':'ScienceQA-IMG Accuracy (%)','pope':'POPE F1 (%)',
       'mme':'MME Perception','textvqa_val':'TextVQA consensus accuracy (%)'}
idx={(r['role'],r['task'],r['budget']):r for r in d['rows']}
assert len(idx)==68 and d['validated_new_experiments']==48 and d['validated_reused_experiments']==20
lines=['# AdaLLaVA-L 自主训练复现：正式结果与差距分析','',
 '整理日期：2026-10-31。结论：自主训练和本轮四benchmark评测均正常完成，主要预算—性能趋势接近官方checkpoint；**满预算MME仍存在明显差距，不能宣称原文全部指标已严格复现**。',
 '', '本轮48组新实验（自主32、Dense4、官方补充12）全部完成且无失败；另复核并复用官方旧20组准确率，形成68个有效模型/任务/预算组合。全部来自完整样本、limit=None、batch1。两个队列分别于12:52和14:55 CST结束。',
 '', '## 满预算：同环境实测对照','',
 '| 指标 | Dense实测 | 官方checkpoint实测 | 自主训练 | 自主−官方 |',
 '|---|---:|---:|---:|---:|']
for task in tasks:
    a=idx['reproduced',task,1.]['score'];b=idx['official',task,1.]['score'];base=idx['dense',task,1.]['score']
    lines.append(f'| {names[task]} | {base:.2f} | {b:.2f} | {a:.2f} | {a-b:+.2f} |')
lines += ['', '百分数差值单位为百分点，MME差值单位为分。POPE采用F1，不混用Accuracy；MME只与论文Perception比较，不把Cognition加进去。',
 '', '与原始Dense相比，自主模型满预算ScienceQA下降2.08个百分点、TextVQA下降2.01个百分点，POPE F1下降0.38个百分点、MME下降49.34分。接近官方checkpoint并不自动意味着完全保持Dense性能。',
 '', '## 各预算准确率','',
 '| 请求budget | 模型 | ScienceQA (%) | POPE F1 (%) | MME Perception | TextVQA (%) |',
 '|---:|---|---:|---:|---:|---:|']
for budget in [1.,.9,.85,.8,.7,.6,.55,.5]:
    for role,label in [('official','官方checkpoint'),('reproduced','自主训练')]:
        vals=[idx[role,t,budget]['score'] for t in tasks]
        lines.append(f'| {budget:.2f} | {label} | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
lines += ['', '85%预算时，自主TextVQA比官方高0.92个百分点、POPE F1高0.14个百分点；60%预算时，ScienceQA、POPE F1、TextVQA与官方分别相差−0.35、−0.14、−0.13个百分点，MME低18.51分。',
 '', '50%为只执行固定前16层的边界：自主TextVQA27.85%，官方29.82%；自主POPE F1低于官方1.67个百分点。两种checkpoint均表现出明显的TextVQA边界退化，不能用ScienceQA/POPE较稳定的分数概括所有任务。准确率不要求随预算严格单调：自主MME在80%预算为1491.78，高于其满预算1463.21，需如实呈现。',
 '', '预算按floor(32×budget)量化，因此0.9/0.85/0.8/0.7/0.6/0.55分别执行28/27/25/22/19/17层，0.5为16层，1.0为32层；不是精确连续FLOPs比例。全部新自适应测量记录均满足预期层数、前16层固定执行及预算约束。',
 '', '![准确率与预算](figures/accuracy_budget.png)',
 '', '## 与论文同预算对比','',
 f"来源：[ICCV2025 Table1]({paper['source']})，普通AdaLLaVA-L/Vicuna7B，无token pruning。论文原始数值和PDF哈希保存在既有paper_comparison目录。论文是参考值，不能当作本机实测。",
 '', '| Budget | 指标 | 原文 | 自主训练 | 差值 |','|---:|---|---:|---:|---:|']
paper_deltas=[]
for key,ref in paper['rows'].items():
    for task in tasks:
        value=idx['reproduced',task,float(key)]['score'];delta=value-ref[task]
        lines.append(f'| {float(key):.2f} | {names[task]} | {ref[task]:.2f} | {value:.2f} | {delta:+.2f} |')
        paper_deltas.append({'task':task,'budget':float(key),'paper':ref[task],'reproduced':value,'delta':delta})
lines += ['', '共同三档内，ScienceQA、POPE F1、TextVQA各项绝对差均小于1个百分点；MME三档低57.79、11.59、24.36分。没有设定事后通过阈值，也不据单次训练称为统计等价。',
 '', 'Dense ScienceQA实测69.51%，高于论文表中的66.8%；其他Dense三项与论文较接近。因此“相对Dense保真率”也存在基线差异，不能直接复述论文平均保真率。尚未锁定该差异的唯一原因。',
 '', '## MME满预算差距来自哪里','',
 '按官方MME聚合函数对保存的回答重新计分，类别和问题ID逐项匹配，总和与正式结果一致。每类最高200分，由问题准确率和同图两问全对率共同构成；并非普通总体准确率。',
 '', '| 类别 | 题数 | 官方 | 自主 | 差值 |','|---|---:|---:|---:|---:|']
counts=m['full_budget']['reproduced']['category_question_counts']
for row in m['category_comparison']:
    lines.append(f"| {row['category']} | {counts[row['category']]} | {row['official']:.2f} | {row['reproduced']:.2f} | {row['delta']:+.2f} |")
lines += ['', f"感知部分共2114道题：官方对而自主错{m['official_correct_own_wrong']}道，自主对而官方错{m['own_correct_official_wrong']}道。OCR只有40题，其类别分数对少量错例比较敏感。全MME2374题中，自主无法解析为yes/no的回答3条，官方4条，Dense3条；现有证据不支持把差距简单归为普遍回答格式失败。",
 '', '这将排查范围缩小到OCR、celebrity、posters等类别，但没有证明唯一训练原因。后续优先查看配对错例、对照输入/评分设置，再考虑受控配置或多seed验证；现阶段不应靠改评分规则或人为选点消除差距。',
 '', '## 实测效率：与Dense按doc_id配对','',
 '每个组合排除最初3个冷启动/预热样本，并取与Dense共同doc_id。完整generate含视觉编码、LM prefill及decode，不含CPU图像预处理和tokenization；LM prefill为首次LM forward的CUDA事件时间。GPU型号相同但物理卡不同，未锁定时钟；下表是观察值，不能把小幅差异归因于Scheduler单一因素。生成token数不同也影响完整generate速度。',
 '', '| 指标任务 | Budget | 自主LM prefill ms | 自主完整generate ms | LM prefill加速比 | 完整generate加速比 |',
 '|---|---:|---:|---:|---:|---:|']
for task in tasks:
    for budget in [1.,.85,.6,.5]:
        row=idx['reproduced',task,budget];o=row['observed']
        c=next(r for r in d['comparison'] if r['task']==task and r['budget']==budget)
        lines.append(f"| {task} | {budget:.2f} | {o['mean_lm_prefill_ms']:.2f} | {o['mean_generate_ms']:.2f} | {c['lm_prefill_speedup_vs_dense']:.2f}× | {c['generate_speedup_vs_dense']:.2f}× |")
lines += ['', '60%预算完整generate观察加速约1.34–1.56×；50%约1.35–1.82×。满预算完整generate约0.92–0.97×，没有加速收益。不能把50%预算写成实测2×整机加速。',
 '', '![准确率与实测LM预填充延迟](figures/accuracy_latency.png)',
 '', '官方旧20组只复用准确率，其分析器prefill_time不作为新的硬件实测。图中官方新测时延仅含90%、80%、50%三个点。分析器FLOPs、roofline时间与本机CUDA事件分别保存在JSON，不混为同一种测量。',
 '', 'Scheduler forward在85%预算约1.07–1.11ms（不同benchmark均值），仅为事件计时观察；不代表完整附加训练/推理成本。Latency token对前缀层的影响、训练forward/backward额外成本仍需专门profiling。',
 '', '## 训练与验证范围','',
 '正式训练5197更新正常结束，最后checkpoint和三份portable权重校验/重载通过。模型入口耗时约55.99小时，约111.97 GPU-hours；最后loss0.6608、全程均值约0.87086。可训练参数6,777,295,376，Scheduler新增17,900,048。Training FLOPs未可靠实测。',
 '', '核心训练方法保持原实现。global128由Stage-2及附录约5200步曲线支持，但作者卡数未明确公开；两卡累积32替代脚本累积4。初始化seed时机、Blackwell软件环境和必要兼容修复详见[严格配置审计](../training_repro/STRICT_ALIGNMENT_REVIEW.md)。不能把这些差异说成已被消除，也不能仅凭当前分数确定其因果贡献。',
 '', '当前完成的是4个benchmark、8个预算的自主/官方准确率对照及Dense4组，还有实际成本与逐样本路由记录。MMBench缺少官方judge凭据，尚未完成；VQAv2未纳入本轮。统一同卡profiling、Scheduler训练额外开销、跨输入routing分布和Fig.4真实attention可视化仍是后续分析项，不宣称整篇论文全部实验已复现。',
 '', '## 可追溯资料','',
 '- [68组复核结果及配对效率](evidence/validated_results_68.json)：包含指标名、样本数、原始文件路径及SHA。',
 '- [MME类别及配对错例](evidence/mme_diagnosis.json)。',
 '- [训练完成成本](../training_repro/evidence/training_completed_20261030.json)。',
 '- [评测启动与恢复说明](HANDOFF_20261031.md)。',
 '- 服务器原始结果：`/home/SigConv/cys/AdaLLaVA/evaluation_repro/20261031/{reproduced,dense,official}/outputs/formal/`；旧20组留在原`outputs/formal/`。',
 '- 结果离线复核脚本副本：`reproduction/evaluation/analyze_results.py`、`diagnose_mme.py`、`plot_results.py`、`build_results_report.py`。','']
(DEST/'RESULTS_20261031.md').write_text('\n'.join(lines),encoding='utf-8')
(DEST/'evidence/paper_comparison_12.json').write_text(json.dumps(paper_deltas,indent=2),encoding='utf-8')
print('REPORT_CREATED',DEST/'RESULTS_20261031.md')
