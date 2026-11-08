# 设置来源与未核实项

公开来源：[PAR论文](https://arxiv.org/html/2412.06458v2)、[官方PAR仓库固定main](https://github.com/ASGO-MM-Lab/Pruning-All-Rounder/tree/1890d445e9cc61fed0de5bc2cd825af77c62e8ef)、[FastV评测入口](https://github.com/pkunlp-icler/FastV/tree/d1659729b5bf1be225e99ee15783deeea80f63b1/src/LLaVA)、[LLaVolta POPE说明](https://github.com/Beckschen/LLaVolta/tree/31ec1516ac282b7aa53a0d11e476f7295523ed3e)。资源revision与逐文件校验值在configs/resources.json、annotations.json。

| 项目 | 本实现 | 依据/限制 |
|---|---|---|
| Backbone | 冻结LLaVA-v1.5-7B，CLIP336，mlp2x_gelu | 公开模型家族一致；作者精确revision未公开 |
| 预算 | 最后16层丢4层；Joint丢400/576视觉token，Layer-only M0 | 对齐公开预算；实际执行28blocks，真实删除token |
| 时机/位置 | 前2blocks完整视觉，第二块后剪token，保留原始RoPE/cache时间轴 | 公开llama代码存在前2层例外；作者完整运行路径不明 |
| Router | 单block40,178,688参数、4096→1536、FFN5120、12heads，MLP1536→512→1、GELU、sigmoid，16控制embedding | 约40M Transformer+MLP来自论文；宽度、激活、共享头、mask、初始化等是明确重建假设 |
| Mask | 默认双向；提供因果配置与受控对照 | 公开简化selector是因果mask；完整作者router未公开。不能宣称默认mask已与官方代码一致 |
| 高分动作 | bit1=丢弃，高分丢弃 | 论文Eq3后写高分丢弃；公开简化selector降序保留高分，与论文冲突。训练和推理内部一致，不能只反转推理排序 |
| 输入/动作 | 原始LLaVA80k固定seed0的5000条，首human输入；每条5个联合动作 | 5000/5对齐；精确作者IDs/seed/turn未知。原样保存完整记录，但只首轮问题进入router |
| token候选 | 各head QK/sqrt(d)加Gaussian std√10，然后softmax、head均值，丢最低400 | N(0,10)解释为方差10；域/方差约定未公开。公开残留std1代码不能视作完整候选构建 |
| KL | Full‖pruned，Full生成响应teacher forcing，响应token平均、含EOS | 方向、归一化、生成范围未公开 |
| DPO | reference-free beta1，592位独立Bernoulli；稳定marginΣ(chosen−rejected)logits | beta1/reference-free对齐；完整联合动作概率未知 |
| 配对 | Joint best/worst各1；Layer-only最小2/最大2全4交叉对均值 | 是披露的未公开项选择，非已确认作者规则 |
| 优化 | AdamW lr1e-4、(.9,.999)、eps1e-8、wd0、constantLR，FP32router，Joint3epoch、Layer1epoch | 作者完整优化配置未知 |
| Batch | 默认全局4；对照16用4个microbatch4累积，最后8输入正常更新 | 论文4×3090/batch4的全局或每卡含义未公开；不是4卡DDP。global4每epoch1250更新，global16为313，数据暴露相同但更新数不同 |
| 选优 | 额外256无标签图，与训练及识别出的benchmark图ID无交集；8候选中实际路由Full-KL最小 | 非作者已公开验证流程；测试分数不用于选优。测试曾用于误差分析，不是完全盲测声明 |
| 推理 | 原始问题含typo，无POPE后缀；vicuna_v1、pad、FP16、greedy1beam、max128/cache | 与保存的FastV/PAR/LLaVolta入口相符 |
| 环境 | Torch2.7.1cu128、TF4.37.2、实际SDPA；RTX PRO6000 | 相对作者3090及公开旧Torch/manual算子有偏差。运行时核验实际backend，不沿用旧eager请求元数据 |

POPE固定原始adversarial3000、popular3000、random2910，共8910，macro F1与macro accuracy都报告。LLaVA与LLaVolta均指向这一POPE历史commit；PAR精确测试文件未知。已知现代9000条release差别不足以单独解释近10点差距，不能把版本数不同当作确定原因。

SEED全部14233个image问题，保留题型3/9重复qid49914的两条记录，使用复合(category,qid)顺序；生成文本strip后与标签严格相等，不归一化为首字母。原始图像下载源为作者固定revision的ZIP；旧实验使用固定lmms-lab缓存，其与原图的完整比对状态见迁移验收报告。

本轮受控实验只比较mask和有效batch；固定原5000输入/候选，未搜索架构、改变公开预算、增加辅助loss或按测试分数选更接近论文的模型。单一训练初始化的结果不代表多seed稳定性。
