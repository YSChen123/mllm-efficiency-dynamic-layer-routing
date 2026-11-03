# 训练管理代码独立审查

最终补充（2026-10-28 00:49 CST）：正式首试的 no_sync 冲突已通过上游兼容条件修复。新增2项回归后，本地全11项通过。重新以32条真实样本、micro2/accum4/world2执行 clean smoke：两步保存81,936,189,661 bytes完整checkpoint，实际resume至step3，模型/Adam/LR/RNG及live保存回调全部通过，portable导出独立重载通过。两rank fresh/resume均exit0，本轮无审计修补。下文早期失败及accum1记录保留以供追溯。

2026-10-28：按 executing-plans / requesting-code-review 技能做一次 fresh-context 全套审查；reviewer `training_final_review`，只读且未启动远程任务。审查时本地9项单元测试通过。

| 发现 | 处理及证据 |
|---|---|
| 旧 Trainer bare torch.load 与 Torch2.7 NumPy RNG 冲突 | CPU最小复现失败；rng_compat.py 对自有状态显式weights_only=False并精确恢复；回归通过；实际续训逐项断言待记录 |
| 官方旧 Trainer 的 epoch尾部82条 | 保留全量JSON及官方循环，不静默修改累积算法/进入下一epoch；CONFIG_AUDIT如实写预计5197steps和呈现数量，不声称每行都参与更新 |
| smoke 未覆盖最终portable导出/重载 | 已加入真实gather导出、独立from_pretrained重载和权重一致性门禁；尚待实测结果 |
| LR scheduler 未显式比对 | 保存并比对Trainer持有的scheduler state；CPU接口测试已通过 |

最终验证：真实smoke续训2→3成功，两rank exit0；模型/Adam/LR/RNG一致性断言通过，修复后的live on_save回调成功；portable真实导出/独立重载及参数一致性通过。SMOKE_PASSED门禁已生成。9项本地数据/门禁测试、1项远程RNG回归及2项远程审计接口回归均通过。

运行中额外发现：DeepSpeed保存后清空底层optimizer param_groups，直接state_dict做审计会KeyError；改为只读raw optimizer.state采样哈希，无状态修改，CPU回归通过。第一次checkpoint保存后审计失败，磁盘模型/优化器/RNG分片本身已写全；独立读取两rank分片、用官方zero_to_fp32 lazy重建两项权重并校验step/world_size后恢复审计sidecars，再通过实际续训对比。此恢复只补充审计记录，没有改写权重/优化器/RNG。
