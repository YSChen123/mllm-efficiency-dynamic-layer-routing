# 可迁移验收

2026-11-08工程验收通过。验收环境与工作目录独立，不共享旧项目site-packages；发布包运行不读取旧任务控制/验收JSON或私有Arrow索引。实际为同一物理服务器上的新venv和新目录，尚未在第二台物理服务器或另一GPU架构上全量验证。

验收包括：CPU单元测试；真实原图到预处理prefix/VQ、Full响应IDs、QK logits、五动作、KL、best/worst标签；至少一次真实optimizer更新；checkpoint恢复下一步；真实生成、预算及cache；官方SEED原图与旧缓存的完整对应检查。

同一原始权重和原始数据文件可在多个工作目录复用；这不等于依赖旧代码或旧训练状态。新服务器使用README的固定公开下载源和配置路径即可创建同样资源。

| 验收 | 结果 | 证据 |
|---|---|---|
| 独立环境 | Python3.10.22、Torch2.7.1cu128、TF4.37.2、Pillow10.2.0；pip check通过；无旧site-packages | evidence/acceptance.json |
| 单元与回归 | 40项通过，包括早期中断step0恢复、optimizer下一步逐位一致、并发身份文件、换路径与换图像区分、真实token/cache长度 | evidence/acceptance.json |
| 原始输入端到端 | 4训练/4验证输入，偏好→4更新→重载→POPE6/SEED4及独立验收完成；SMOKE_ONLY | evidence/acceptance.json |
| 偏好数值 | 4训练+4验证的V/Q、Full响应、QK、5动作、标签一致；全部KL差0；双向/因果单次optimizer更新全权重逐位一致 | evidence/numerical_parity.json |
| 完整训练引擎 | 5000输入、3epochs、3750步；epoch1/3全部router权重与原实现逐位一致 | evidence/full_training_engine_parity.json |
| 正式权重迁移 | Layer-only/Joint各POPE6+SEED4，共20固定案例；prompt、logits、动作与完整回答均一致 | evidence/reference_inference_parity.json |
| SEED原图 | 作者ZIP通过官方SHA256，全部14233条与旧缓存的解码RGB一致，差异0 | evidence/seed_image_parity.json |
| 实际图像交集 | 固定5000训练图、256验证图与官方SEED原图解码像素交集均0 | evidence/seed_image_overlap.json |
| 精确Git树 | 将已提交代码导出到空目录后40项测试、依赖锁dry-run、shell语法检查通过；导入路径来自新目录 | evidence/git_export_verification.json |

完整训练引擎的5000输入测试使用了**明确转换的、已认证旧偏好/特征**，不是通过新包重新生成全部5000条原始偏好；原始输入重建的直接逐项核对范围为上述8条。此范围分开标记，不能将3750步的引擎等价测试冒充全量原始数据从头重建。新服务器从头运行时，`run`仍会生成全部5000/256条偏好，固定样本ID与5256张图像指纹全部校验。

模型/数据文件允许更换存放路径；科学设置与实际资源校验值分别核对。评测开始与恢复前绑定完整标注、按顺序的图像文件SHA和模型身份，拒绝同名换图像后混用旧预测。只有全量记录、identity和hash通过才计正式成绩；SMOKE_ONLY和诊断子集不能混报。

依赖的实际冻结版本见 `requirements-lock-cu128.txt`；安装脚本优先安装此锁文件。`acceptance.json`记录实现与证据哈希，可与当前代码的 `par_repro.io.code_sha()` 对照。代码安装支持Git checkout+editable方式，未提供独立wheel分发。

精确Git树的代码验收commit为efcd342；之后的更新仅加入结果、证据和说明，数值实现hash保持 `f16e2696dcb53cdb66a98ac429c0074eded7ec6d8cc887a0c83384caab66edff`。
