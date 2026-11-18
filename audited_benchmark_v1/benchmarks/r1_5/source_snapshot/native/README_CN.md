# 新版原生机制基线：审核后分区的few-shot适配

本目录与历史近似基线分开；不能称为原论文表值的直接复现。

## 实现与证据

- MFF：根据原论文Table5重实现1D SE-ResNet18、AE与分类MLP。共同512bytes及25/77/29维统计输入适配原1024bytes/52维；CE+0.1重建MSE，0.1为本轮预定实现选择。冻结编码器后用支持集拟合二类softmax，不改成ProtoNet。
- GP-VAE：固定Open-Detect作者代码commit，直接调用其ResNet18编码/解码与Gaussian prototype、重建/KL/熵/判别损失。共同512bytes补零到32x32；支持集拟合Gaussian均值，按KL分类。
- FlowPic：从原PCAP逐包核对raw_sha256后提取IP长度与时间，构建1500x1500直方图；按官方notebook移植CNN，保留分类CE。共享审核样本边界，不另做原版60秒窗口和时长筛选；这是明确的session-histogram适配。零时长样本时间bin0，超过原直方图长度范围的包记录排除数量。
- MetaTraffic：直接使用作者MASF及DFNet5000，启用IMP/ATS/AUG。输入原包带方向IP长度，Train-only合成长度比例1/.9/1.1作为三个训练域；不是已测量的独立网络环境。100外层步包含多次原生内/元更新，不能等同于100次单一Adam更新。
- Trafficformer-Light：作者UER embedding/Transformer/MLM+MSP模块，2层hidden128/head4轻量配置；只用Train做前50episode预训练、后50episode分类微调，不使用可能接触目标数据的外部预训练权重。源包方向run提供burst边界，保留官方5类MSP语义；短样本不存在的相邻burst不伪造。
- Pre-Contrast：按用户回复“不等待这一项”跳过。未找到可核实作者实现和可读取原文，不用本地无预训练版本冒充原方法。

完整设置：ADAPTATION_PROTOCOL.json（MFF/GP-VAE）、ADAPTATION_PROTOCOL_V2.json（FlowPic/MetaTraffic/Trafficformer-Light）。VENDOR_VERSIONS.json记录作者仓库和commit。保留源码哈希、接口检查、正式checkpoint和逐episode预测。

## 范围与公平性

四shot、5种子和共享Val/Test评估计划沿用审核后新实验。固定预算及原方法合理默认配置，没有重新执行Optuna；不宣称充分调优最优性能。所有适配、不同输入模态和额外训练机制需在正文/补充明确。

FlowPic和MetaTraffic的原生输入不因字节替换IBP30变化，所以只报告Clean；IBP30列留空，不填为鲁棒性优势。MFF、GP-VAE、Trafficformer-Light沿用同一字节扰动实现。

所有新模型先通过smoke检查，再由统一scheduler启动正式任务；smoke不进入结果汇总。每GPU最多2个任务，CPU准备与训练重叠；训练仍可能短暂等待I/O，不承诺GPU利用率始终100%。优先USTC/ISCX；CIC_IDS随后，CIC_DDOS/CIC_DOH必须正式审核通过后才启动。

## 查看

统一tmux：r1_5_audited_benchmark_v1。队列在相邻R1_5_audited_benchmark_v1/QUEUE_STATUS.json，正式新基线输出在本目录runs/，检查在smoke/，日志在相邻旧队列logs/和failures/。

cache/严格缓存已冻结训练与评估计划实际引用的输入，逐字节核对源数组；不重新划分、不按成绩选样。原AdaFuse及RF结果不重写。

aggregate.py联合读取两目录已COMPLETE的正式任务，输出tables/逐任务逐种子结果、drop、均值/样本SD及完成数量。未全量完成前标记partial，不得发布完整排名。版本差异保留implementation_version，不混入历史结果。

## 后续执行记录

五个新增模型均已通过对应端到端检查并开始正式排队。MetaTraffic未修改作者仓库；原始smoke非有限更新失败保留，私有版本对KL概率比采用clamp后的log差，详见META_NUMERICAL_PATCH_V3.json及ADAPTATION_PROTOCOL_V3.json。原语义与三模块保持，明确标注数值补丁与合成训练域。版本2配置中的models_sha256字段指向共享models.py；完整实际扩展代码身份由相邻CODE_SHA256.json冻结models_v2.py/worker_v2.py记录。版本3已准确记录其模型文件。

MetaTraffic后续完整训练发现V3第26步非有限更新：已退役V3，样本结果不进入正式汇总，文件保存在retired_meta_v3/，原失败日志仍在相邻队列failures/。V4先使用float64并启用原clip_value=2做100外层步完整检查，全部完成后才允许正式排队。当前状态以CURRENT_STATUS.json为准，不把此前短检查PASS解释为完整训练稳定。详见META_V3_RETIREMENT.json及ADAPTATION_PROTOCOL_V4.json。

原生FlowPic/MetaTraffic的多对一输入表示另有audits/逐折特征重合统计：它们与源会话交集是不同问题，不能把原联合输入hash为零概括成所有模型原生输入hash均为零。

原生结构移植仍有明确的初始化差异：GP-VAE采用作者模型构造器默认初始化，未额外调用其训练脚本全局weight_init；FlowPic采用PyTorch层默认初始化，未声称与Keras初始权重逐项相同。所有这些结果均为记录清楚的few-shot适配版本，不是作者原表数值复现。原生准备、审核与表格脚本身份见PREPARATION_AND_REPORT_CODE_SHA256.json。

CIC-IDS后续折由cache_fullplans_v2.py使用Train拟合ID和分组清单独立准备，不依赖AdaFuse先启动；已在ISCX验证全部100步训练ID和顺序与原计划精确一致，证据TRAINING_PLAN_EQUIVALENCE_V2.json。新增生成器不重新划分，也不使用模型分数。


## Pre-Contrast-Local：用户授权的原项目版本
采用原项目 Abation_copy/compare/precontrast_baseline.py 和 universal_model.py，以及原 runner 的监督对比损失。保留 support 原型及平方欧氏距离分类规则，无作者预训练；不能作为作者官方 Pre-Contrast 复现。原先暂缓该项的状态已被用户最新授权替代。仅将缓存 uint8 转为 int64 以兼容原编码器，字节值不变。使用新版审核划分从头训练，正式结果单独保存在 runs/<dataset>/Pre-Contrast-Local，先通过端到端检查再由队列启动；每卡最多两个任务。
