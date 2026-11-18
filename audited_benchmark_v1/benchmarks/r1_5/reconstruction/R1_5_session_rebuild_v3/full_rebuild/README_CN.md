# v3全量来源重建与分区审核

本目录与先前66条固定样例诊断分开。**全量任务正在运行，未产生正式训练结果，也没有启动Optuna。** 处理范围为CIC-IDS的五个来源域和CIC-DDoS的两个来源域，不代表其他三个数据集已重建。

## 已冻结的处理口径

- `PROTOCOL_FROZEN.json`：真实记录接受前冻结的全量规则及代码哈希。先前full-source v1在CSV建索引阶段暂停，未接受记录；原代码按原SHA256验证后保存在`archive_full_source_v1/`。v2执行版本复用每键累计统计，1032组窗口搜索等价性检查通过。这里的执行版本v2不是旧数据重建目录中的v2。
- CSV全量进入候选核查；没有每类5000条、每键100行或50000包的截断。77维统计量保持未缩放，非有限值固定替零并记录，不拟合任何scaler。
- 每域按五元组稳定哈希顺序选前4096个键作为校准锚点范围，所有标签一视同仁。先用方向包数和时长寻找唯一窗口，再核对两种物理字节定义及时间。每个来源域只选择一个字节定义，不按单条记录或标签选择。至少8个有区分力字节锚点、95%一致性；至少10个时钟锚点、95%一致性。证据不足阻断该域正式接受。完整锚点保存在各域`SOURCE_POLICY.json`。
- 时钟采用整分钟偏移，残差必须符合CSV精度区间；CIC-IDS保留12小时表示歧义。亚秒CSV偏移估计容许2微秒数值误差；最终匹配仍按冻结窗口规则。分钟精度残差不等于亚秒校准质量。
- 来源分片按原`source_domain_registry.csv`归域，编号不跨域碰撞。缺少握手保留标记；TCP/UDP分别使用原冻结120秒/30秒空闲规则，RST/双向FIN后2秒内仅保留无SYN、无载荷控制尾包。不称观察段为完整真实连接。
- 1包、2包及超过10000包不因包数自动拒绝。使用磁盘SQLite索引；资源不足会使阶段失败，不能以不完整候选搜索接受记录。碎片未重组会阻断相关IP对的候选搜索，无法归属的截短记录阻断来源。
- CIC-DDoS两个已核验截断文件采用另存的完整记录前缀，原件不变。原文件、前缀hash、保留字节数和末条完整时间在`../sources/PREFIX_PROVENANCE.json`；CSV时间不确定区间及持续时间触及缺失边界的记录拒绝。不是恢复完整原始数据。

## 正式输出及位置

每个来源目录：`outputs/<dataset>/<domain>/`

| 文件 | 内容 |
|---|---|
| `STATUS.json` | 当前阶段、进程、读入量或失败原因 |
| `CSV_COMPLETE.json` | 全量CSV行数、特征顺序、CSV文件hash |
| `PACKETS_COMPLETE.json` | 所有分片实际读取范围、源hash、包数、解析异常及搜索完整性 |
| `SOURCE_POLICY.json` | 该域固定字节定义、所有锚点、时钟参数和残差 |
| `accepted.jsonl` | 每条接受记录及未缩放模型输入、来源、观察会话、缺握手/尾包/少包标记 |
| `accepted_packet_membership.jsonl` | 每条接受记录的CSV行→原报文文件/索引/偏移/hash→观察会话 |
| `parse_rejections.jsonl` | 解析、字段及有效性阶段的逐条拒绝 |
| `matching_rejections.jsonl` | 范围、来源证据、匹配、会话及覆盖边界的逐条拒绝 |
| `session_relations.jsonl` | 不确定中途延续的必须同组关联 |
| `REBUILD_COMPLETE.json` | 数量闭合、每标签观察会话和保守来源组数、标记统计 |

`reports/all_sources_labels_decisions.csv`统一合并解析拒绝及匹配拒绝，避免漏掉解析阶段。**未完成来源没有结果行，不能解释为零。** `reports/source_domain_policies.csv`逐域列出口径；尚未核定时标PENDING。

## 分区准备与阻断规则

`PARTITION_PROTOCOL_FROZEN.json`冻结后续规则。守护进程等待同一数据集全部来源完成后运行`prepare_partitions.py`：

1. 所有跨标签相同模型输入组全部剔除，公布每条排除记录；其余按会话、来源关联、相同输入、相同原始包内容和中途延续关联形成连通分组。
2. 输出`class_group_feasibility.csv`。每类至少5个组件且至少150条是初步门槛；最终还须固定划分后的各角色每类至少50条、支持池至少10条、query池至少15条。未达标阻断；门槛不是性能或方差保证。
3. 沿用既有60/20/20组划分比例、种子42，验证/测试进一步按组件划支持/query池。每折Test为一个held-out攻击，Val为下一个攻击，其他攻击Train；BENIGN同样分组隔离。比例针对组，不能保证样本数比例。
4. 生成`partitions/<dataset>/split_manifest_v3.csv`；审核所有分区对及支持/query池的ID、会话、来源、组件、输入内容和原始内容交集。
5. 每折scaler仅fit该折Train，保存fit样本ID和scaler；检查float32缩放后新出现的跨分区/池内容碰撞。
6. 每个shot/任务生成100个评价episode，记录实际支持/query ID；本流程不使用校准池、不拟合判别阈值。训练episode、训练预算、基线配置和后续模型选择仍需正式实验协议，不能把准备阶段写成训练已复现。

审核报告为`partitions/<dataset>/leakage_audit_report.txt`及`PARTITION_GATE.json`。只有真实结果PASS才称相应审核通过；合成检查不替代真实审核。任何失败保留报告，不能自动重试种子或拆组。

## 查看进度

```bash
cat RUN_STATUS.json
cat PARTITION_WATCH_STATUS.json
cat reports/LIVE_STATUS.json
tail -n 10 execution/CIC_IDS_Friday.log
tail -n 10 execution/CIC_DDOS_Test_Day_03-11.log
```

重算报告：`/home/zzu/.conda/envs/zyz_env/bin/python report.py`。守护进程只重建、审核、报告，不调用模型训练或Optuna。源数据及历史v2未改写，先前冻结的44项v3证据亦保持不变。
