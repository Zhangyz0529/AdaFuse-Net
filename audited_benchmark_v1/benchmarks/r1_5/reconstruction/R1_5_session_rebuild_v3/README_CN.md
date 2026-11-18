# R1-5 v3：规则冻结与固定样例核查

**本目录没有模型训练或Optuna任务。正式数据尚未生成。**

已完成：冻结v2失败状态；66个固定记录（44诊断、22留出）和来源追溯；三个PCAP/五个CSV的hash复核；原始包完整相关键扫描与5250条独立报文复核；v3规则冻结；13项边界/失败关闭检查；留出样例核查；CIC-DDoS压缩包与截断记录取证；逐条决策及拒绝日志。

入口：
- `../R1_5_session_rebuild_v2/V2_STATUS.md`：旧试运行冻结，不混淆CIC-IDS Friday的DDoS类别与CIC-DDoS数据集。
- `reports/STEP2_V3_REPORT_CN.md`：完整结论、数量与边界。
- `reports/REVIEW_WITNESSES_CN.md`：固定7个见证，CSV→窗口→连接边界→决策。
- `reports/ALL_66_CSV_PACKET_SESSION_CHAINS.json`：66条完整证据链。
- `V3_RULES_FROZEN.json`：规则、脚本和输入hash；明确禁用训练。
- `rules_v3.py`、`packet_io_v3.py`：v3观察连接、字节计量、匹配和逐原因日志实现。
- `source_domain_registry.csv`、`reports/DOMAIN_TIME_AUDIT.json`：970文件、7来源域及时间范围证据；ID唯一不等于划分独立。
- `diagnostics/V3_CASE_RESULTS.csv`、`diagnostics/V3_DECISIONS.jsonl`：接受/拒绝逐记录结果。
- `diagnostics/V3_VALIDATION_SUMMARY.json`：小规模核查41接受/25拒绝；不是全量接受率或模型结果。
- `diagnostics/CIC_DDOS_TRUNCATION_FORENSICS.json`、`sources/PREFIX_PROVENANCE.json`：本地ZIP成员本身截断的证明，以及未改原文件的完整记录前缀。
- `reports/PORTSCAN_FILTER_IMPACT.json`：原5000条PortScan预选中4958条被少于3包规则排除；不应归因于Session隔离。

v3只允许有据可查的来源域字节定义。诊断组发现Friday/Wednesday的部分记录需计入Ethernet填充，Tuesday符合IP内载荷；未按单条记录或标签选择口径，未降低duration/唯一性标准。全来源仍需按预定、标签无关锚点复核，证据不足或混合则停止。CIC-DDoS字节口径未验证，不能接受正式记录。

关闭宽限为RST/双FIN后2秒内的无SYN、无载荷控制尾包；新SYN或新载荷不被吞并。不确定中途延续必须形成整体分组关系。缺少握手保留标记，观察段不等于完整连接真值。没有3包下限或10000包上限作为有效性判据；资源不足应完整落盘/分块或停止，不允许部分搜索后接受。

下一阶段尚未完成：全来源的字节/时钟证据、完整逐行拒绝计数、剩余文件内容校验、CIC-DDoS可用覆盖范围/字节口径审核、正式重建、分区与冲突内容连通分组审核。当前不得宣称第2步全数据重建结项，更不能启动模型调参。
