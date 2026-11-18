# 审核后新基准：独立于历史结果

本目录只针对新版会话/内容关联组隔离分区。不要与历史PKL或历史表混合平均。

## 启动范围

- AdaFuse-Net：发布包FactorialNet，全模型CE+SCL+Polar+Dec，干净训练。
- RandomForest：原生sklearn森林，每个episode只用带标签support拟合，不改成原型分类器。无需在旧攻击标签上预训练；其种子是森林估计器随机重复，与深度模型训练重复分开解释。
- MFF、GP-VAE、MetaTraffic、Pre-Contrast、Trafficformer-Light、FlowPic：列入待审核范围，暂不训练；已有本地适配器不能直接视为原论文方法复现。不会将近似基线结果伪装成上述原方法。

## 固定协议

详见PROTOCOL.json。5种子42/3407/2026/123/5678，四shot1/3/5/10，各折Val/Test各100episodes，每类15query。主要结果Clean；IBP30次要结果，使用发布包逐字节独立替换概率0.3的输入空间算子，不是合法PCAP修改或连续块插入。只扰动query序列，support和统计特征保持干净。

固定发布默认配置，训练预算从默认50预先改为100updates，末步checkpoint。没有执行或宣称50次Optuna；没有宣称各模型分别充分调优。Val结果完整保留，但不选择最后checkpoint。这里是固定配置比较，而不是最优性能搜索。

输入保留重建的匿名化定义，不额外对统计前四列或序列前54字节随意清零。各折仅使用已有Train-fitted scaler。训练support/query按每类split_group_id分池；同一训练计划用于所有训练种子。测试使用冻结evaluation_episodes.json，不重新抽样、不根据结果更改划分。

## 排队

tmux会话 `r1_5_audited_benchmark_v1`，运行scheduler.py。每GPU最多2任务，且空闲显存不足9000MiB时不启动新的任务；CPU森林最多2任务。实际并发可以低于上限。失败记录到failures/，不自动重试、不改种子取代失败。

按USTC、ISCX、CIC_IDS、CIC_DDOS、CIC_DOH检查数据。只有正式PARTITION_GATE为PASS_PARTITIONS_AND_EVALUATION_POOLS且manifest存在才启动。数据准备报告的training_allowed=false保留不改；新的训练协议和训练episode审计由worker实际执行。

CODE_SHA256.json冻结工作器、队列、协议及模型/损失/扰动实现；源码变化时停止调度。现有重建任务不受此队列控制，不停止数据重建。

## 输出

- QUEUE_STATUS.json：当前任务、GPU、PID、等待原因。
- launches/、logs/、failures/：逐任务命令与状态。
- runs/<dataset>/<model>/seed*/round*/shot*/：有效配置、训练episode、训练轨迹、checkpoint（RF为逐episode森林）、Val/Test预测及episode级指标、完成标记。
- smoke/：2步训练与2episodes的接口检查，绝不进入正式汇总。
- aggregate.py：只汇总有COMPLETE.json的正式任务，生成逐任务/逐种子/条件绝对F1、drop、均值及样本SD、匹配差。未完成的结果必须标记partial，不能声称完整排名。

查看：`tmux attach -t r1_5_audited_benchmark_v1`；`cat QUEUE_STATUS.json`。

用户后续资源调整见RESOURCE_LIMIT_OVERRIDE.json：每卡上限2；训练协议不变，已有任务接管而不重训。

## 新增原生基线扩展

此前“6个深度基线待审核”是首次启动时状态，后续已有5个原生机制few-shot适配通过检查并接入本队列，最新版本/范围请查看相邻R1_5_native_baselines_v1/README_CN.md和本目录MODEL_READINESS.json。Pre-Contrast按用户指示跳过。统一完整结果汇总在新目录tables/，本目录tables/仍仅覆盖初始AdaFuse/RF。
