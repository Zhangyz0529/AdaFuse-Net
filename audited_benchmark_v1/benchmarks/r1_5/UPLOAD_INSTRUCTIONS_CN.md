# 如何使用本上传包

1. 将总ZIP下载到本地，完整解压。总ZIP只用于传输，不上传为普通Git文件。
2. `repository_files/`的内容对应GitHub仓库根目录。先在新分支中合并；这些是建议发布文件，保留原仓库历史。不要上传本地validation_artifacts目录或构建日志。
3. 检查benchmarks/r1_5/results/completeness.csv和verification/，如实保留未完成/未验证范围。特别是MetaTraffic仍可能未完成全部运行。
4. 创建新版本Release，把release_assets/里的所有文件作为附件上传。单附件控制在1500MiB以内；.part文件按artifact_index的reassemble_as名称拼接。
5. 将artifact_index.csv中的TO_BE_FILLED_AFTER_UPLOAD替换成真实Release附件地址，再同步仓库中的索引。原始PCAP不包含在本包。
6. SHA256SUMS校验每个附件。本包没有替你上传或发布任何内容。第三方代码和原始数据的分发范围请按对应许可说明处理。
7. 基线权重不包含在首版附件；所有已选完成运行的指标/配置/训练计划已提供，不能宣称所有基线均可加载已发布权重复核。
8. 表格名称目前是新主表的逻辑标识，正式投稿前替换成最终论文表号。
