#!/bin/bash
cd /home/zzu/zyz_workspace
exec env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/zzu/.conda/envs/zyz_env/bin/python -u /home/zzu/zyz_workspace/TNSM/datasets/V_second_DSTA/R1_5_audited_benchmark_v1/scheduler.py >> /home/zzu/zyz_workspace/TNSM/datasets/V_second_DSTA/R1_5_audited_benchmark_v1/scheduler.log 2>&1
