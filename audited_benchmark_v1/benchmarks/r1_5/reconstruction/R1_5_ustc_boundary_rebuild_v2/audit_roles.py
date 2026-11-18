"""Apply unchanged task-role algorithm to declared available-suffix USTC sources."""
import argparse,sys
from pathlib import Path
role=Path(__file__).resolve().parent.parent/'R1_5_task_role_partitions_v1';sys.path.insert(0,str(role))
from prepare import prepare
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare('USTC',a.source_root,a.output)
