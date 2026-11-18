import subprocess,time,sys
from pathlib import Path
r=Path(__file__).resolve().parent
while True:
 subprocess.run([sys.executable,str(r/"aggregate.py")],check=False)
 time.sleep(60)
