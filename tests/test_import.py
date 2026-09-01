import time

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:.1f}s] {msg}", flush=True)

log("开始逐步导入测试...")

log("1. import backend")
import backend

log("2. from backend import app")
from backend import app

log(" ALL IMPORTS OK!")
