"""
机床故障诊断系统 - 主运行文件
============================================
功能：
1. 启动 FastAPI 后端服务器
2. 提供 Web 前端界面
3. 整合知识图谱（KG）+ RAG + 联网搜索
4. 意图识别自动路由到最佳数据源

使用方法：
    python main.py  # 启动完整系统

然后访问: <ADDRESS_REMOVED>
"""
"""
Machine Fault Diagnosis System (Main Entry)

Dependencies:
    - Python >= 3.9
    - Neo4j
    - Milvus
    - FastAPI
"""
import uvicorn
from backend.app import app
import threading
import time
import webbrowser
import subprocess
import os
import sys

def kill_port_process(port: int):
    """
    自动终止占用指定端口的进程（Windows）
    """
    try:
        # 查找占用端口的进程
        cmd_find = f'netstat -ano | findstr ":{port}"'
        result = subprocess.run(cmd_find, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0 or not result.stdout.strip():
            return True  # 端口未被占用
        
        # 提取 PID（最后一列）
        lines = result.stdout.strip().split("\n")
        pids = set()
        for line in lines:
            parts = line.split()
            if parts:
                pid = parts[-1]
                if pid.isdigit():
                    pids.add(pid)
        
        if not pids:
            return True
        
        # 终止所有相关进程
        for pid in pids:
            try:
                subprocess.run(f"taskkill /PID {pid} /F", shell=True, 
                             capture_output=True, text=True, timeout=5)
                print(f"  [清理] 已终止占用端口 {port} 的进程 (PID: {pid})")
            except Exception as e:
                print(f"  [警告] 无法终止进程 {pid}: {e}")
        
        time.sleep(1)  # 等待端口释放
        return True
    except Exception as e:
        print(f"  [警告] 端口清理失败: {e}")
        return False


def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    """
    查找可用端口
    """
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("0.0.0.0", port))
            sock.close()
            return port
        except OSError:
            continue
    
    raise RuntimeError(f"无法找到可用端口 (从 {start_port} 开始)")


def open_browser():
    """延迟打开浏览器"""
    time.sleep(2)  # 等待服务器启动
    webbrowser.open('http://localhost:8000')


if __name__ == "__main__":
    print("=" * 70)
    print("  机床故障诊断系统 - 启动中...")
    print("  Machine Fault Diagnosis System - Starting...")
    print("=" * 70)
    print()
    
    # 自动清理端口 8000
    print("  [1/3] 检查端口状态...")
    kill_port_process(8000)
    
    # 验证端口是否可用
    print("  [2/3] 验证端口可用性...")
    try:
        available_port = find_available_port(8000, 1)
        if available_port != 8000:
            print(f"  [警告] 端口 8000 不可用，切换到端口 {available_port}")
    except RuntimeError as e:
        print(f"  [错误] {e}")
        sys.exit(1)
    
    print("  [3/3] 启动服务器...")
    print()
    print("功能模块：")
    print("  [+] 知识图谱问答 (Neo4j)")
    print("  [+] RAG 文档问答 (Milvus)")
    print("  [+] 联网搜索 (Tavily)")
    print("  [+] 意图识别 (自动路由)")
    print()

    
    # 在新线程中打开浏览器
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # 启动 FastAPI 服务器
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
