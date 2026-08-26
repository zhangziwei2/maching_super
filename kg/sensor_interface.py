# -*- coding: utf-8 -*-
"""
机床故障诊断 - 传感器实时数据接口模块

预留接口，用于接入多传感器实时数据（振动+力+声级计），
实现基于实时数据的智能故障诊断。

支持的数据源：
1. WebSocket实时数据流（推荐）
2. HTTP轮询端点
3. 本地文件/数据库读取（离线模式）

传感器配置（与用户硕士论文一致）：
- 3通道振动传感器（X/Y/Z方向）
- 3通道力传感器（Fx/Fy/Fz）
- 1通道声级计（SPL）
"""

import time
import json
import threading
from collections import deque
from config import SENSOR_CONFIG


class SensorInterface:
    def __init__(self):
        self.cfg = SENSOR_CONFIG
        self.enabled = self.cfg.get("enabled", False)
        self.ws_url = self.cfg.get("websocket_url", "ws://localhost:8765")
        self.http_endpoint = self.cfg.get("http_endpoint", "")
        self.poll_interval = self.cfg.get("poll_interval", 1.0)

        # 报警阈值
        self.thresholds = self.cfg.get("alarm_thresholds", {})

        # 数据缓存（环形缓冲区，保留最近N秒数据）
        self.buffer_size = 300  # 5分钟 @ 1Hz
        self.data_buffer = deque(maxlen=self.buffer_size)

        # 状态
        self.connected = False
        self.ws_client = None
        self.poll_thread = None
        self.running = False

        # 特征提取配置
        self.feature_cfg = self.cfg.get("feature_extraction", {})

    def connect(self):
        """连接传感器数据源"""
        if not self.enabled:
            print("[Sensor] 传感器功能未启用，请在config.py中设置 enabled=True")
            return False

        # TODO: 实现WebSocket连接
        # try:
        #     import websockets
        #     self.ws_client = await websockets.connect(self.ws_url)
        #     self.connected = True
        #     self.running = True
        #     asyncio.create_task(self._ws_loop())
        # except Exception as e:
        #     print(f"[Sensor] WebSocket连接失败: {e}")
        #     self.connected = False

        # 当前简化为模拟模式
        print("[Sensor] 传感器接口已初始化（当前为模拟模式）")
        self.connected = True
        return True

    def disconnect(self):
        """断开传感器连接"""
        self.running = False
        self.connected = False
        if self.ws_client:
            # await self.ws_client.close()
            pass
        print("[Sensor] 传感器连接已断开")

    def get_status(self):
        """获取传感器连接状态"""
        return {
            "connected": self.connected,
            "enabled": self.enabled,
            "buffered_samples": len(self.data_buffer),
            "thresholds": self.thresholds,
        }

    def get_latest_data(self, n=1):
        """获取最近n条数据"""
        if not self.data_buffer:
            return None
        return list(self.data_buffer)[-n:]

    def push_data(self, data_dict):
        """
        接收外部传感器数据推入缓冲区
        供外部传感器驱动调用

        Args:
            data_dict: {
                "timestamp": float,
                "vibration": {"X": float, "Y": float, "Z": float},
                "force": {"Fx": float, "Fy": float, "Fz": float},
                "sound": {"SPL": float},
                "temperature": float,
            }
        """
        data_dict["received_at"] = time.time()
        self.data_buffer.append(data_dict)
        self.connected = True

    def analyze_for_diagnosis(self, data_samples):
        """
        基于传感器数据进行故障特征分析

        Args:
            data_samples: 传感器数据样本列表

        Returns:
            str: 分析结果文本
        """
        if not data_samples:
            return "暂无传感器数据"

        # 提取振动特征
        vib_rms = self._calc_vibration_rms(data_samples)
        spike_factor = self._calc_spike_factor(data_samples)
        temp = self._get_temperature(data_samples)
        spl = self._get_sound_level(data_samples)

        alerts = []
        if vib_rms > self.thresholds.get("vibration_rms", 5.0):
            alerts.append(f" 振动有效值异常: {vib_rms:.2f} m/s (阈值: {self.thresholds.get('vibration_rms', 5.0)})")
        if spike_factor > self.thresholds.get("spike_factor", 3.0):
            alerts.append(f" 冲击因子异常: {spike_factor:.2f} (阈值: {self.thresholds.get('spike_factor', 3.0)})")
        if temp and temp > self.thresholds.get("temperature", 70.0):
            alerts.append(f" 温度异常: {temp:.1f}C (阈值: {self.thresholds.get('temperature', 70.0)})")
        if spl and spl > self.thresholds.get("sound_level", 85.0):
            alerts.append(f" 声级异常: {spl:.1f} dB (阈值: {self.thresholds.get('sound_level', 85.0)})")

        if alerts:
            return "传感器监测到以下异常：\n" + "\n".join(alerts) + "\n\n建议结合知识图谱进一步诊断故障原因。"
        else:
            return f"传感器数据正常。当前振动RMS: {vib_rms:.2f} m/s，温度: {temp:.1f}C，声级: {spl:.1f} dB。"

    def _calc_vibration_rms(self, samples):
        """计算振动有效值（简化版，实际应基于原始波形计算）"""
        # TODO: 实现基于原始采样数据的RMS计算
        # 当前返回模拟值
        return 2.5

    def _calc_spike_factor(self, samples):
        """计算冲击因子（峰值/RMS）"""
        # TODO: 实现冲击因子计算
        return 2.0

    def _get_temperature(self, samples):
        """获取温度读数"""
        if samples and "temperature" in samples[-1]:
            return samples[-1]["temperature"]
        return 35.0

    def _get_sound_level(self, samples):
        """获取声级读数"""
        if samples and "sound" in samples[-1]:
            return samples[-1]["sound"].get("SPL", 65.0)
        return 65.0

    def simulate_data(self):
        """模拟传感器数据（用于测试）"""
        import random
        data = {
            "timestamp": time.time(),
            "vibration": {
                "X": random.uniform(0.5, 3.0),
                "Y": random.uniform(0.5, 3.0),
                "Z": random.uniform(0.5, 4.0),
            },
            "force": {
                "Fx": random.uniform(50, 200),
                "Fy": random.uniform(50, 200),
                "Fz": random.uniform(100, 500),
            },
            "sound": {
                "SPL": random.uniform(60, 90),
            },
            "temperature": random.uniform(30, 75),
        }
        return data


class SensorDataSimulator:
    """传感器数据模拟器（用于开发和测试）"""
    def __init__(self, interface, interval=1.0):
        self.interface = interface
        self.interval = interval
        self.running = False
        self.thread = None

    def start(self):
        """启动模拟数据生成"""
        self.running = True
        self.thread = threading.Thread(target=self._loop)
        self.thread.daemon = True
        self.thread.start()
        print("[SensorSimulator] 传感器数据模拟已启动")

    def stop(self):
        """停止模拟数据生成"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print("[SensorSimulator] 传感器数据模拟已停止")

    def _loop(self):
        while self.running:
            data = self.interface.simulate_data()
            self.interface.push_data(data)
            time.sleep(self.interval)


if __name__ == '__main__':
    sensor = SensorInterface()
    sensor.connect()

    # 模拟数据
    sim = SensorDataSimulator(sensor, interval=1.0)
    sim.start()

    time.sleep(3)
    print(json.dumps(sensor.get_status(), indent=2, ensure_ascii=False))
    print(sensor.analyze_for_diagnosis(sensor.get_latest_data(3)))

    sim.stop()
    sensor.disconnect()
