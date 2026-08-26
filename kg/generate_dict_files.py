# -*- coding: utf-8 -*-
"""
从machine_fault.json生成所有必要的词典文件
"""

import json
import os

# 读取数据文件
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "machine_fault.json")
dict_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dict")

# 确保dict目录存在
os.makedirs(dict_dir, exist_ok=True)

# 读取所有故障数据
faults = []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            faults.append(json.loads(line.strip()))

print(f"读取了 {len(faults)} 条故障记录")

# 提取所有唯一值
fault_types = set()
phenomena = set()
components = set()
solutions = set()
checks = set()
causes = set()
prevents = set()
parameters = set()
cure_ways = set()
categories = set()
materials = set()

for fault in faults:
    # 故障类型
    if 'name' in fault:
        fault_types.add(fault['name'])
    
    # 症状/现象
    if 'symptom' in fault and isinstance(fault['symptom'], list):
        for item in fault['symptom']:
            phenomena.add(item)
    
    # 部件
    if 'component' in fault and isinstance(fault['component'], list):
        for item in fault['component']:
            components.add(item)
    
    # 解决方案
    if 'solution' in fault and isinstance(fault['solution'], list):
        for item in fault['solution']:
            solutions.add(item)
    
    # 检查方法
    if 'check' in fault and isinstance(fault['check'], list):
        for item in fault['check']:
            checks.add(item)
    
    # 原因（兼容列表/字符串两种格式）
    if 'cause' in fault:
        cause_data = fault['cause']
        if isinstance(cause_data, list):
            for cause in cause_data:
                cause = str(cause).strip()
                if cause and len(cause) > 2:
                    causes.add(cause)
        elif isinstance(cause_data, str) and cause_data.strip():
            import re
            cause_list = re.split(r'\d+\.', cause_data)
            for cause in cause_list:
                cause = cause.strip()
                if cause and len(cause) > 2:
                    causes.add(cause)
    
    # 预防方法（兼容列表/字符串两种格式）
    if 'prevent' in fault:
        prevent_data = fault['prevent']
        if isinstance(prevent_data, list):
            for prevent in prevent_data:
                prevent = str(prevent).strip()
                if prevent and len(prevent) > 2:
                    prevents.add(prevent)
        elif isinstance(prevent_data, str) and prevent_data.strip():
            prevent_list = re.split(r'\d+\.', prevent_data)
            for prevent in prevent_list:
                prevent = prevent.strip()
                if prevent and len(prevent) > 2:
                    prevents.add(prevent)
    
    # 参数（兼容列表套字典/字符串两种格式）
    if 'parameter' in fault:
        param_data = fault['parameter']
        if isinstance(param_data, list):
            for param in param_data:
                if isinstance(param, dict):
                    param_name = param.get('param_name', '').strip()
                    if param_name:
                        parameters.add(param_name)
        elif isinstance(param_data, str) and param_data.strip():
            parameters.add(param_data.strip())
    
    # 治疗方法
    if 'cure_way' in fault and isinstance(fault['cure_way'], list):
        for item in fault['cure_way']:
            cure_ways.add(item)
    
    # 类别
    if 'category' in fault and isinstance(fault['category'], list):
        for item in fault['category']:
            categories.add(item)
    
    # 材料
    if 'material' in fault and isinstance(fault['material'], list):
        for item in fault['material']:
            materials.add(item)

# 写入词典文件
dict_files = {
    'fault_type.txt': fault_types,
    'phenomenon.txt': phenomena,
    'component.txt': components,
    'solution.txt': solutions,
    'detection.txt': checks,
    'cause.txt': causes,
    'prevent.txt': prevents,
    'parameter.txt': parameters,
    'cure_way.txt': cure_ways,
    'category.txt': categories,
    'material.txt': materials,
}

for filename, data_set in dict_files.items():
    filepath = os.path.join(dict_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in sorted(data_set):
            if item and isinstance(item, str):
                f.write(item + '\n')
    print(f" created {filename}: {len(data_set)} entries")

# 创建deny.txt（否定词）
deny_path = os.path.join(dict_dir, 'deny.txt')
if not os.path.exists(deny_path):
    deny_words = ['不', '否', '无', '没有', '非', '不是', '不对', '不可以', '不能']
    with open(deny_path, 'w', encoding='utf-8') as f:
        for word in deny_words:
            f.write(word + '\n')
    print(f" created deny.txt: {len(deny_words)} entries")

print("\n所有词典文件生成完成！")
