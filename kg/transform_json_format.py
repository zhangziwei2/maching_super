# -*- coding: utf-8 -*-
"""
改造machine_fault.json格式
把cause、prevent从字符串改成数组，parameter改成结构化数组
"""

import json
import re
import os

def split_text_to_items(text):
    """把'1. xxx\n2. yyy'格式的文本拆分成数组"""
    if not text or not isinstance(text, str):
        return []
    
    # 按数字序号分割
    items = re.split(r'\d+\.', text)
    result = []
    for item in items:
        item = item.strip()
        if item and len(item) > 2:
            result.append(item)
    
    return result


def transform_json(input_path, output_path):
    """改造JSON格式"""
    print(" 开始改造JSON格式...")
    
    # 读取原数据
    faults = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                faults.append(json.loads(line.strip()))
    
    print(f"  读取了 {len(faults)} 条记录")
    
    # 改造每条记录
    new_faults = []
    for fault in faults:
        new_fault = fault.copy()
        
        # 1. 改造 cause（原因）- 从字符串改成数组
        if 'cause' in fault and isinstance(fault['cause'], str):
            new_fault['cause'] = split_text_to_items(fault['cause'])
        
        # 2. 改造 prevent（预防）- 从字符串改成数组
        if 'prevent' in fault and isinstance(fault['prevent'], str):
            new_fault['prevent'] = split_text_to_items(fault['prevent'])
        
        # 3. 改造 parameter（参数）- 从字符串改成数组
        # 原格式："切削速度vc: 80-150m/min(钢件); 进给量f: 0.1-0.3mm/r; ..."
        # 新格式：[{"param_name": "切削速度vc", "value": "80-150m/min(钢件)", "adjustment": "降低15%-25%可延长刀具寿命"}, ...]
        if 'parameter' in fault and isinstance(fault['parameter'], str):
            param_text = fault['parameter']
            param_array = []
            
            # 按分号分割
            param_items = param_text.split(';')
            for item in param_items:
                item = item.strip()
                if not item:
                    continue
                
                # 按冒号分割参数名和值
                if ':' in item:
                    parts = item.split(':', 1)
                    param_name = parts[0].strip()
                    param_value = parts[1].strip()
                    
                    param_array.append({
                        "param_name": param_name,
                        "value": param_value,
                        "adjustment": ""  # 留空，后续可手动补充
                    })
                else:
                    # 没有冒号，整个作为一个参数
                    if item:
                        param_array.append({
                            "param_name": item,
                            "value": "",
                            "adjustment": ""
                        })
            
            new_fault['parameter'] = param_array
        
        new_faults.append(new_fault)
    
    # 写入新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for fault in new_faults:
            f.write(json.dumps(fault, ensure_ascii=False) + '\n')
    
    print(f"   改造完成，写入: {output_path}")
    
    # 显示改造示例
    print("\n 改造示例（前3条）:")
    for i, fault in enumerate(new_faults[:3]):
        print(f"\n  记录 {i+1}: {fault['name']}")
        print(f"    cause: {fault.get('cause', [])[:2]}...")  # 只显示前2条
        print(f"    prevent: {fault.get('prevent', [])[:2]}...")
        print(f"    parameter: {fault.get('parameter', [])[:2]}...")


if __name__ == '__main__':
    input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "machine_fault.json")
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "machine_fault_new.json")
    
    print("=" * 70)
    print("  JSON格式改造脚本")
    print("  把cause、prevent改成数组，parameter改成结构化数组")
    print("=" * 70)
    
    transform_json(input_path, output_path)
    
    print("\n 改造完成！")
    print(f"  原文件: {input_path}")
    print(f"  新文件: {output_path}")
    print("\n  请检查新文件格式是否正确，确认后替换原文件。")
