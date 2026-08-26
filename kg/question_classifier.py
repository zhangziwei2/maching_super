# -*- coding: utf-8 -*-
"""
机床故障诊断 - 意图识别/问题分类模块
中文标签版：实体类型使用中文（故障类型、症状、部件等）
"""

import os
import ahocorasick


class QuestionClassifier:
    """意图识别类 - 中文标签版"""

    def __init__(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        # 特征词路径
        self.fault_type_path = os.path.join(cur_dir, 'dict/fault_type.txt')
        self.phenomenon_path = os.path.join(cur_dir, 'dict/phenomenon.txt')
        self.component_path = os.path.join(cur_dir, 'dict/component.txt')
        self.solution_path = os.path.join(cur_dir, 'dict/solution.txt')
        self.detection_path = os.path.join(cur_dir, 'dict/detection.txt')
        self.cause_path = os.path.join(cur_dir, 'dict/cause.txt')
        self.prevent_path = os.path.join(cur_dir, 'dict/prevent.txt')
        self.parameter_path = os.path.join(cur_dir, 'dict/parameter.txt')
        self.cure_way_path = os.path.join(cur_dir, 'dict/cure_way.txt')
        self.category_path = os.path.join(cur_dir, 'dict/category.txt')
        self.material_path = os.path.join(cur_dir, 'dict/material.txt')
        self.deny_path = os.path.join(cur_dir, 'dict/deny.txt')

        # 加载特征词
        self.fault_type_wds = [i.strip() for i in open(self.fault_type_path, encoding="utf-8") if i.strip()]
        self.phenomenon_wds = [i.strip() for i in open(self.phenomenon_path, encoding="utf-8") if i.strip()]
        self.component_wds = [i.strip() for i in open(self.component_path, encoding="utf-8") if i.strip()]
        self.solution_wds = [i.strip() for i in open(self.solution_path, encoding="utf-8") if i.strip()]
        self.detection_wds = [i.strip() for i in open(self.detection_path, encoding="utf-8") if i.strip()]
        self.cause_wds = [i.strip() for i in open(self.cause_path, encoding="utf-8") if i.strip()]
        self.prevent_wds = [i.strip() for i in open(self.prevent_path, encoding="utf-8") if i.strip()]
        self.parameter_wds = [i.strip() for i in open(self.parameter_path, encoding="utf-8") if i.strip()]
        self.cure_way_wds = [i.strip() for i in open(self.cure_way_path, encoding="utf-8") if i.strip()]
        self.category_wds = [i.strip() for i in open(self.category_path, encoding="utf-8") if i.strip()]
        self.material_wds = [i.strip() for i in open(self.material_path, encoding="utf-8") if i.strip()]
        self.deny_wds = [i.strip() for i in open(self.deny_path, encoding="utf-8") if i.strip()]

        # 合并所有领域词
        self.region_words = set(
            self.fault_type_wds + self.phenomenon_wds + self.component_wds +
            self.solution_wds + self.detection_wds + self.cause_wds +
            self.prevent_wds + self.parameter_wds + self.cure_way_wds +
            self.category_wds + self.material_wds
        )

        # 构建AC自动机
        self.region_tree = self.build_actree(list(self.region_words))
        # 构建词典
        self.wdtype_dict = self.build_wdtype_dict()

        # ===== 问句疑问词（保持不变）=====
        self.cause_qwds = ['原因', '成因', '为什么', '怎么会', '怎样才', '咋样才', '怎样会', '如何会',
                           '为啥', '为何', '如何才会', '怎么才会', '会导致', '会造成', '引起', '导致',
                           '由什么引起', '是什么造成的', '怎么造成的', '什么问题']

        self.fault_type_qwds = ['什么故障', '什么毛病', '什么问题', '什么故障类型', '属于什么', '是什么病',
                                '怎么判断', '怎么诊断', '是什么问题', '什么毛病', '咋回事', '怎么回事',
                                '是什么故障', '判定为', '诊断为']

        self.solution_qwds = ['怎么处理', '怎么解决', '怎么办', '咋办', '咋处理', '如何处理',
                              '如何解决', '怎么修', '怎么排除', '排除方法', '修复方法', '应对措施',
                              '怎么处理', '怎么弄', '怎么搞', '怎么办好', '该咋整', '维修方法',
                              '解决办法', '处理方案', '修复方案', '怎么修复', '怎么维修']

        self.prevent_qwds = ['预防', '防范', '抵制', '抵御', '防止', '躲避', '逃避', '避开', '免得',
                             '逃开', '避掉', '躲开', '躲掉', '绕开',
                             '怎样才能不', '怎么才能不', '咋样才能不', '咋才能不', '如何才能不',
                             '怎样才不', '怎么才不', '咋样才不', '咋才不', '如何才不',
                             '怎样才可以不', '怎么才可以不', '咋样才可以不', '咋才可以不', '如何可以不',
                             '怎样才可不', '怎么才可不', '咋样才可不', '咋才可不', '如何可不',
                             '如何避免', '怎么避免', '如何防止', '怎么防止', '注意什么', '要注意',
                             '需要预防', '怎么杜绝', '如何杜绝']

        self.parameter_qwds = ['参数', '调参数', '调整参数', '优化参数', '怎么调', '怎么设置',
                               '该怎么配', '如何设置', '如何调整', '参数设置', '参数调整',
                             '切削参数', '加工参数', '转速多少', '进给多少', '切深多少',
                             '该怎么调', '怎样调整', '怎么配置', '参数配置', '参数推荐',
                             '该怎么设', '怎么优化', '如何优化', '最佳参数', '合理参数']

        self.detection_qwds = ['检测', '怎么检测', '如何检测', '怎么查', '如何查', '检查',
                               '诊断', '监测', '测量', '测试', '怎么看', '怎么判断',
                               '检测方法', '诊断方法', '监测手段', '检查项目', '用什么检测',
                               '用什么仪器', '怎么发现', '如何发现', '判定方法', '识别方法']

        self.material_qwds = ['材料', '加工材料', '材质', '什么材料', '哪种材料']
        self.desc_qwds = ['介绍', '简介', '是什么', '什么样', '了解一下', '详细说明', '说明']

        # 新增：类别查询
        self.category_qwds = ['属于什么', '什么类别', '什么类型', '归类', '分类']

        # 新增：修复方式
        self.cure_way_qwds = ['怎么修复', '怎么维修', '如何修复', '如何维修', '修复方式', '维修方式']

        # 新增：易发情况
        self.easy_get_qwds = ['什么情况', '什么时候', '何种情况', '容易', '易发生', '情况下发生']

        # 新增：修复时间
        self.cure_lasttime_qwds = ['需要多长时间', '多久', '修复时间', '多长时间', '需要多久']

        # 新增：修复概率
        self.cured_prob_qwds = ['能修好', '修复概率', '能修好吗', '成功率', '可能性']

        # 新增：综合查询
        self.full_info_qwds = ['详细介绍', '所有信息', '全部信息', '详细信息', '完整信息']

        print('[成功] 机床故障诊断意图识别模型初始化完成 (中文标签版) ......')

    def build_wdtype_dict(self):
        """构造词对应的类型 - 中文版"""
        wd_dict = dict()
        for wd in self.region_words:
            wd_dict[wd] = []
            if wd in self.fault_type_wds:
                wd_dict[wd].append('故障类型')
            if wd in self.phenomenon_wds:
                wd_dict[wd].append('症状')
            if wd in self.component_wds:
                wd_dict[wd].append('部件')
            if wd in self.solution_wds:
                wd_dict[wd].append('解决方案')
            if wd in self.detection_wds:
                wd_dict[wd].append('检查')
            if wd in self.material_wds:
                wd_dict[wd].append('材料')
            if wd in self.category_wds:
                wd_dict[wd].append('类别')
            if wd in self.cure_way_wds:
                wd_dict[wd].append('修复方式')
            if wd in self.cause_wds:
                wd_dict[wd].append('原因')
            if wd in self.prevent_wds:
                wd_dict[wd].append('预防')
            if wd in self.parameter_wds:
                wd_dict[wd].append('参数')
        return wd_dict

    def build_actree(self, wordlist):
        """构造AC自动机，加速过滤"""
        actree = ahocorasick.Automaton()
        for index, word in enumerate(wordlist):
            actree.add_word(word, (index, word))
        actree.make_automaton()
        return actree

    def check_medical(self, question):
        """问句过滤，提取领域实体"""
        region_wds = []
        for i in self.region_tree.iter(question):
            wd = i[1][1]
            region_wds.append(wd)
        # 去子串（保留最长的）
        stop_wds = []
        for wd1 in region_wds:
            for wd2 in region_wds:
                if wd1 in wd2 and wd1 != wd2:
                    stop_wds.append(wd1)
        final_wds = [i for i in region_wds if i not in stop_wds]
        final_dict = {i: self.wdtype_dict.get(i) for i in final_wds}
        return final_dict

    def check_words(self, wds, sent):
        """基于特征词进行分类"""
        for wd in wds:
            if wd in sent:
                return True
        return False

    def classify(self, question):
        """分类主函数 - 中文标签版"""
        data = {}
        medical_dict = self.check_medical(question)
        if not medical_dict:
            return {}
        data['args'] = medical_dict

        # 收集问句中涉及的实体类型
        types = []
        for type_ in medical_dict.values():
            types += type_

        question_types = []

        # ===== 1. 故障原因查询 =====
        if self.check_words(self.cause_qwds, question) and ('故障类型' in types):
            question_types.append('fault_cause')

        # ===== 2. 现象诊断（已知症状查故障）=====
        if self.check_words(self.fault_type_qwds, question) and ('症状' in types):
            question_types.append('phenomenon_diagnosis')

        # ===== 3. 解决方法推荐 =====
        if self.check_words(self.solution_qwds, question) and ('故障类型' in types):
            question_types.append('fault_solution')

        # ===== 4. 部件关联故障 =====
        if self.check_words(self.fault_type_qwds, question) and ('部件' in types):
            question_types.append('component_fault')

        # ===== 5. 预防措施 =====
        if self.check_words(self.prevent_qwds, question) and ('故障类型' in types):
            question_types.append('fault_prevent')

        # ===== 6. 参数优化（基于症状）=====
        if self.check_words(self.parameter_qwds, question) and ('症状' in types):
            question_types.append('parameter_optimize')
        if self.check_words(self.parameter_qwds, question) and ('故障类型' in types):
            question_types.append('fault_parameter')

        # ===== 7. 检测手段 =====
        if self.check_words(self.detection_qwds, question) and ('故障类型' in types):
            question_types.append('fault_detection')
        if self.check_words(self.detection_qwds, question) and ('症状' in types):
            question_types.append('phenomenon_detection')

        # ===== 8. 材料相关故障 =====
        if self.check_words(self.material_qwds + self.fault_type_qwds, question) and ('材料' in types):
            question_types.append('material_fault')

        # ===== 9. 故障描述 =====
        if question_types == [] and '故障类型' in types:
            question_types = ['fault_desc']
        if question_types == [] and '症状' in types:
            question_types = ['phenomenon_fault']
        if question_types == [] and '部件' in types:
            question_types = ['component_fault']

        # ===== 10. 故障类别查询 =====
        if self.check_words(self.category_qwds, question) and ('故障类型' in types):
            question_types.append('fault_category')

        # ===== 11. 修复方式查询 =====
        if self.check_words(self.cure_way_qwds, question) and ('故障类型' in types):
            question_types.append('fault_cure_way')

        # ===== 12. 易发情况查询 =====
        if self.check_words(self.easy_get_qwds, question) and ('故障类型' in types):
            question_types.append('fault_easy_get')

        # ===== 13. 修复时间查询 =====
        if self.check_words(self.cure_lasttime_qwds, question) and ('故障类型' in types):
            question_types.append('fault_cure_lasttime')

        # ===== 14. 修复概率查询 =====
        if self.check_words(self.cured_prob_qwds, question) and ('故障类型' in types):
            question_types.append('fault_cured_prob')

        # ===== 15. 综合查询（故障全信息）=====
        if self.check_words(self.full_info_qwds, question) and ('故障类型' in types):
            question_types.append('fault_full_info')

        # 去重
        question_types = list(set(question_types))
        data['question_types'] = question_types
        return data


if __name__ == '__main__':
    handler = QuestionClassifier()

    # 测试所有20种意图类型
    test_questions = [
        "主轴过热是什么原因？",  # fault_cause
        "铣削时出现振纹是什么故障？",  # phenomen_diagnosis
        "刀具崩刃怎么处理？",  # fault_solution
        "导轨一般会出现什么故障？",  # component_fault
        "如何避免刀具磨损？",  # fault_prevent
        "表面粗糙度差该怎么调参数？",  # parameter_optimize
        "怎么检测主轴是否过热？",  # fault_detection
        "加工时出现异响是怎么回事？",  # phenomen_fault
        "轴承损坏如何预防？",  # fault_prevent
        "钛合金加工容易出现什么问题？",  # material_fault
        "刀具磨损属于什么类别？",  # fault_category
        "主轴过热该怎么修复？",  # fault_cure_way
        "刀具磨损在什么情况下容易发生？",  # fault_easy_get
        "主轴过热修复需要多长时间？",  # fault_cure_lasttime
        "刀具磨损能修好吗？",  # fault_cured_prob
        "详细介绍刀具磨损的所有信息",  # fault_full_info
    ]

    print("=" * 80)
    print("[测试] Testing all intent types...")
    print("=" * 80)
    for q in test_questions:
        data = handler.classify(q)
        print(f"Q: {q}")
        print(f"A: {data}")
        print()
