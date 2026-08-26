# -*- coding: utf-8 -*-
"""
机床故障诊断知识图谱 - 人工/专家三元组导入器

职责：
1. 解析三元组文件（兼容 JSON 数组 / {"triples": [...]} 包装 / JSONL 三种格式，
   格式 {"h": {"name","type"}, "r": {"type"}, "t": {"name","type"}}）
2. 可选 LLM 一致性校验：关系类型合理性、实体命名冲突提示（离线辅助，不影响导入结果）
3. 入库到 Upload 命名空间 + 增量持久化

设计要点：
- LLM 校验为"软校验"：只产出建议报告，不阻塞导入（工业场景下人工审核为最终裁决）；
- 未配置 LLM API Key 时自动跳过 LLM 校验，仅做格式与必填校验；
- 导入后触发图谱缓存保存；删除对应 JSONL文件后执行 rebuild 即可整体回滚。
"""

import json
import os
from typing import Optional

from graph_schema import NAMESPACE_UPLOAD


class TripleImporter:
    def __init__(self, store):
        self.store = store

    # ---------- 解析与格式校验 ----------

    @staticmethod
    def _validate_triple(triple: dict) -> Optional[str]:
        """返回错误信息；合法返回 None"""
        if not isinstance(triple, dict):
            return "三元组必须是 JSON 对象"
        h = triple.get("h") or {}
        t = triple.get("t") or {}
        r = triple.get("r") or {}
        if not h.get("name") or not str(h.get("name", "")).strip():
            return "头实体 h.name 缺失"
        if not t.get("name") or not str(t.get("name", "")).strip():
            return "尾实体 t.name 缺失"
        if not r.get("type") or not str(r.get("type", "")).strip():
            return "关系 r.type 缺失"
        return None

    def parse_file(self, filepath: str) -> tuple:
        """
        解析三元组文件。返回 (valid_triples, errors)。
        兼容三种格式：
        1. JSON 数组：[{...}, {...}]（单行或多行书写）
        2. JSON 对象包装：{"triples": [{...}, ...]}（可选 extra 字段）
        3. JSONL：每行一个三元组对象
        """
        valid, errors = [], []
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            return valid, errors

        items = []  # [(line_no, item)]
        try:
            root = json.loads(text)
        except json.JSONDecodeError:
            root = None

        if root is not None:
            if isinstance(root, list):
                items = [(i, it) for i, it in enumerate(root, 1)]
            elif isinstance(root, dict):
                # 兼容 {"triples": [...]} 包装
                inner = root.get("triples") or root.get("data")
                if isinstance(inner, list):
                    items = [(i, it) for i, it in enumerate(inner, 1)]
                else:
                    items = [(1, root)]
        else:
            # JSONL 兜底（兼容行尾逗号 / 数组分行写）
            for line_no, line in enumerate(text.splitlines(), 1):
                line = line.strip()
                if not line or line in ("[", "]", "{", "}"):
                    continue
                if line.endswith(","):
                    line = line[:-1].strip()
                try:
                    items.append((line_no, json.loads(line)))
                except json.JSONDecodeError as e:
                    errors.append({"line": line_no, "error": f"JSON 解析失败: {e}"})

        for line_no, triple in items:
            err = self._validate_triple(triple)
            if err:
                errors.append({"line": line_no, "error": err})
                continue
            valid.append(triple)
        return valid, errors

    # ---------- LLM 软校验 ----------

    def llm_validate(self, triples: list) -> list:
        """
        用 LLM 对三元组做一致性软校验，返回建议报告列表。
        未配置 LLM 时返回 []（跳过校验）。
        """
        if not triples:
            return []
        api_key = (os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "").strip()
        if not api_key:
            return []
        base_url = (os.getenv("LLM_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL") or "https://api.deepseek.com/v1").strip()
        model = os.getenv("LLM_MODEL", "deepseek-chat").strip()
        try:
            import requests
            sample = json.dumps(triples[:20], ensure_ascii=False)
            prompt = (
                "你是机床故障诊断领域的知识图谱质检员。检查以下三元组列表，"
                "对每条给出判定：OK（合理）或 FLAG（关系类型与实体语义不匹配/实体命名冲突等），"
                "并给出简短中文说明。按行输出：序号|判定|说明\n\n"
                f"三元组：\n{sample}"
            )
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 800},
                timeout=30,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            report = []
            for line in content.splitlines():
                line = line.strip()
                if not line or "|" not in line:
                    continue
                parts = line.split("|")
                report.append({
                    "index": parts[0].strip(),
                    "verdict": parts[1].strip() if len(parts) > 1 else "",
                    "note": parts[2].strip() if len(parts) > 2 else "",
                })
            return report
        except Exception as e:
            return [{"index": "-", "verdict": "LLM_UNAVAILABLE", "note": f"LLM 校验不可用: {e}"}]

    # ---------- 入库 ----------

    def import_file(self, filepath: str, llm_validate: bool = True) -> dict:
        """导入一个 JSONL 文件到 Upload 命名空间"""
        valid, errors = self.parse_file(filepath)
        imported = 0
        for triple in valid:
            try:
                self.store.add_triple(
                    triple.get("h", {}),
                    triple.get("r", {}),
                    triple.get("t", {}),
                    namespace=NAMESPACE_UPLOAD,
                )
                imported += 1
            except (ValueError, KeyError) as e:
                errors.append({"line": "-", "error": f"入库失败: {e}"})
        report = []
        if llm_validate and valid:
            report = self.llm_validate(valid)
        self.store.save()
        return {
            "file": os.path.basename(filepath),
            "total": len(valid) + len(errors),
            "imported": imported,
            "rejected": len(errors),
            "errors": errors[:20],
            "llm_report": report,
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python triple_importer.py <triples.jsonl>")
        sys.exit(1)
    from graph_store import GraphStore
    store = GraphStore()
    store.ensure_built()
    importer = TripleImporter(store)
    result = importer.import_file(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False, indent=2))
