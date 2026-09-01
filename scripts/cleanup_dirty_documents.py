"""清理 data/documents/ 下的"双 uuid 前缀"脏文件。

背景（踩坑记录，详见 docs/TECH_GUIDE.md 第 9 节）：
  早期 `_save_upload_stream()` 内部又叠加了一次 uuid 前缀，导致磁盘落盘名为
  `<uuid2>_<uuid1>_原名`（双前缀），而提交给任务队列的 file_path 只有单前缀
  `<uuid1>_原名`。后台解析时两者对不上，报：
      [Errno 2] No such file or directory: .../<uuid1>_原名
  该 bug 已修复（api.py 中不再叠加第二个 uuid），但历史遗留的双前缀脏文件
  仍堆积在 data/documents/ 下，未被任何任务引用，可安全清理。

判定规则（双前缀脏文件）：
  文件名形如 `8位hex_8位hex_原始文件名`，例如：
      df6ea2a6_58b09bf2_故障文档.docx
      b6ef300c_82a3e123_故障文档.docx
  正常（单前缀）文件如 `cfb85856_故障排除.txt` 不受影响，绝不误删。

安全策略：
  - 默认 **dry-run**：只列出将被删除的文件，不做任何实际操作。
  - 必须显式传 `--apply` 才会真正删除（删除不可逆，请谨慎）。

用法：
  python scripts/cleanup_dirty_documents.py            # 预演，仅列出
  python scripts/cleanup_dirty_documents.py --apply     # 确认无误后真正删除
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# 双 uuid 前缀：两段 8 位十六进制，以下划线分隔，紧跟原始文件名
DIRTY_PATTERN = re.compile(r"^[0-9a-fA-F]{8}_[0-9a-fA-F]{8}_.+")

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = ROOT / "data" / "documents"


def find_dirty_files(directory: Path) -> list[Path]:
    """找出符合"双 uuid 前缀"特征的脏文件（只看文件，不递归目录）。"""
    if not directory.is_dir():
        return []
    return sorted(
        (p for p in directory.iterdir() if p.is_file() and DIRTY_PATTERN.match(p.name)),
        key=lambda p: p.name,
    )


def _human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}GB"


def main() -> int:
    apply_delete = "--apply" in sys.argv

    if not DOCUMENTS_DIR.is_dir():
        print(f"目录不存在，无需清理: {DOCUMENTS_DIR}")
        return 0

    dirty = find_dirty_files(DOCUMENTS_DIR)

    print(f"扫描目录: {DOCUMENTS_DIR}")
    print(f"模式: {'实际删除 (--apply)' if apply_delete else '预演 dry-run（不删除，加 --apply 才会真删）'}")
    print("-" * 70)

    if not dirty:
        print("未发现双 uuid 前缀的脏文件，无需清理。")
        return 0

    total_bytes = 0
    for p in dirty:
        size = p.stat().st_size
        total_bytes += size
        print(f"  脏文件: {p.name}  ({_human_size(size)})")

    print("-" * 70)
    print(f"共 {len(dirty)} 个脏文件，合计 {_human_size(total_bytes)}")

    if not apply_delete:
        print("\n这是预演。确认列表无误后，重新运行并加 --apply 执行删除：")
        print("  python scripts/cleanup_dirty_documents.py --apply")
        return 0

    deleted = 0
    failed = 0
    for p in dirty:
        try:
            p.unlink()
            print(f"  已删除: {p.name}")
            deleted += 1
        except OSError as e:
            print(f"  删除失败: {p.name} -> {e}")
            failed += 1

    print("-" * 70)
    print(f"清理完成：删除 {deleted} 个，失败 {failed} 个。")

    remaining = [p.name for p in DOCUMENTS_DIR.iterdir() if p.is_file()]
    print(f"\n当前 {DOCUMENTS_DIR} 剩余文件：")
    for name in sorted(remaining):
        print(f"  - {name}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
