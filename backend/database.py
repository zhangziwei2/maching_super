import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# 显式加载项目根目录 .env，确保 DATABASE_URL 环境变量生效（不依赖启动 CWD）
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# 默认 PostgreSQL（SuperMew 架构），可通过环境变量 DATABASE_URL 覆盖为 SQLite 等
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/langchain_app",
)

# SQLite 在多线程（后台上传任务、进度回调等）下有两个坑：
# 1) 默认 check_same_thread=True 禁止跨线程用连接 → 需设 False；
# 2) 若用 StaticPool 强制"单连接在线程间共享"，并发 commit 会让 SQLite C 层
#    崩溃并返回 "commit() returned NULL without setting an exception"（Python 层无异常信息）。
# 因此改用默认连接池（每线程/每请求独立连接）+ check_same_thread=False + timeout，
# 既允许多线程各自拿连接，又避免单连接被并发复用导致的 NULL commit 崩溃。
_connect_args = {}
_engine_kwargs = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    _connect_args["check_same_thread"] = False
    _connect_args["timeout"] = 30

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    **_engine_kwargs,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()


def init_db() -> None:
    # Delayed import to avoid circular dependency.
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
