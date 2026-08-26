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

# SQLite 在多线程（ThreadPoolExecutor 后台上传任务）下默认禁止跨线程复用连接，
# 会导致 "SQLite objects created in a thread..." 错误，甚至进度回调死锁卡在 89%。
# 通过 check_same_thread=False + StaticPool 让单一连接在线程间安全共享（已有 manager 锁保护并发）。
_connect_args = {}
_engine_kwargs = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    _connect_args["check_same_thread"] = False
    _engine_kwargs["poolclass"] = __import__("sqlalchemy.pool", fromlist=["StaticPool"]).StaticPool

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
