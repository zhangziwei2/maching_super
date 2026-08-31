from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base

# 机器级公共记忆的去重键（user_id 为 NULL 时的替代值）
MACHINE_OWNER_KEY = "__machine__"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), default="user", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    __table_args__ = (UniqueConstraint("user_id", "session_id", name="uq_user_session"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_ref_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    message_type: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    rag_trace: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    session = relationship("ChatSession", back_populates="messages")


class ChatTranscript(Base):
    """被上下文压缩掉的原始对话归档。

    压缩只应把内容"移出上下文"，而不是销毁：归档后历史可追溯，
    也避免 Redis 缓存回退时出现"只剩一句摘要"的不可逆丢失。
    """

    __tablename__ = "chat_transcripts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    # 存 username（与调用方 user_id 语义一致），避免每次归档都查一次 users 表
    user_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)


class ContextBlob(Base):
    """被卸载(out-of-context)的大块内容：超长检索片段、超长历史消息。

    对应 harness 的 .task_outputs/tool-results/ 落盘机制，但保留 blob_id
    以便按需回源，形成"预览层 + 全文层"两级上下文。
    """

    __tablename__ = "context_blobs"

    blob_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(32), default="chunk", nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    preview: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)


class Memory(Base):
    """跨会话记忆（对应 harness s09 memory，适配多用户机床诊断场景）。

    与 harness（单用户 .memory/ 目录）的关键差异是**多用户隔离**，因此：
    - user_id 为 NULL 表示机器级公共记忆（全用户可见、仅 admin 可写），
      存 username 表示个人记忆
    - owner_key 是去重的落库键：机器级固定为 MACHINE_OWNER_KEY。
      之所以不直接对 user_id 建唯一索引，是因为 PG 与 SQLite 都规定
      NULL 不参与 UNIQUE 约束，机器级记忆将无法去重

    mem_type 语义（适配机床诊断）：
        user     → 学员画像，如"新手，需解释专业术语"
        feedback → 纠正反馈，如"主轴答案有误，实为皮带打滑"
        project  → 设备事实，如"XK7132 数控铣床，主轴最高 8000rpm"
        reference→ 外部引用，如"见操作手册 P47"
    """

    __tablename__ = "memories"
    __table_args__ = (
        UniqueConstraint("owner_key", "name", name="uq_memories_owner_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    # 存 username（与 ChatTranscript.user_id 语义一致）；NULL = 机器级公共记忆
    user_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    # 去重键：个人记忆为 username，机器级为 MACHINE_OWNER_KEY
    owner_key: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    created_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    scope: Mapped[str] = mapped_column(String(20), default="personal", nullable=False, index=True)
    mem_type: Mapped[str] = mapped_column(String(20), default="project", nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    body: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)


class ParentChunk(Base):
    __tablename__ = "parent_chunks"

    chunk_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    file_type: Mapped[str] = mapped_column(String(50), default="", nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    parent_chunk_id: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    root_chunk_id: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    chunk_level: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_idx: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class UploadJob(Base):
    """上传/删除后台任务状态（持久化到 SQLite，服务重启不丢失）。"""

    __tablename__ = "upload_jobs"

    job_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False, index=True)
    current_step: Mapped[str] = mapped_column(String(50), default="", nullable=False)
    message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    completion_step: Mapped[str] = mapped_column(String(50), default="", nullable=False)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    processed_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    steps_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
