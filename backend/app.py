from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import logging
import os

from . import api as api_module
from .database import init_db

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

logger = logging.getLogger("startup")


def create_app() -> FastAPI:
    app = FastAPI(title="Cute Cat Bot API")

    @app.on_event("startup")
    async def _startup_init_db():
        import asyncio
        init_db()
        # 预热 embedding 模型，避免第一次上传时卡死
        async def _warmup():
            try:
                from .embedding import embedding_service

                logger.info("[startup] 正在预热 embedding 模型...")
                # 在后台线程执行同步的模型加载，避免阻塞事件循环
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, lambda: embedding_service.get_embeddings(["预热"]))
                logger.info("[startup] embedding 模型预热完成")
            except Exception as e:
                # 预热失败不影响服务启动，但会记录警告
                logger.warning(f"[startup] 预热失败（首次上传可能较慢）: {e}")

        async def _warmup_chatter():
            """预热颤振诊断模型（约 100MB，耗时较长）"""
            try:
                logger.info("[startup] 正在预热 chatter 诊断模型...")
                from .api import _warmup_chatter_models
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, _warmup_chatter_models)
                logger.info("[startup] chatter 诊断模型预热完成")
            except Exception as e:
                logger.warning(f"[startup] chatter 模型预热失败（不影响服务）: {e}")

        # 在后台执行预热，不阻塞 startup 完成
        asyncio.create_task(_warmup())
        asyncio.create_task(_warmup_chatter())

    app.add_middleware(
        CORSMiddleware,
        # 仅允许本机来源（localhost/127.0.0.1 任意端口），避免通配符 + credentials 的非法组合
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 请求日志中间件 - 放在 CORS 之后，确保能捕获所有请求
    @app.middleware("http")
    async def _log_requests(request, call_next):
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[REQUEST] {ts} {request.method} {request.url.path}", flush=True)
        # 打印所有 headers（用于调试认证问题）
        auth = request.headers.get("authorization", "")
        print(f"[REQUEST]   Authorization: {'Bearer ***' if auth else '(none)'}", flush=True)
        response = await call_next(request)
        print(f"[REQUEST]   -> {response.status_code}", flush=True)
        return response

    # No-cache middleware for development
    @app.middleware("http")
    async def _no_cache(request, call_next):
        response = await call_next(request)
        path = request.url.path or ""
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    app.include_router(api_module.router)

    # serve frontend static files at root
    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)))
