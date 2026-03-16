from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.router import api_router
from app.core.config import settings
from app.llm.factory import build_llm_service
from app.services.chat_service import ChatService
from app.rag.factory import build_rag_module



@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm_service = build_llm_service(settings.llm)

    rag_module = build_rag_module(settings.rag)

    app.state.rag_pipeline = rag_module.rag_pipeline
    app.state.ingestor = rag_module.ingestor
    app.state.strategy_pipeline = rag_module.strategy_pipeline
    app.state.chat_service = ChatService(
        llm_service=app.state.llm_service,
        rag_pipeline=app.state.rag_pipeline,
        strategy_pipeline=app.state.strategy_pipeline,
    )

    yield

app = FastAPI(
    title=settings.app.app_name,
    debug=settings.app.debug,
    lifespan=lifespan
)

app.include_router(api_router)
