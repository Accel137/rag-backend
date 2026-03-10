from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.router import api_router
from app.core.config import settings
from app.llm.factory import build_llm_service
from app.services.chat_service import ChatService



@asynccontextmanager
async def lifespan(app: FastAPI):

    # embedder = build_embedder(settings.rag)
    # chunker = build_chunker(settings.rag)

    app.state.llm_service = build_llm_service(settings.llm)
    # app.state.rag_service = RAGService(chunker, embedder)
    app.state.chat_service = ChatService(
        llm_service=app.state.llm_service,
        # rag_service=app.state.rag_service,
    )

    yield

app = FastAPI(
    title=settings.app.app_name,
    debug=settings.app.debug,
    lifespan=lifespan
)

app.include_router(api_router)
