from fastapi import APIRouter

router = APIRouter()


@router.post("/completions")
async def completions():
    return {"message": "TODO: chat"}