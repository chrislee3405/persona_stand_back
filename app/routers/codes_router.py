from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel

from app.services.code_service import CodeService, InvalidCodeError


router = APIRouter()

class CodeSubmission(BaseModel):
    input_code: str
    conversation_id: str | None = None

@router.post("/api/code")
async def verify_code(payload: CodeSubmission, service: CodeService = Depends()):
    try:
        processed_result = await service.verify_and_link(
            input_code=payload.input_code,
            conversation_id=payload.conversation_id
        )
    except InvalidCodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Process result not found"
        )

    return {
        "status": "success",
        "received": payload.input_code,
        "returned_result": processed_result
    }