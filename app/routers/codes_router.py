from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel

from app.services.code_service import CodeService
from app.services.conversation_service import ConversationService


router = APIRouter()

# Define the data structure you expect from the frontend
class CodeSubmission(BaseModel):
    input_code: str
    conversation_id: str | None = None

@router.post("/api/code")
async def verify_code(payload: CodeSubmission, service: CodeService = Depends(), conversation_service: ConversationService = Depends()):
    # Call service file 
    processed_result = await service.process_invite_code(payload.input_code)
    
    # Check result from service file
    if processed_result is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Process result not found"
        )


    # If the user already has a conversation going (sent messages as guest
    # before verifying), update that row's code column in place instead of
    # leaving it stuck on "GUEST".
    if payload.conversation_id:
        await conversation_service.update_conversation_code(
            conversation_id=payload.conversation_id,
            code=processed_result
        )
        
    return {
        "status": "success",
        "received": payload.input_code,
        "returned_result": processed_result
    }