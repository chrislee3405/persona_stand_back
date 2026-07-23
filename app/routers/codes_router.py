from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel

from app.services.code_service import CodeService

router = APIRouter()

# Define the data structure you expect from the frontend
class CodeSubmission(BaseModel):
    input_code: str

@router.post("/api/code")
async def verify_code(payload: CodeSubmission, service: CodeService = Depends()):
    # Call service file 
    processed_result = await service.process_invite_code(payload.input_code)
    
    # Check result from service file
    if processed_result is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid invitation code."
        )
        
    return {
        "status": "success",
        "received": payload.input_code,
        "returned_result": processed_result
    }