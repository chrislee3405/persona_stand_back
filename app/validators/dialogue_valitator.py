from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class DialogueLine(BaseModel):
    speaker: str
    dialogue: str
    timestamp: datetime

class UserInputCreate(BaseModel):
    content: List[DialogueLine]
    code: Optional[str] = "GUEST"