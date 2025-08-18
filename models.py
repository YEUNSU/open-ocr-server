from pydantic import BaseModel, Field
from typing import List, Optional

class TargetItem(BaseModel):
    target_relation: str
    target_name: str
    target_birth: str = Field(..., pattern=r"^\d{6}$")


class DetectItem(BaseModel):
    detect_relation: str
    detect_name: str
    detect_birth: str = Field(..., pattern=r"^\d{6}$")

class ChildCreateRequest(BaseModel):
    tran_id: str
    saved_file: str
    host: str
    type: str
    channel: Optional[str] = "nais"
    detect: Optional[List[DetectItem]] = None
    target: List[TargetItem]

class JsonResponseModel(BaseModel):
    tran_id: str
    ocr_id: Optional[str] = None
    statusCd: str
    statusMessage: str

class StringINDTO(BaseModel):
    stringValue: Optional[str] = None