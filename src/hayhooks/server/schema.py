from pydantic import BaseModel
from typing import List, Literal


class ModelObject(BaseModel):
    id: str
    name: str
    object: Literal["model"]
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    data: List[ModelObject]
    object: Literal["list"]
