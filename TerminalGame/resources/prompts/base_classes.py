from typing import List
from pydantic import BaseModel

class SimpleOutput(BaseModel):
    answer: str

# class TkgWithAnswer(BaseModel):
#     TKG: List[dict]
#     answer: str 

# class Reasoning(BaseModel):
#     reasoning: str
#     answer: str 

# class Combo(BaseModel):
#     TKG: List[dict]
#     reasoning: str
#     answer: str