from __future__ import annotations
from typing import Generic, TypeVar, Any
from pydantic import BaseModel, Field

# Define a TypeVar for your Pydantic Output Model
T = TypeVar("T", bound=BaseModel | str)

class PromptTurn(BaseModel, Generic[T]):
    """A single turn: Input is always a dict, Answer is either str or Pydantic"""
    user_input: dict[str, str]
    assistant_output: T | None = None

class FullPrompt(BaseModel, Generic[T]):
    system_prompt: str
    main_header: str = ""
    examples: list[PromptTurn[T]] = Field(default_factory=list)