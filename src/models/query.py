from dataclasses import dataclass
from typing import Any

@dataclass
class Query:
    input_data: Any 
    input_type: str 
    top_k: int = 3


