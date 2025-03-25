from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Document:
    id: str 
    vector: list 
    metadata: Dict[str, Any]