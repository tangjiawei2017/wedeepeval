from typing import Dict, Optional, Any
from pydantic import BaseModel


class Knowledge(BaseModel):
    data: Dict[str, Any]


class KnowledgeRetentionVerdict(BaseModel):
    verdict: str
    reason: Optional[str] = None


class KnowledgeRetentionScoreReason(BaseModel):
    reason: str
