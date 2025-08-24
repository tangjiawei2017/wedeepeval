from typing import List, Optional
from pydantic import BaseModel, Field


class QAItem(BaseModel):
    question: str
    expected_output: str
    context: List[str] = Field(default_factory=list)
    context_length: int = 0


class FromContextRequest(BaseModel):
    contexts: List[str] = Field(..., description="每条上下文信息一行，建议每组2-5条相关信息")
    num_questions: int = Field(10, ge=1, le=100, description="期望生成的问答对数量")


class FromTopicRequest(BaseModel):
    topic: str = Field(..., description="主题或场景，例如：技术运维/银行业务/客服服务等")
    num_questions: int = Field(20, ge=1, le=200, description="期望生成的问题数量")


class AugmentDatasetResponseItem(BaseModel):
    input: str
    expected_output: Optional[str] = None


class TopicDatasetResponseItem(BaseModel):
    input: str


class QAResponse(BaseModel):
    items: List[QAItem]


class TopicResponse(BaseModel):
    items: List[TopicDatasetResponseItem]


class AugmentResponse(BaseModel):
    items: List[AugmentDatasetResponseItem]

