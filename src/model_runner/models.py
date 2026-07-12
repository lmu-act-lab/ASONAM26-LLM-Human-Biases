from typing import Annotated
from typing_extensions import Literal
from pydantic import BaseModel, Field


class PoliticalBiasAssessmentWithCoT(BaseModel):
    # This field MUST come first to force the model to think before deciding
    thought_process: str = Field(description="Step-by-step internal reasoning")
    assessment: Literal["is-biased", "is-not-biased"]
    confidence_score: Annotated[int, Field(ge=1, le=100)]
    explanation: str = Field(description="Final public-facing explanation for the reader.")
class PoliticalBiasAssessment(BaseModel):
    assessment: Literal["is-biased", "is-not-biased"]
    confidence_score: Annotated[int, Field(ge=1, le=100)]
    explanation: str

class PoliticalBiasAssessmentNoExplanation(BaseModel):
  assessment: Literal["is-biased", "is-not-biased"]
  confidence_score: Annotated[int, Field(ge=1, le=100)]
