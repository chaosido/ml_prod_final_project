from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Response model for ASR Quality Estimation."""

    predicted_wer: float = Field(
        ..., description="Predicted Word Error Rate (0.0 to 1.0)"
    )
    review_recommended: bool = Field(
        ..., description="Flag indicating if manual review is needed"
    )
    transcript: str = Field(..., description="Transcript of the audio file")
