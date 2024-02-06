from pydantic import BaseModel, Field
from typing import List, Union


class SummaryOutputClass(BaseModel):
    summary: str = Field(description="Summary of a product description")


class KeywordsOutputClass(BaseModel):
    keywords: List[str] = Field(description="List of keywords")


class AggregateOutputClass(BaseModel):
    summaries: List[Union[SummaryOutputClass, KeywordsOutputClass]] = Field(description="List of aggregated outputs")
