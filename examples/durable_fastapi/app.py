"""Embed the Hayhooks durable engine in a standalone FastAPI application."""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from haystack import Pipeline, component
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from hayhooks.durable import (
    DurableContext,
    DurableDeployment,
    DurableRuntime,
    create_durable_router,
    current_durable_context,
)
from hayhooks.durable.haystack import HaystackDurableAdapter
from hayhooks.durable.redis import RedisExecutionStore

DEPLOYMENT_NAME = "document-analysis"
DEPLOYMENT_REVISION = "document-analysis-v1"


class DocumentRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=100_000)
    require_approval: bool = True
    processing_delay_seconds: float = Field(default=0, ge=0, le=30)


class Approval(BaseModel):
    approved: bool


class AnalysisResult(BaseModel):
    document_id: str
    word_count: int
    unique_terms: int


@component
class ExtractTerms:
    @component.output_types(terms=list[str])
    def run(self, text: str) -> dict[str, list[str]]:
        return {"terms": re.findall(r"[\w'-]+", text.lower())}


@component
class AnalyzeDocument:
    @component.output_types(document_id=str, word_count=int, unique_terms=int)
    def run(
        self,
        document_id: str,
        terms: list[str],
        processing_delay_seconds: float,
    ) -> dict[str, str | int]:
        context = current_durable_context()
        if context is not None:
            context.check_cancelled_sync()
        time.sleep(processing_delay_seconds)
        if context is not None:
            context.check_cancelled_sync()
        return {
            "document_id": document_id,
            "word_count": len(terms),
            "unique_terms": len(set(terms)),
        }


def build_pipeline() -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("extract", ExtractTerms())
    pipeline.add_component("analyze", AnalyzeDocument())
    pipeline.connect("extract.terms", "analyze.terms")
    return pipeline


async def run_document_analysis(
    context: DurableContext,
    request: DocumentRequest,
) -> AnalysisResult:
    if request.require_approval and not context.state.get("approved"):
        resume_input = context.resume_input
        if resume_input is None:
            await context.suspend(
                {
                    "kind": "approval",
                    "message": f"Approve analysis of '{request.document_id}'?",
                }
            )
        if not Approval.model_validate(resume_input).approved:
            message = "document analysis was rejected"
            raise ValueError(message)
        context.state["approved"] = True
        await context.report_progress("Document analysis approved", kind="approval")
        await context.checkpoint()

    result = await context.run_pipeline_async(
        {
            "extract": {"text": request.text},
            "analyze": {
                "document_id": request.document_id,
                "processing_delay_seconds": request.processing_delay_seconds,
            },
        },
        checkpoint_at="analyze",
    )
    return AnalysisResult.model_validate(result["analyze"])


bearer = HTTPBearer(auto_error=False)


def owner_id(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> str:
    """Replace this API-key check with the authenticated subject from your app."""
    configured_key = os.getenv("APP_API_KEY")
    if configured_key is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "APP_API_KEY is not configured")
    if credentials is None or not secrets.compare_digest(credentials.credentials.encode(), configured_key.encode()):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return hashlib.sha256(credentials.credentials.encode()).hexdigest()


redis = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=False)
store = RedisExecutionStore(
    redis,
    DEPLOYMENT_NAME,
    key_prefix=os.getenv("DURABLE_REDIS_KEY_PREFIX", "hayhooks:durable"),
)
pipeline = build_pipeline()
adapter = HaystackDurableAdapter(pipeline)
deployment = DurableDeployment(
    DEPLOYMENT_NAME,
    DEPLOYMENT_REVISION,
    store,
    DocumentRequest,
    run_document_analysis,
    kind=adapter.kind,
    result_model=AnalysisResult,
    resume_model=Approval,
    adapter=adapter,
)
runtime = DurableRuntime()
runtime.add(deployment)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await runtime.start()
    try:
        yield
    finally:
        await runtime.close()
        await redis.aclose()


app = FastAPI(title="Durable document API", lifespan=lifespan)
app.include_router(
    create_durable_router(deployment, owner_id_dependency=owner_id),
    prefix=f"/jobs/{DEPLOYMENT_NAME}",
)


@app.get("/health")
async def health(response: Response) -> dict[str, object]:
    report = await runtime.health()
    if not report["healthy"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return report
