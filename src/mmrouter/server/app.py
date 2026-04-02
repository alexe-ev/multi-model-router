"""FastAPI app: OpenAI-compatible REST API for mmrouter."""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from mmrouter import __version__
from mmrouter.providers.litellm_provider import ProviderError
from mmrouter.router.engine import Router
from mmrouter.server.auth import verify_api_key
from mmrouter.server.models import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelInfo,
    ModelListResponse,
    UsageInfo,
)


def create_app(
    config_path: str | Path = "configs/default.yaml",
    db_path: str | Path = "mmrouter.db",
) -> FastAPI:
    """Create the OpenAI-compatible REST API FastAPI app."""
    router_instance: Router | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal router_instance
        router_instance = Router(config_path, db_path=db_path)
        app.state.router = router_instance
        yield
        if router_instance:
            router_instance.close()

    app = FastAPI(
        title="mmrouter API",
        version=__version__,
        lifespan=lifespan,
    )

    def _get_router(request: Request) -> Router:
        return request.app.state.router

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": __version__}

    @app.get("/v1/models", dependencies=[Depends(verify_api_key)])
    async def list_models(request: Request):
        router = _get_router(request)
        config = router.get_config()

        models_seen: set[str] = set()
        model_list: list[ModelInfo] = []

        # Always include "auto" as the routing model
        model_list.append(ModelInfo(id="auto", owned_by="mmrouter"))

        for complexity_routes in config.routes.values():
            for route in complexity_routes.values():
                for m in [route.model] + route.fallbacks:
                    if m not in models_seen:
                        models_seen.add(m)
                        model_list.append(ModelInfo(id=m, owned_by="config"))

        return ModelListResponse(data=model_list)

    @app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        router = _get_router(request)
        messages = [{"role": m.role, "content": m.content or ""} for m in body.messages]

        # Build kwargs for provider
        provider_kwargs: dict = {}
        if body.temperature is not None:
            provider_kwargs["temperature"] = body.temperature
        if body.max_tokens is not None:
            provider_kwargs["max_tokens"] = body.max_tokens
        if body.top_p is not None:
            provider_kwargs["top_p"] = body.top_p
        if body.stop is not None:
            provider_kwargs["stop"] = body.stop

        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if body.stream:
            return _handle_stream(router, messages, body, provider_kwargs, request_id, created)

        return _handle_non_stream(router, messages, body, provider_kwargs, request_id, created)

    def _handle_non_stream(
        router: Router,
        messages: list[dict],
        body: ChatCompletionRequest,
        provider_kwargs: dict,
        request_id: str,
        created: int,
    ):
        try:
            if body.model == "auto" or not body.model:
                result = router.route_messages(messages, **provider_kwargs)
                classification = result.classification

                response = ChatCompletionResponse(
                    id=request_id,
                    created=created,
                    model=result.model_used,
                    choices=[
                        ChatCompletionChoice(
                            message=ChatMessage(role="assistant", content=result.completion.content),
                            finish_reason="stop",
                        )
                    ],
                    usage=UsageInfo(
                        prompt_tokens=result.completion.tokens_in,
                        completion_tokens=result.completion.tokens_out,
                        total_tokens=result.completion.tokens_in + result.completion.tokens_out,
                    ),
                )

                headers = {
                    "X-MMRouter-Complexity": classification.complexity.value,
                    "X-MMRouter-Category": classification.category.value,
                    "X-MMRouter-Confidence": str(classification.confidence),
                    "X-MMRouter-Model": result.model_used,
                    "X-MMRouter-Fallback": str(result.fallback_used).lower(),
                    "X-MMRouter-Escalated": str(result.escalated).lower(),
                    "X-MMRouter-Budget-Downgraded": str(result.budget_downgraded).lower(),
                }

                from fastapi.responses import JSONResponse

                return JSONResponse(
                    content=response.model_dump(),
                    headers=headers,
                )
            else:
                # Explicit model: bypass routing, call provider directly
                completion = router._provider.complete_messages(
                    messages, body.model, **provider_kwargs
                )

                response = ChatCompletionResponse(
                    id=request_id,
                    created=created,
                    model=completion.model,
                    choices=[
                        ChatCompletionChoice(
                            message=ChatMessage(role="assistant", content=completion.content),
                            finish_reason="stop",
                        )
                    ],
                    usage=UsageInfo(
                        prompt_tokens=completion.tokens_in,
                        completion_tokens=completion.tokens_out,
                        total_tokens=completion.tokens_in + completion.tokens_out,
                    ),
                )

                return response

        except ProviderError as e:
            raise HTTPException(status_code=502, detail=f"Upstream provider error: {e}")
        except (ValueError, RuntimeError) as e:
            raise HTTPException(status_code=502, detail=str(e))

    def _handle_stream(
        router: Router,
        messages: list[dict],
        body: ChatCompletionRequest,
        provider_kwargs: dict,
        request_id: str,
        created: int,
    ):
        try:
            if body.model == "auto" or not body.model:
                classification, model, fallback_used, escalated, budget_downgraded, chunks = (
                    router.route_messages_stream(messages, **provider_kwargs)
                )
                extra_headers = {
                    "X-MMRouter-Complexity": classification.complexity.value,
                    "X-MMRouter-Category": classification.category.value,
                    "X-MMRouter-Confidence": str(classification.confidence),
                    "X-MMRouter-Model": model,
                    "X-MMRouter-Fallback": str(fallback_used).lower(),
                    "X-MMRouter-Escalated": str(escalated).lower(),
                    "X-MMRouter-Budget-Downgraded": str(budget_downgraded).lower(),
                }
            else:
                chunks = router._provider.stream_messages(
                    messages, body.model, **provider_kwargs
                )
                model = body.model
                extra_headers = {}

        except ProviderError as e:
            raise HTTPException(status_code=502, detail=f"Upstream provider error: {e}")
        except (ValueError, RuntimeError) as e:
            raise HTTPException(status_code=502, detail=str(e))

        def generate():
            # First chunk with role
            first_chunk = ChatCompletionChunk(
                id=request_id,
                created=created,
                model=model,
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(role="assistant"),
                    )
                ],
            )
            yield f"data: {first_chunk.model_dump_json()}\n\n"

            try:
                for chunk in chunks:
                    chunk_obj = ChatCompletionChunk(
                        id=request_id,
                        created=created,
                        model=chunk.model or model,
                        choices=[
                            ChatCompletionChunkChoice(
                                delta=ChatCompletionChunkDelta(content=chunk.content),
                                finish_reason=chunk.finish_reason,
                            )
                        ],
                    )
                    yield f"data: {chunk_obj.model_dump_json()}\n\n"
            except (ProviderError, RuntimeError) as e:
                error_data = json.dumps({"error": {"message": str(e), "type": "server_error"}})
                yield f"data: {error_data}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                **extra_headers,
            },
        )

    return app
