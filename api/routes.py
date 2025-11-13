"""
FastAPI router factory for webhook endpoints.

Routes are created in a function to avoid importing FastAPI at module
import time (helps unit tests that don't have FastAPI installed).
"""
from typing import Any, Dict


def get_router():
    from fastapi import APIRouter, Request

    router = APIRouter()

    @router.get("/status")
    async def status() -> Dict[str, Any]:
        return {"status": "ok"}

    @router.post("/webhook/trade")
    async def webhook_trade(request: Request) -> Dict[str, Any]:
        payload = await request.json()
        integration = request.app.state.integration
        # call the integration's process_webhook; allow sync or async
        result = integration.process_webhook(payload)
        if hasattr(result, "__await__"):
            result = await result
        return result

    @router.post("/webhook/batch")
    async def webhook_batch(request: Request) -> Dict[str, Any]:
        payload = await request.json()
        integration = request.app.state.integration
        result = integration.process_webhook(payload)
        if hasattr(result, "__await__"):
            result = await result
        return result

    return router
