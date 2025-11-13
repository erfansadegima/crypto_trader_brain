"""
Entry-point FastAPI application factory for the trading brain webhook.

This file uses lazy imports so importing the module doesn't require FastAPI
to be installed (helpful for running tests in environments without FastAPI).
"""
from typing import Any, Dict


def create_app() -> Any:
    # Lazy imports
    from fastapi import FastAPI
    from api.routes import get_router
    app = FastAPI(title="Crypto Trading Brain Webhook")

    # attach router
    app.include_router(get_router())

    @app.on_event("startup")
    async def startup_event():
        # instantiate integration/brain lazily to avoid heavy work on import
        try:
            from n8n_integration import N8NIntegration
        except Exception:
            # If the integration module isn't available, put a minimal stub
            class _Stub:
                async def process_webhook(self, payload: Dict) -> Dict:
                    return {"status": "error", "reason": "integration missing"}

            app.state.integration = _Stub()
            return

        config_values = {}
        app.state.integration = N8NIntegration(**config_values)

    return app


app = create_app()


if __name__ == "__main__":
    # run with: python app.py
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
