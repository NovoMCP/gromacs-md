"""Register all API routers."""

from fastapi import FastAPI

from . import health, audit, metal, simulate, status


def register_routes(app: FastAPI):
    app.include_router(health.router)
    app.include_router(audit.router)
    app.include_router(metal.router)
    app.include_router(simulate.router)
    app.include_router(status.router)
