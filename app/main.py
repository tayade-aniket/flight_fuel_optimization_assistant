from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware

from app.dependencies import load_model_and_features
from app.routers import (
    dashboard,
    predict,
    trajectory,
    benchmarks,
    anomaly,
    simulator,
    optimizer,
    explainability,
    methodology,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load model once on startup, stash in app.state
    model, feature_names = load_model_and_features()
    app.state.model = model
    app.state.feature_names = feature_names
    yield


app = FastAPI(
    title="AeroFuel AI",
    description="Flight Fuel Optimization Assistant",
    version="2.0.0",
    lifespan=lifespan,
)

# middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# static files
app.mount("/static", StaticFiles(directory=Path("app/static")), name="static")

# routers
app.include_router(dashboard.router)
app.include_router(predict.router)
app.include_router(trajectory.router)
app.include_router(benchmarks.router)
app.include_router(anomaly.router)
app.include_router(simulator.router)
app.include_router(optimizer.router)
app.include_router(explainability.router)
app.include_router(methodology.router)
