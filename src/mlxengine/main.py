import argparse
import os

import uvicorn
from starlette.middleware import Middleware
from turboapi import TurboAPI

from .middleware.logging import RequestResponseLoggingMiddleware
from .routers import api_router

# Create a list of middleware
middlewares = [
    Middleware(RequestResponseLoggingMiddleware)
    # Add other middleware instances here if needed
]

app = TurboAPI(title="MLX Omni Server", middleware=middlewares)

app.include_router(api_router)


def build_parser():
    """Create and configure the argument parser for the server."""
    parser = argparse.ArgumentParser(description="MLX Omni Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to, defaults to 0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10240,
        help="Port to bind the server to, defaults to 10240",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers to use, defaults to 1",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set the logging level, defaults to info",
    )
    return parser


def start():
    """Start the MLX Omni Server."""
    parser = build_parser()
    args = parser.parse_args()

    # Set log level through environment variable
    os.environ["MLX_OMNI_LOG_LEVEL"] = args.log_level

    # Start server with uvicorn
    uvicorn.run(
        "mlxengine.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        use_colors=True,
        workers=args.workers,
    )
