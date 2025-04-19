import argparse
import os

import uvicorn
# Replace FastAPI with TurboAPI
from turboapi import TurboAPI
# from fastapi import FastAPI

# Remove FastAPI middleware import
# from .middleware.logging import RequestResponseLoggingMiddleware
# Import the route handler function directly
from .chat.router import create_chat_completion
# Import the dictionary of sub-apps to mount (excluding chat)
from .routers import sub_apps

# Instantiate the main TurboAPI application
app = TurboAPI(title="MLX Omni Server - TurboAPI")

# Remove FastAPI middleware usage
# app.add_middleware(
#     RequestResponseLoggingMiddleware,
#     # exclude_paths=["/health"]
# )

# Remove FastAPI router inclusion
# app.include_router(api_router)

# Define chat routes directly on the main app instance
@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def main_create_chat_completion(request):
    # Call the original handler function
    return await create_chat_completion(request)

# Mount the sub-applications from routers.py
# Using app.mount as confirmed by TurboAPI documentation for sub-applications.
# This loop will now be empty or handle other modules if they exist
for prefix, sub_app in sub_apps.items():
    try:
        app.mount(prefix, sub_app) # Use mount for TurboAPI sub-apps
        print(f"Mounted sub-app: {sub_app.title if hasattr(sub_app, 'title') else 'sub-app'} at {prefix}")
    except AttributeError:
        print(f"Error: Could not mount sub-app at {prefix}. "
              f"Does TurboAPI instance have a 'mount()' method? Check API.")
    except Exception as e:
        print(f"Error mounting sub-app at {prefix}: {e}")


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
    os.environ["MLX_OMNI_LOG_LEVEL"] = args.log_level # Keep this, might be useful

    # Update uvicorn run command to point to the correct module:app
    # Make sure your project structure allows importing mlxengine.main
    # If running from the root of the project, this might need adjustment
    # based on how you structure your runnable application.
    # Assuming `src` is in PYTHONPATH or running from within `src`.
    uvicorn.run(
        "mlxengine.main:app", # Changed from "mlx_omni_server.main:app"
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        use_colors=True,
        workers=args.workers,
        # Add reload=True for development if needed
    )
