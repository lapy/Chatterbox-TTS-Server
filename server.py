# File: server.py
# Main FastAPI application for the TTS Server.
# Route handlers live under server_routes/; this module wires the app, static files, and lifespan.

import logging
import logging.handlers
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import engine
from config import (
    config_manager,
    get_app_version,
    get_host,
    get_log_file_path,
    get_output_path,
    get_port,
    get_predefined_voices_path,
    get_reference_audio_path,
    get_ssl_config,
    get_ui_title,
)

# --- Logging Configuration ---
log_file_path_obj = get_log_file_path()
log_file_max_size_mb = config_manager.get_int("server.log_file_max_size_mb", 10)
log_backup_count = config_manager.get_int("server.log_file_backup_count", 5)

log_file_path_obj.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.handlers.RotatingFileHandler(
            str(log_file_path_obj),
            maxBytes=log_file_max_size_mb * 1024 * 1024,
            backupCount=log_backup_count,
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

startup_complete_event = threading.Event()


def _delayed_browser_open(host: str, port: int):
    """
    Waits for the startup_complete_event, then opens the web browser
    to the server's main page after a short delay.
    """
    try:
        startup_complete_event.wait(timeout=30)
        if not startup_complete_event.is_set():
            logger.warning(
                "Server startup did not signal completion within timeout. Browser will not be opened automatically."
            )
            return

        time.sleep(1.5)
        display_host = "localhost" if host == "0.0.0.0" else host
        browser_url = f"http://{display_host}:{port}/"
        logger.info(f"Attempting to open web browser to: {browser_url}")
        webbrowser.open(browser_url)
    except Exception as e:
        logger.error(f"Failed to open browser automatically: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown events."""
    logger.info("TTS Server: Initializing application...")
    try:
        logger.info(f"Configuration loaded. Log file at: {get_log_file_path()}")

        paths_to_ensure = [
            get_output_path(),
            get_reference_audio_path(),
            get_predefined_voices_path(),
            Path("ui"),
            config_manager.get_path(
                "paths.model_cache", "./model_cache", ensure_absolute=True
            ),
        ]
        for p in paths_to_ensure:
            p.mkdir(parents=True, exist_ok=True)

        if not engine.load_model():
            logger.critical(
                "CRITICAL: TTS Model failed to load on startup. Server might not function correctly."
            )
        else:
            logger.info("TTS Model loaded successfully via engine.")
            host_address = get_host()
            server_port = get_port()
            browser_thread = threading.Thread(
                target=lambda: _delayed_browser_open(host_address, server_port),
                daemon=True,
            )
            browser_thread.start()

        _h = get_host()
        if _h in ("0.0.0.0", "::", "[::]") and not config_manager.get_bool(
            "server.use_auth", False
        ):
            logger.warning(
                "Server is bound to all interfaces (%s) without server.use_auth. "
                "Anyone who can reach this port can change settings, upload files, "
                "and use the TTS API. Use 127.0.0.1, enable use_auth, or place the "
                "service behind a trusted reverse proxy.",
                _h,
            )

        logger.info("Application startup sequence complete.")
        startup_complete_event.set()
        yield
    except Exception as e_startup:
        logger.error(
            f"FATAL ERROR during application startup: {e_startup}", exc_info=True
        )
        startup_complete_event.set()
        yield
    finally:
        logger.info("TTS Server: Application shutdown sequence initiated...")
        logger.info("TTS Server: Application shutdown complete.")


app = FastAPI(
    title=get_ui_title(),
    description="Text-to-Speech server with advanced UI and API capabilities.",
    version=get_app_version(),
    lifespan=lifespan,
)

if config_manager.get_bool("server.cors_allow_all", False):
    logger.warning(
        "server.cors_allow_all is enabled: using permissive CORS. "
        "Disable on untrusted networks."
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*", "null"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
else:
    _raw_origins = config_manager.get("server.cors_origins", []) or []
    if isinstance(_raw_origins, str):
        _cors_origins = [x.strip() for x in _raw_origins.split(",") if x.strip()]
    elif isinstance(_raw_origins, list):
        _cors_origins = [str(x).strip() for x in _raw_origins if str(x).strip()]
    else:
        _cors_origins = []
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

ui_static_path = Path(__file__).parent / "ui"
if ui_static_path.is_dir():
    app.mount("/ui", StaticFiles(directory=ui_static_path), name="ui_static_assets")
else:
    logger.warning(
        f"UI static assets directory not found at '{ui_static_path}'. UI may not load correctly."
    )

if (ui_static_path / "vendor").is_dir():
    app.mount(
        "/vendor",
        StaticFiles(directory=ui_static_path / "vendor"),
        name="vendor_files",
    )
else:
    logger.warning(
        f"Vendor directory not found at '{ui_static_path}' /vendor. Wavesurfer might not load."
    )

outputs_static_path = get_output_path(ensure_absolute=True)
try:
    app.mount(
        "/outputs",
        StaticFiles(directory=str(outputs_static_path)),
        name="generated_outputs",
    )
except RuntimeError as e_mount_outputs:
    logger.error(
        f"Failed to mount /outputs directory '{outputs_static_path}': {e_mount_outputs}. "
        "Output files may not be accessible via URL."
    )

from server_routes.admin_routes import router as admin_router
from server_routes.health import router as health_router
from server_routes.openai_routes import router as openai_router
from server_routes.tts_routes import router as tts_router
from server_routes.web_routes import router as web_router

app.include_router(health_router)
app.include_router(web_router)
app.include_router(admin_router)
app.include_router(tts_router)
app.include_router(openai_router)

if __name__ == "__main__":
    server_host = get_host()
    server_port = get_port()
    ssl_kwargs = get_ssl_config()
    protocol = "https" if ssl_kwargs else "http"

    logger.info(
        f"Starting TTS Server directly on {protocol}://{server_host}:{server_port}"
    )
    logger.info(
        f"API documentation will be available at {protocol}://{server_host}:{server_port}/docs"
    )
    logger.info(
        f"Web UI will be available at {protocol}://{server_host}:{server_port}/"
    )

    import uvicorn

    uvicorn.run(
        "server:app",
        host=server_host,
        port=server_port,
        log_level="info",
        workers=1,
        reload=False,
        **ssl_kwargs,
    )
