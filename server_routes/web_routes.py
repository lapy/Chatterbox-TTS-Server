"""Web UI HTML and UI-helper JSON routes."""
import logging
from pathlib import Path
from typing import Dict, List

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

import engine
import utils
from config import get_full_config_for_template

logger = logging.getLogger(__name__)
UI_STATIC_PATH = Path(__file__).resolve().parent.parent / "ui"
templates = Jinja2Templates(directory=str(UI_STATIC_PATH))
router = APIRouter()

@router.get("/styles.css", include_in_schema=False)
async def get_main_styles():
    styles_file = UI_STATIC_PATH / "styles.css"
    if styles_file.is_file():
        return FileResponse(styles_file)
    raise HTTPException(status_code=404, detail="styles.css not found")


@router.get("/script.js", include_in_schema=False)
async def get_main_script():
    script_file = UI_STATIC_PATH / "script.js"
    if script_file.is_file():
        return FileResponse(script_file)
    raise HTTPException(status_code=404, detail="script.js not found")

templates = Jinja2Templates(directory=str(UI_STATIC_PATH))

# --- Main UI Route ---
@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_web_ui(request: Request):
    """Serves the main web interface (index.html)."""
    logger.info("Request received for main UI page ('/').")
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e_render:
        logger.error(f"Error rendering main UI page: {e_render}", exc_info=True)
        return HTMLResponse(
            "<html><body><h1>Internal Server Error</h1><p>Could not load the TTS interface. "
            "Please check server logs for more details.</p></body></html>",
            status_code=500,
        )


# --- API Endpoint for Model Information ---
@router.get("/api/model-info", tags=["Model Information"])
async def get_model_info_endpoint():
    """
    Returns detailed information about the currently loaded TTS model.
    This endpoint is used by the UI to display model status and
    conditionally show features like paralinguistic tags.
    """
    logger.debug("Request received for /api/model-info")
    try:
        model_info = engine.get_model_info()
        return model_info
    except Exception as e:
        logger.error(f"Error getting model info: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve model information"
        )


# --- API Endpoint for Initial UI Data ---
@router.get("/api/ui/initial-data", tags=["UI Helpers"])
async def get_ui_initial_data():
    """
    Provides all necessary initial data for the UI to render,
    including configuration, file lists, presets, and model information.
    """
    logger.info("Request received for /api/ui/initial-data.")
    try:
        full_config = get_full_config_for_template()
        reference_files = utils.get_valid_reference_files()
        predefined_voices = utils.get_predefined_voices()

        # Get model information for UI
        model_info = engine.get_model_info()

        loaded_presets = []
        presets_file = UI_STATIC_PATH / "presets.yaml"
        if presets_file.exists():
            with open(presets_file, "r", encoding="utf-8") as f:
                yaml_content = yaml.safe_load(f)
                if isinstance(yaml_content, list):
                    loaded_presets = yaml_content
                else:
                    logger.warning(
                        f"Invalid format in {presets_file}. Expected a list, got {type(yaml_content)}."
                    )
        else:
            logger.info(
                f"Presets file not found: {presets_file}. No presets will be loaded for initial data."
            )

        initial_gen_result_placeholder = {
            "outputUrl": None,
            "filename": None,
            "genTime": None,
            "submittedVoiceMode": None,
            "submittedPredefinedVoice": None,
            "submittedCloneFile": None,
        }

        return {
            "config": full_config,
            "reference_files": reference_files,
            "predefined_voices": predefined_voices,
            "presets": loaded_presets,
            "initial_gen_result": initial_gen_result_placeholder,
            "model_info": model_info,  # NEW: Include model information
        }
    except Exception as e:
        logger.error(f"Error preparing initial UI data for API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to load initial data for UI."
        )

# --- UI Helper API Endpoints ---
@router.get("/get_reference_files", response_model=List[str], tags=["UI Helpers"])
async def get_reference_files_api():
    """Returns a list of valid reference audio filenames (.wav, .mp3)."""
    logger.debug("Request for /get_reference_files.")
    try:
        return utils.get_valid_reference_files()
    except Exception as e:
        logger.error(f"Error getting reference files for API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve reference audio files."
        )


@router.get(
    "/get_predefined_voices", response_model=List[Dict[str, str]], tags=["UI Helpers"]
)
async def get_predefined_voices_api():
    """Returns a list of predefined voices with display names and filenames."""
    logger.debug("Request for /get_predefined_voices.")
    try:
        return utils.get_predefined_voices()
    except Exception as e:
        logger.error(f"Error getting predefined voices for API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve predefined voices list."
        )

