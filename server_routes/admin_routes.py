"""Configuration and file-upload routes (optionally HTTP Basic protected)."""
import logging
import shutil
from typing import Dict, List

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

import engine
import utils
from admin_auth import verify_admin_access
from config import config_manager, get_predefined_voices_path, get_reference_audio_path
from models import UpdateStatusResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Configuration Management API Endpoints ---
@router.post(
    "/save_settings",
    response_model=UpdateStatusResponse,
    tags=["Configuration"],
    dependencies=[Depends(verify_admin_access)],
)
async def save_settings_endpoint(request: Request):
    """
    Saves partial configuration updates to the config.yaml file.
    Merges the update with the current configuration.
    """
    logger.info("Request received for /save_settings.")
    try:
        partial_update = await request.json()
        if not isinstance(partial_update, dict):
            raise ValueError("Request body must be a JSON object for /save_settings.")
        logger.debug(f"Received partial config data to save: {partial_update}")

        if config_manager.update_and_save(partial_update):
            restart_needed = any(
                key in partial_update
                for key in ["server", "tts_engine", "paths", "model"]
            )
            message = "Settings saved successfully."
            if restart_needed:
                message += " A server restart may be required for some changes to take full effect."
            return UpdateStatusResponse(message=message, restart_needed=restart_needed)
        else:
            logger.error(
                "Failed to save configuration via config_manager.update_and_save."
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to save configuration file due to an internal error.",
            )
    except ValueError as ve:
        logger.error(f"Invalid data format for /save_settings: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid request data: {str(ve)}")
    except Exception as e:
        logger.error(f"Error processing /save_settings request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during settings save: {str(e)}",
        )


@router.post(
    "/reset_settings",
    response_model=UpdateStatusResponse,
    tags=["Configuration"],
    dependencies=[Depends(verify_admin_access)],
)
async def reset_settings_endpoint():
    """Resets the configuration in config.yaml back to hardcoded defaults."""
    logger.warning("Request received to reset all configurations to default values.")
    try:
        if config_manager.reset_and_save():
            logger.info("Configuration successfully reset to defaults and saved.")
            return UpdateStatusResponse(
                message="Configuration reset to defaults. Please reload the page. A server restart may be beneficial.",
                restart_needed=True,
            )
        else:
            logger.error("Failed to reset and save configuration via config_manager.")
            raise HTTPException(
                status_code=500, detail="Failed to reset and save configuration file."
            )
    except Exception as e:
        logger.error(f"Error processing /reset_settings request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during settings reset: {str(e)}",
        )


@router.post(
    "/restart_server",
    response_model=UpdateStatusResponse,
    tags=["Configuration"],
    dependencies=[Depends(verify_admin_access)],
)
async def restart_server_endpoint():
    """
    Triggers a hot-swap of the TTS model engine.
    Unloads the current model, clears VRAM, and loads the model defined in config.
    """
    logger.info("Request received for /restart_server (Model Hot-Swap).")

    try:
        # Attempt to reload the engine with the new configuration
        success = engine.reload_model()

        if success:
            model_info = engine.get_model_info()
            new_model_name = model_info.get("class_name", "Unknown Model")
            new_model_type = model_info.get("type", "unknown")
            message = f"Model hot-swap successful. Now running: {new_model_name} ({new_model_type})"
            logger.info(message)

            # restart_needed=False because we just performed the hot-swap successfully
            return UpdateStatusResponse(message=message, restart_needed=False)
        else:
            error_msg = "Model reload failed. The server may be in an inconsistent state. Check logs for details."
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Critical error during model hot-swap: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during model reload: {str(e)}",
        )


@router.post(
    "/api/unload",
    tags=["Configuration"],
    dependencies=[Depends(verify_admin_access)],
)
async def unload_model_endpoint():
    """
    Unloads the TTS model and releases all CUDA/GPU memory.
    The model will need to be reloaded (via /restart_server) before TTS requests can be processed.
    """
    logger.info("Request received for /api/unload (Model Unload).")

    try:
        success = engine.unload_model()

        if success:
            logger.info("Model successfully unloaded and GPU memory released.")
            return {"status": "unloaded"}
        else:
            error_msg = "Model unload failed. Check logs for details."
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Critical error during model unload: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during model unload: {str(e)}",
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


# --- File Upload Endpoints ---
@router.post(
    "/upload_reference",
    tags=["File Management"],
    dependencies=[Depends(verify_admin_access)],
)
async def upload_reference_audio_endpoint(files: List[UploadFile] = File(...)):
    """
    Handles uploading of reference audio files (.wav, .mp3) for voice cloning.
    Validates files and saves them to the configured reference audio path.
    """
    logger.info(f"Request to /upload_reference with {len(files)} file(s).")
    ref_path = get_reference_audio_path(ensure_absolute=True)
    uploaded_filenames_successfully: List[str] = []
    upload_errors: List[Dict[str, str]] = []

    for file in files:
        if not file.filename:
            upload_errors.append(
                {"filename": "Unknown", "error": "File received with no filename."}
            )
            logger.warning("Upload attempt with no filename.")
            continue

        safe_filename = utils.sanitize_filename(file.filename)
        destination_path = ref_path / safe_filename

        try:
            if not (
                safe_filename.lower().endswith(".wav")
                or safe_filename.lower().endswith(".mp3")
            ):
                raise ValueError("Invalid file type. Only .wav and .mp3 are allowed.")

            if destination_path.exists():
                logger.info(
                    f"Reference file '{safe_filename}' already exists. Skipping duplicate upload."
                )
                if safe_filename not in uploaded_filenames_successfully:
                    uploaded_filenames_successfully.append(safe_filename)
                continue

            with open(destination_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(
                f"Successfully saved uploaded reference file to: {destination_path}"
            )

            max_duration = config_manager.get_int(
                "audio_output.max_reference_duration_sec", 30
            )
            is_valid, validation_msg = utils.validate_reference_audio(
                destination_path, max_duration
            )
            if not is_valid:
                logger.warning(
                    f"Uploaded file '{safe_filename}' failed validation: {validation_msg}. Deleting."
                )
                destination_path.unlink(missing_ok=True)
                upload_errors.append(
                    {"filename": safe_filename, "error": validation_msg}
                )
            else:
                uploaded_filenames_successfully.append(safe_filename)

        except Exception as e_upload:
            error_msg = f"Error processing file '{file.filename}': {str(e_upload)}"
            logger.error(error_msg, exc_info=True)
            upload_errors.append({"filename": file.filename, "error": str(e_upload)})
        finally:
            await file.close()

    all_current_reference_files = utils.get_valid_reference_files()
    response_data = {
        "message": f"Processed {len(files)} file(s).",
        "uploaded_files": uploaded_filenames_successfully,
        "all_reference_files": all_current_reference_files,
        "errors": upload_errors,
    }
    status_code = (
        200 if not upload_errors or len(uploaded_filenames_successfully) > 0 else 400
    )
    if upload_errors:
        logger.warning(
            f"Upload to /upload_reference completed with {len(upload_errors)} error(s)."
        )
    return JSONResponse(content=response_data, status_code=status_code)


@router.post(
    "/upload_predefined_voice",
    tags=["File Management"],
    dependencies=[Depends(verify_admin_access)],
)
async def upload_predefined_voice_endpoint(files: List[UploadFile] = File(...)):
    """
    Handles uploading of predefined voice files (.wav, .mp3).
    Validates files and saves them to the configured predefined voices path.
    """
    logger.info(f"Request to /upload_predefined_voice with {len(files)} file(s).")
    predefined_voices_path = get_predefined_voices_path(ensure_absolute=True)
    uploaded_filenames_successfully: List[str] = []
    upload_errors: List[Dict[str, str]] = []

    for file in files:
        if not file.filename:
            upload_errors.append(
                {"filename": "Unknown", "error": "File received with no filename."}
            )
            logger.warning("Upload attempt for predefined voice with no filename.")
            continue

        safe_filename = utils.sanitize_filename(file.filename)
        destination_path = predefined_voices_path / safe_filename

        try:
            if not (
                safe_filename.lower().endswith(".wav")
                or safe_filename.lower().endswith(".mp3")
            ):
                raise ValueError(
                    "Invalid file type. Only .wav and .mp3 are allowed for predefined voices."
                )

            if destination_path.exists():
                logger.info(
                    f"Predefined voice file '{safe_filename}' already exists. Skipping duplicate upload."
                )
                if safe_filename not in uploaded_filenames_successfully:
                    uploaded_filenames_successfully.append(safe_filename)
                continue

            with open(destination_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(
                f"Successfully saved uploaded predefined voice file to: {destination_path}"
            )
            # Basic validation (can be extended if predefined voices have specific requirements)
            is_valid, validation_msg = utils.validate_reference_audio(
                destination_path, max_duration_sec=None
            )  # No duration limit for predefined
            if not is_valid:
                logger.warning(
                    f"Uploaded predefined voice '{safe_filename}' failed basic validation: {validation_msg}. Deleting."
                )
                destination_path.unlink(missing_ok=True)
                upload_errors.append(
                    {"filename": safe_filename, "error": validation_msg}
                )
            else:
                uploaded_filenames_successfully.append(safe_filename)

        except Exception as e_upload:
            error_msg = f"Error processing predefined voice file '{file.filename}': {str(e_upload)}"
            logger.error(error_msg, exc_info=True)
            upload_errors.append({"filename": file.filename, "error": str(e_upload)})
        finally:
            await file.close()

    all_current_predefined_voices = (
        utils.get_predefined_voices()
    )  # Fetches formatted list
    response_data = {
        "message": f"Processed {len(files)} predefined voice file(s).",
        "uploaded_files": uploaded_filenames_successfully,  # List of raw filenames uploaded
        "all_predefined_voices": all_current_predefined_voices,  # Formatted list for UI
        "errors": upload_errors,
    }
    status_code = (
        200 if not upload_errors or len(uploaded_filenames_successfully) > 0 else 400
    )
    if upload_errors:
        logger.warning(
            f"Upload to /upload_predefined_voice completed with {len(upload_errors)} error(s)."
        )
    return JSONResponse(content=response_data, status_code=status_code)

