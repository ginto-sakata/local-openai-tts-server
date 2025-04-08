import os
import time
import logging
import io
import wave
import pathlib
import subprocess
import yaml
import torch
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from dotenv import dotenv_values
from huggingface_hub import hf_hub_download

try:
    from piper.voice import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG_ENV_PATH = "config.env"
VOICES_DEF_PATH = "voices_definitions.yaml"

app_config = {}
voice_definitions = {}
loaded_silero_models = {}
loaded_piper_voices = {}
enabled_silero_voice_map = {}
enabled_piper_voice_map = {}
active_voices_api_list = []

# --- Global variable for selected device ---
selected_torch_device = torch.device('cpu') # Default to CPU, will be updated in startup

app = FastAPI()

def parse_list_env(env_var):
    return {item.strip() for item in env_var.split(',') if item.strip()} if env_var else set()

def map_silero_quality_to_sr(quality_str, supported_rates):
    mapping = {'high': 48000, 'medium': 24000, 'low': 8000}
    target_sr = mapping.get(quality_str.lower(), 48000) # Default to high
    if supported_rates and target_sr not in supported_rates:
        logger.warning(f"Chosen Silero quality '{quality_str}' maps to {target_sr}Hz, which is not listed as supported {supported_rates}. Attempting anyway.")
    return target_sr

def ensure_silero_helpers(repo_dir, git_url):
    repo_path = pathlib.Path(repo_dir)
    hubconf_path = repo_path / "hubconf.py"
    # Simple check: if hubconf is missing, assume helpers need cloning/copying
    if not hubconf_path.exists():
        logger.info(f"Silero helper files (hubconf.py) not found in {repo_dir}. Attempting to clone from {git_url}...")
        try:
            # Create parent if needed
            repo_path.parent.mkdir(parents=True, exist_ok=True)
            # Clone to a temporary location first, then copy necessary files
            temp_clone_dir = repo_path.parent / f"{repo_path.name}_temp_clone"
            if temp_clone_dir.exists():
                 import shutil
                 shutil.rmtree(temp_clone_dir)

            subprocess.run(['git', 'clone', '--depth', '1', git_url, str(temp_clone_dir)], check=True, capture_output=True)
            logger.info(f"Cloned Silero repo to temporary location: {temp_clone_dir}")

            # Ensure target repo_dir exists
            repo_path.mkdir(parents=True, exist_ok=True)

            # Selectively copy essential files (adjust if Silero structure changes)
            files_to_copy = ['hubconf.py', 'models.yml'] # Start with minimal set
            dirs_to_copy = ['src/silero'] # Copy relevant source code needed by hubconf

            for file_name in files_to_copy:
                 src_file = temp_clone_dir / file_name
                 if src_file.exists():
                     import shutil
                     shutil.copy2(src_file, repo_path / file_name)
                     logger.debug(f"Copied {file_name} to {repo_path}")

            for dir_name in dirs_to_copy:
                 src_dir = temp_clone_dir / dir_name
                 if src_dir.is_dir():
                     import shutil
                     target_dir = repo_path / dir_name
                     # Copy entire directory tree
                     shutil.copytree(src_dir, target_dir, dirs_exist_ok=True)
                     logger.debug(f"Copied directory {dir_name} to {repo_path}")

            # Clean up temporary clone
            import shutil
            shutil.rmtree(temp_clone_dir)
            logger.info("Cleaned up temporary Silero clone.")
            return True # Indicate success

        except Exception as e:
            logger.exception(f"ERROR: Failed to automatically clone/copy Silero helper files: {e}")
            return False # Indicate failure
    else:
        logger.debug(f"Silero helper files seem present in {repo_dir}.")
        return True # Helpers already exist

def ensure_silero_package(repo_dir, package_dest, package_url):
    package_path = pathlib.Path(repo_dir) / package_dest
    if not package_path.exists():
        logger.info(f"Silero model package {package_dest} not found. Downloading from {package_url}...")
        try:
            package_path.parent.mkdir(parents=True, exist_ok=True)
            # Use torch.hub.download_url_to_file for potential caching and better progress
            torch.hub.download_url_to_file(package_url, str(package_path), progress=True)
            logger.info(f"Successfully downloaded Silero model package to {package_path}.")
            return True
        except Exception as e:
            logger.exception(f"ERROR: Failed to download Silero model package: {e}")
            return False
    else:
        logger.debug(f"Silero model package {package_dest} found locally.")
        return True

def ensure_piper_voice(models_dir, hf_repo_id, subpath_no_ext):
    base_path = pathlib.Path(models_dir) / subpath_no_ext
    onnx_path = base_path.with_suffix(".onnx")
    json_path = base_path.with_suffix(".onnx.json")

    files_exist = True
    try:
        if not onnx_path.exists():
            logger.info(f"Piper model file {onnx_path.name} not found. Downloading...")
            # Make sure parent directories exist before download
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(repo_id=hf_repo_id, filename=str(pathlib.Path(subpath_no_ext).with_suffix(".onnx")),
                            local_dir=models_dir, local_dir_use_symlinks=False, resume_download=True)
            logger.info(f"Downloaded {onnx_path.name}.")
        if not json_path.exists():
            logger.info(f"Piper config file {json_path.name} not found. Downloading...")
            # Make sure parent directories exist before download
            json_path.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(repo_id=hf_repo_id, filename=str(pathlib.Path(subpath_no_ext).with_suffix(".onnx.json")),
                             local_dir=models_dir, local_dir_use_symlinks=False, resume_download=True)
            logger.info(f"Downloaded {json_path.name}.")
    except Exception as e:
        logger.exception(f"ERROR: Failed to download Piper voice file for {subpath_no_ext}: {e}")
        files_exist = False

    return files_exist, str(onnx_path), str(json_path)

@app.on_event("startup")
async def load_models_and_config():
    global app_config, voice_definitions, loaded_silero_models, loaded_piper_voices
    global enabled_silero_voice_map, enabled_piper_voice_map, active_voices_api_list
    global selected_torch_device

    logger.info("Application startup: Loading config, definitions, and models...")

    # 1. Load Config (.env)
    try:
        config_path = os.path.abspath(CONFIG_ENV_PATH)
        if os.path.exists(config_path):
            app_config = dotenv_values(config_path)
            if not app_config: app_config = {} # Handle empty file case
            logger.info(f"Loaded app configuration from {config_path}")
        else:
             app_config = {} # No config file found, use defaults
             logger.info(f"Config file not found at {config_path}. Using defaults.")
    except Exception as e:
        logger.exception(f"ERROR loading config {CONFIG_ENV_PATH}: {e}")
        app_config = {} # Reset to defaults on error

    # --- Configure Torch Device and Threads EARLY (using defaults if not set) ---
    try:
        # Configure CPU Threads
        cpu_threads_str = app_config.get("TORCH_CPU_THREADS", "0") # Default to '0'
        num_threads = int(cpu_threads_str)
        if num_threads > 0:
            torch.set_num_threads(num_threads)
            logger.info(f"PyTorch CPU threads set to: {num_threads} (from config)")
        else:
            # Only log default if it was explicitly '0' or not set
            if cpu_threads_str == "0":
                 logger.info(f"Using default PyTorch CPU threads (set to 0 in config).")
            else:
                 logger.info(f"Using default PyTorch CPU threads (TORCH_CPU_THREADS not set or invalid).")

        # Configure Device (CPU/CUDA)
        device_preference = app_config.get("DEVICE", "auto").lower() # Default to 'auto'
        if device_preference == "cuda":
            if torch.cuda.is_available():
                selected_torch_device = torch.device('cuda')
                logger.info("CUDA device selected (specified in config and available).")
            else:
                logger.warning("CUDA specified in config but torch.cuda.is_available() is False. Falling back to CPU.")
                selected_torch_device = torch.device('cpu')
        elif device_preference == "auto":
            if torch.cuda.is_available():
                selected_torch_device = torch.device('cuda')
                logger.info("CUDA device selected automatically (available).")
            else:
                selected_torch_device = torch.device('cpu')
                logger.info("CPU device selected automatically (CUDA not available).")
        else: # Assume 'cpu' or other invalid preference defaults to CPU
            selected_torch_device = torch.device('cpu')
            if device_preference != 'cpu':
                 logger.warning(f"Invalid DEVICE setting '{device_preference}' in config. Using CPU.")
            else:
                 logger.info("CPU device selected (specified in config).")

        # Log torch version and device details for debugging
        logger.info(f"PyTorch version: {torch.__version__}")
        if selected_torch_device.type == 'cuda':
             logger.info(f"Using CUDA device: {torch.cuda.get_device_name(selected_torch_device)}")

    except ValueError:
        logger.error(f"Invalid value for TORCH_CPU_THREADS: '{cpu_threads_str}'. Using default.")
        torch.set_num_threads(torch.get_num_threads()) # Ensure default is set if error occurs
    except Exception as e:
        logger.exception(f"Error during Torch performance setup: {e}. Using defaults.")
        selected_torch_device = torch.device('cpu') # Ensure CPU fallback on error
        torch.set_num_threads(torch.get_num_threads()) # Ensure default is set if error occurs
    # --- END Performance Setup ---


    # 2. Load Voice Definitions (YAML)
    def_path = VOICES_DEF_PATH # Use fixed path
    try:
        if not os.path.exists(def_path):
             logger.critical(f"CRITICAL ERROR: Voice definitions file '{def_path}' not found. Cannot proceed.")
             # Consider raising an exception or sys.exit() here to halt startup cleanly
             return # Stop loading process
        with open(def_path, 'r', encoding='utf-8') as f:
            voice_definitions = yaml.safe_load(f)
        if not voice_definitions: voice_definitions = {}
        logger.info(f"Loaded voice definitions from {def_path}")
    except Exception as e:
        logger.exception(f"CRITICAL ERROR loading voice definitions {def_path}. Cannot proceed.")
        # Consider raising an exception or sys.exit() here
        return # Stop loading process


    # 3. Parse Enabled Settings from Config
    enabled_engines = parse_list_env(app_config.get("ENABLED_ENGINES", "silero,piper")) # Default engines
    enabled_languages = parse_list_env(app_config.get("LANGUAGES", "en,ru")) # Default languages
    silero_quality = app_config.get("SILERO_QUALITY", "high") # Default quality
    default_voice_id = app_config.get("DEFAULT_VOICE", "") # No default voice by default

    # 4. Prepare to Load Models
    loaded_silero_models = {}
    loaded_piper_voices = {}
    enabled_silero_voice_map = {} # {voice_id: {'name': name, 'model_key': key, 'sample_rate': sr, 'lang': lang}}
    enabled_piper_voice_map = {} # {voice_id: {'name': name, 'lang': lang}}
    active_voices_api_list = []
    temp_api_voices = {} # Use dict to handle potential ID conflicts temporarily


    # 5. Load Silero Models
    if 'silero' in enabled_engines and 'silero' in voice_definitions:
        silero_def = voice_definitions['silero']
        repo_dir = os.path.abspath(silero_def.get('repo_dir', 'silero_repo_files'))
        git_url = silero_def.get('repo_git_url')

        helpers_ok = ensure_silero_helpers(repo_dir, git_url) if git_url else False

        if helpers_ok:
            for model_key, model_def in silero_def.get('models', {}).items():
                lang_code = model_def.get('language_code')
                if lang_code in enabled_languages:
                    env_var_key = f"ENABLED_SILERO_VOICES_{lang_code.upper()}"
                    # Get specific voices list; defaults to empty set if not present
                    enabled_ids_for_model = parse_list_env(app_config.get(env_var_key))
                    defined_voices = model_def.get('voices', {})
                    speakers_generic_count = model_def.get('speakers_generic_count', 0)
                    speakers_generic_pattern = model_def.get('speakers_generic_pattern')

                    # Build set of all possible defined voice IDs for this model
                    all_defined_ids_for_model = set(defined_voices.keys())
                    if speakers_generic_pattern and speakers_generic_count > 0:
                        all_defined_ids_for_model.update(
                            {speakers_generic_pattern.format(id=i) for i in range(speakers_generic_count)}
                        )

                    # Determine which specific voices from this model to ACTUALLY enable
                    if env_var_key in app_config:
                         # User specified a list, use only those defined voices that intersect
                         voices_to_enable_this_model = enabled_ids_for_model.intersection(all_defined_ids_for_model)
                         if not voices_to_enable_this_model:
                              logger.info(f"Silero model '{model_key}' ({lang_code}) specified but no valid voices listed in {env_var_key} match definitions. Skipping model load.")
                              continue
                    else:
                         # No specific list in .env, enable ALL defined voices for this model
                         voices_to_enable_this_model = all_defined_ids_for_model
                         logger.info(f"{env_var_key} not found in config.env, enabling all defined Silero voices for {lang_code}: {len(voices_to_enable_this_model)} voices.")

                    if not voices_to_enable_this_model:
                         logger.info(f"No voices to enable for Silero model '{model_key}'. Skipping model load.")
                         continue

                    logger.info(f"Attempting to load Silero model package '{model_key}' for language '{lang_code}'...")
                    pkg_dest = model_def.get('package_dest')
                    pkg_url = model_def.get('package_url')
                    if pkg_dest and pkg_url and ensure_silero_package(repo_dir, pkg_dest, pkg_url):
                        try:
                            model_id_for_load = model_def.get('model_id', model_key)
                            model_tuple = torch.hub.load(
                                repo_or_dir=repo_dir, model='silero_tts',
                                language=lang_code,
                                speaker=model_id_for_load,
                                source='local', trust_repo=True
                            )
                            loaded_model = model_tuple[0] if isinstance(model_tuple, tuple) else model_tuple
                            loaded_model.to(selected_torch_device) # Move to selected device
                            logger.info(f"Successfully loaded Silero model package '{model_key}' and moved to device '{selected_torch_device}'.")
                            loaded_silero_models[model_key] = loaded_model

                            # Add enabled voices from this package to the map
                            supported_rates = model_def.get('supported_sample_rates', [])
                            sample_rate = map_silero_quality_to_sr(silero_quality, supported_rates)

                            if speakers_generic_pattern and speakers_generic_count > 0:
                                prefix = model_def.get('speakers_generic_display_prefix', '')
                                for i in range(speakers_generic_count):
                                     gen_id = speakers_generic_pattern.format(id=i)
                                     if gen_id in voices_to_enable_this_model:
                                         display_name = f"{prefix}{i}"
                                         enabled_silero_voice_map[gen_id] = {'name': display_name, 'model_key': model_key, 'sample_rate': sample_rate, 'lang': lang_code}
                                         temp_api_voices[gen_id] = {'id': gen_id, 'name': display_name}
                            # Add specific named voices
                            for voice_id, display_name in defined_voices.items():
                                 if voice_id in voices_to_enable_this_model:
                                     enabled_silero_voice_map[voice_id] = {'name': display_name, 'model_key': model_key, 'sample_rate': sample_rate, 'lang': lang_code}
                                     temp_api_voices[voice_id] = {'id': voice_id, 'name': display_name}

                        except Exception as e:
                            logger.exception(f"ERROR loading Silero model package '{model_key}' or processing voices: {e}")
                    else:
                         logger.error(f"Could not ensure Silero package file for model '{model_key}'. Skipping load.")
        else:
             logger.warning("Silero helper files could not be prepared. Skipping all Silero models.")
    elif 'silero' in enabled_engines:
         logger.info("Silero engine enabled but definitions missing in YAML or helpers failed.")
    else:
         logger.info("Silero engine disabled in config.")


    # 6. Load Piper Voices
    if 'piper' in enabled_engines and 'piper' in voice_definitions and PIPER_AVAILABLE:
        piper_def = voice_definitions['piper']
        models_dir = os.path.abspath(piper_def.get('models_dir', 'piper_models'))
        hf_repo_id = piper_def.get('hf_repo_id')

        if not hf_repo_id:
             logger.warning("Piper 'hf_repo_id' missing in definitions. Cannot download models.")
        else:
            for lang_code, lang_def in piper_def.get('languages', {}).items():
                if lang_code in enabled_languages:
                    env_var_key = f"ENABLED_PIPER_VOICES_{lang_code.upper()}"
                    enabled_ids_for_lang = parse_list_env(app_config.get(env_var_key))
                    defined_voices_for_lang = lang_def.get('voices', {})
                    defined_ids_for_lang = set(defined_voices_for_lang.keys())

                    if env_var_key in app_config:
                        # User specified list
                        voices_to_enable_this_lang = enabled_ids_for_lang.intersection(defined_ids_for_lang)
                        if not voices_to_enable_this_lang:
                             logger.info(f"Piper language '{lang_code}' enabled but no valid voices listed in {env_var_key} match definitions. Skipping.")
                             continue
                    else:
                        # Load all defined voices for this language
                        voices_to_enable_this_lang = defined_ids_for_lang
                        logger.info(f"{env_var_key} not found in config.env, enabling all defined Piper voices for {lang_code}: {len(voices_to_enable_this_lang)} voices.")

                    if not voices_to_enable_this_lang:
                         logger.info(f"No voices to enable for Piper language '{lang_code}'. Skipping.")
                         continue

                    logger.info(f"Attempting to load enabled Piper voices for language '{lang_code}': {list(voices_to_enable_this_lang)}")
                    for voice_id in voices_to_enable_this_lang:
                         if voice_id in defined_voices_for_lang:
                             voice_def = defined_voices_for_lang[voice_id]
                             subpath = voice_def.get('subpath_no_ext')
                             display_name = voice_def.get('name', voice_id) # Use ID as fallback name

                             if subpath:
                                 files_ok, onnx_path, json_path = ensure_piper_voice(models_dir, hf_repo_id, subpath)
                                 if files_ok:
                                     try:
                                         # Specify execution providers if needed later, default is CPU
                                         # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if selected_torch_device.type == 'cuda' else ['CPUExecutionProvider']
                                         loaded_voice = PiperVoice.load(onnx_path, config_path=json_path) # Piper doesn't directly take device arg here
                                         loaded_piper_voices[voice_id] = loaded_voice
                                         enabled_piper_voice_map[voice_id] = {'name': display_name, 'lang': lang_code}
                                         temp_api_voices[voice_id] = {'id': voice_id, 'name': display_name}
                                         logger.info(f"Successfully loaded Piper voice '{voice_id}'.")
                                     except Exception as e:
                                         logger.exception(f"ERROR loading Piper voice '{voice_id}' from {onnx_path}: {e}")
                                 else:
                                      logger.error(f"Could not ensure files for Piper voice '{voice_id}'. Skipping.")
                             else:
                                  logger.error(f"Skipping Piper voice '{voice_id}': Missing 'subpath_no_ext' in definitions.")
            if loaded_piper_voices:
                 logger.info("Piper models loaded. Note: Piper inference currently runs on CPU via ONNX Runtime within this server.")

    elif 'piper' in enabled_engines and not PIPER_AVAILABLE:
        logger.warning("Piper engine enabled but library not available. Install 'piper-tts'.")
    elif 'piper' in enabled_engines:
         logger.info("Piper engine enabled but definitions missing in YAML.")
    else:
         logger.info("Piper engine disabled in config.")

    # 7. Finalize API Voice List (sorted)
    active_voices_api_list = sorted(temp_api_voices.values(), key=lambda x: x['name'])

    # 8. Set Default Voice Check
    if default_voice_id and default_voice_id not in enabled_silero_voice_map and default_voice_id not in enabled_piper_voice_map:
         logger.warning(f"Configured DEFAULT_VOICE '{default_voice_id}' is not among the enabled voices. API might error if no voice is specified in requests.")
    elif not default_voice_id:
         logger.info("No DEFAULT_VOICE specified in config.")

    logger.info(f"Model loading finished. Using device: {selected_torch_device}. Total voices loaded: {len(active_voices_api_list)}")
    logger.info(f"Enabled Silero voices: {list(enabled_silero_voice_map.keys())}")
    logger.info(f"Enabled Piper voices: {list(enabled_piper_voice_map.keys())}")


@app.get("/audio/voices")
async def list_available_voices():
    """Returns a list of available voices."""
    return {"voices": active_voices_api_list}


@app.post("/audio/speech")
async def generate_speech(request: Request):
    """Generates audio speech from text using a specified voice."""
    request_id = os.urandom(4).hex() # Unique ID for logging request flow
    try:
        body = await request.json()
        text = body.get("input")
        # Use default voice from config if specified and no voice provided in request
        requested_voice_id = body.get("voice", default_voice_id)

        if not text:
            raise HTTPException(status_code=400, detail={"error": "Input text cannot be empty"})
        if not requested_voice_id:
             # This case only happens if default_voice_id is also empty
             raise HTTPException(status_code=400, detail={"error": "Voice ID must be specified (or a DEFAULT_VOICE set in config.env)"})

        logger.info(f"[{request_id}] Request: voice='{requested_voice_id}', text='{text[:50]}...'")

        engine = None
        selected_model = None
        params = {}
        synthesis_device = torch.device('cpu') # Default device for synthesis operation if needed

        if requested_voice_id in enabled_silero_voice_map:
            engine = "Silero"
            voice_info = enabled_silero_voice_map[requested_voice_id]
            model_key = voice_info['model_key']
            if model_key in loaded_silero_models:
                 selected_model = loaded_silero_models[model_key]
                 params['speaker'] = requested_voice_id # Silero needs speaker ID
                 params['sample_rate'] = voice_info['sample_rate']
                 synthesis_device = selected_model.device # Use the device the model is already on
                 logger.debug(f"[{request_id}] Routing to Silero model '{model_key}', speaker: {requested_voice_id}, device: {synthesis_device}")
            else:
                 # This shouldn't happen if startup logic is correct, but safeguard anyway
                 logger.error(f"[{request_id}] Voice '{requested_voice_id}' enabled, but Silero model '{model_key}' not found in loaded models.")
                 raise HTTPException(status_code=503, detail={"error": f"Service unavailable: Silero model for voice '{requested_voice_id}' is not loaded."})

        elif requested_voice_id in enabled_piper_voice_map:
             engine = "Piper"
             if requested_voice_id in loaded_piper_voices:
                 selected_model = loaded_piper_voices[requested_voice_id]
                 # Piper/ONNX runs on CPU by default in this setup
                 logger.debug(f"[{request_id}] Routing to Piper voice: {requested_voice_id} (CPU)")
             else:
                  # Safeguard
                  logger.error(f"[{request_id}] Voice '{requested_voice_id}' enabled, but Piper voice not found in loaded models.")
                  raise HTTPException(status_code=503, detail={"error": f"Service unavailable: Piper voice '{requested_voice_id}' is not loaded."})
        else:
            logger.error(f"[{request_id}] Requested voice '{requested_voice_id}' is not enabled or defined.")
            raise HTTPException(status_code=400, detail={"error": f"Voice ID '{requested_voice_id}' is not available or not enabled."})

        logger.debug(f"[{request_id}] Starting {engine} TTS generation...")
        start_time = time.time()
        audio_bytes = None
        media_type = "audio/wav" # Always WAV for now

        if engine == "Silero":
            # Ensure model is on the correct device before inference
            # Note: apply_tts expects the model to be on the target device already
            if selected_model.device != synthesis_device:
                 logger.warning(f"[{request_id}] Silero model device ({selected_model.device}) differs from target synthesis device ({synthesis_device}). This shouldn't happen.")
                 # Attempting synthesis anyway, PyTorch might handle it or error out.

            audio_tensors = selected_model.apply_tts(text=text, speaker=params['speaker'], sample_rate=params['sample_rate'])
            # Move tensor to CPU *after* inference if it was on GPU, before numpy conversion
            audio_numpy = audio_tensors.cpu().numpy()

            # Normalize audio amplitude to prevent clipping
            # Use a small epsilon to avoid division by zero for silence
            norm_factor = np.max(np.abs(audio_numpy))
            if norm_factor < 1e-4: norm_factor = 1e-4 # Avoid amplifying silence too much
            audio_numpy = audio_numpy * (32767.0 / norm_factor)
            audio_numpy = np.clip(audio_numpy, -32767, 32767) # Ensure within int16 range
            audio_numpy = audio_numpy.astype(np.int16)

            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1) # Silero produces mono
                wf.setsampwidth(2) # 16-bit
                wf.setframerate(params['sample_rate'])
                wf.writeframes(audio_numpy.tobytes())
            audio_bytes = wav_buffer.getvalue()

        elif engine == "Piper":
            # Piper currently runs on CPU via its ONNX backend
            wav_buffer = io.BytesIO()
            # Piper's synthesize method writes directly to the wave file object
            with wave.open(wav_buffer, 'wb') as wf:
                 # PiperVoice object handles setting channels, sampwidth, framerate internally
                 selected_model.synthesize(text, wf)
            audio_bytes = wav_buffer.getvalue()

        generation_time = time.time() - start_time
        logger.info(f"[{request_id}] {engine} TTS finished in {generation_time:.3f}s. Output size: {len(audio_bytes)} bytes.")

        return Response(content=audio_bytes, media_type=media_type)

    except HTTPException as http_exc:
         # Log HTTP exceptions raised intentionally
         logger.warning(f"[{request_id}] HTTP Exception: {http_exc.status_code} - {http_exc.detail}")
         raise http_exc # Re-raise to let FastAPI handle the response
    except Exception as e:
        # Log unexpected errors
        logger.exception(f"[{request_id}] UNHANDLED ERROR during TTS generation: {e}")
        # Return a generic 500 error to the client
        raise HTTPException(status_code=500, detail={"error": "Internal server error during TTS generation."})


@app.get("/audio/models")
async def list_models():
    """Placeholder endpoint, returning an empty list."""
    logger.debug("Responding to /audio/models request (returning empty).")
    # Note: OpenAI TTS API doesn't use /audio/models. The general models endpoint is /v1/models.
    return {"models": []}


if __name__ == "__main__":
   import uvicorn
   # This block runs only when script is executed directly (e.g., python main.py)
   # It won't run when executed by a production server like `uvicorn main:app`
   logger.info("Starting Uvicorn server directly for debugging...")

   # Perform startup tasks manually since @app.on_event("startup") doesn't run here
   # This is imperfect for debugging as state might differ from production startup
   # Consider running `uvicorn main:app --reload` for better dev experience
   # asyncio.run(load_models_and_config()) # This would require making startup async compatible if needed

   uvicorn.run("main:app", host="0.0.0.0", port=5003, reload=True) # Use reload only for dev