# 🗣️ Local OpenAI-Compatible TTS Server (Silero + Piper)

Run your own fast, offline Text-to-Speech server that acts like OpenAI's API. Uses Silero and Piper models. 

Perfect for **Open WebUI** or other projects needing local TTS!

## ✨ Features

*   OpenAI API Compatible (`/audio/speech`, `/audio/voices`)
*   Supports Silero & Piper TTS engines 🚀
*   Works Offline (after initial model download) 🚫☁️
*   CPU-Friendly & Efficient 💪
*   Auto-Downloads Models 💾
*   Simple `config.env` File ⚙️

## 🚀 Quick Start (Get Running!)

**1. Prerequisites:**

*   🐍 Python 3.9+
*   🐙 `git`

**2. Setup:**

```bash
# Clone the project
git clone https://github.com/ginto-sakata/local-openai-tts-server.git
cd local-openai-tts-server

# Create a virtual environment (keeps things tidy!)
python -m venv .venv

# Activate the environment
# Linux/macOS:
source .venv/bin/activate
# Windows CMD/PowerShell (use one):
# .venv\Scripts\activate.bat
# .venv\Scripts\Activate.ps1

# Install required packages (uses requirements.txt)
pip install --upgrade pip
pip install -r requirements.txt

# Create your configuration file from the example
cp .env.example config.env
```

**3. Configure (Optional First Run):**

*   Edit `config.env` with a text editor if you want to change languages or voices (defaults are usually fine to start). See **⚙️ Configuration** below.

**4. Run the Server! 🎉**

```bash
# Make sure your virtual environment is active! (See step 2)
uvicorn main:app --host 0.0.0.0 --port 5003
```

*   ⏳ **First time?** It will download models specified in `config.env`. This might take a few minutes!
*   ✅ Server is running when you see lines like `Application startup complete.`
*   ➡️ Access the API at `http://localhost:5003` (or your server's IP).

## ⚙️ Configuration (`config.env`)

Edit `config.env` to control:

*   `LANGUAGES=en,ru,...`: Languages to enable (e.g., `en,de,fr`).
*   `ENABLED_ENGINES=silero,piper`: Which TTS engines to use.
*   `DEFAULT_VOICE=id`: Default voice if none requested (e.g., `lessac-high`). Leave blank for no default.
*   `SILERO_QUALITY=high|medium|low`: Quality for Silero (default `high`).
*   `ENABLED_SILERO_VOICES_<LANG>=id1,id2`: Specific Silero voices for a language (e.g., `ENABLED_SILERO_VOICES_RU=aidar,baya`). *If missing, ALL voices for that language load (uses more RAM!).*
*   `ENABLED_PIPER_VOICES_<LANG>=id1,id2`: Specific Piper voices (e.g., `ENABLED_PIPER_VOICES_EN=lessac-high,amy-medium`). *If missing, ALL voices load.*
*   `DEVICE=auto|cpu|cuda`: (Optional) Device for Silero (`auto` is default). See **✨ Optional Steps**.
*   `TORCH_CPU_THREADS=0|N`: (Optional) CPU threads for PyTorch (`0` is default = all cores).

## ▶️ API Usage

*   **List Voices:** `GET /audio/voices`
    ```bash
    curl http://localhost:5003/audio/voices
    ```
*   **Generate Speech:** `POST /audio/speech` (Body: `{"input": "Text", "voice": "voice_id"}`)
    ```bash
    curl -X POST http://localhost:5003/audio/speech \
      -H "Content-Type: application/json" \
      -d '{"input": "Hello from my local server!", "voice": "lessac-high"}' \
      -o output.wav
    ```

## ✨ Optional Steps & Integrations

*   **⚡ CUDA / GPU Support:**
    1.  Ensure you have NVIDIA drivers & CUDA toolkit installed.
    2.  Upgrade PyTorch: Get command from <https://pytorch.org/get-started/locally/> (Select Pip, your OS, CUDA). Run the command (e.g., `pip install --upgrade torch ... --index-url ...`).
    3.  Set `DEVICE=cuda` in `config.env`.

*   **🖥️ Run as Service (Systemd):** Keep the server running in the background.
    1.  Create `/etc/systemd/system/local-tts.service` (use `sudo nano ...`):
        ```ini
        [Unit]
        Description=Local TTS Server
        After=network.target
        [Service]
        User=your_username # CHANGE THIS
        WorkingDirectory=/path/to/local-openai-tts-server # CHANGE THIS
        ExecStart=/path/to/local-openai-tts-server/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 5003 # CHANGE THIS
        Restart=on-failure
        [Install]
        WantedBy=multi-user.target
        ```
    2.  `sudo systemctl daemon-reload`
    3.  `sudo systemctl enable --now local-tts.service`
    4.  Check logs: `sudo journalctl -u local-tts.service -f`

*   **🐸 Open WebUI:**
    1.  Go to Open WebUI Settings -> Connections -> Audio.
    2.  Enable OpenAI API TTS.
    3.  API Key: `any_text`
    4.  Base URL: `http://<your_server_ip>:5003` (Use your server's actual IP).
    5.  Save. Select voices in chat!

*   **🔄 Reverse Proxy (Caddy Example):** Use HTTPS / standard ports.
    ```caddyfile
    # Add to your Caddyfile (/etc/caddy/Caddyfile)
    tts.yourdomain.com { # CHANGE THIS
        reverse_proxy localhost:5003
    }
    ```
    Reload Caddy: `sudo systemctl reload caddy`. Use `https://tts.yourdomain.com` as Base URL.

*   **🧊 Conda Users:** You can use Conda instead of `venv`. Create env (`conda create -n tts-server python=3.11 -y`), activate (`conda activate tts-server`), then `pip install -r requirements.txt`. Use the Conda Systemd template if needed.

## 📚 Available Voices & Languages

See `voices_definitions.yaml` for all possible models. Includes Silero (RU, EN) and Piper (~25+ languages like DE, FR, ES, IT, ZH, etc.). Enable desired languages/voices in `config.env`.

## ⚖️ License

*   Server Code: MIT License (see `LICENSE` file).
*   Models: Silero & Piper models have their **own licenses** (check their repos!). Silero may have non-commercial restrictions. **Your responsibility to comply.**

---
