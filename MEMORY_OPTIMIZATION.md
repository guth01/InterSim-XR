# Memory Optimization for InterSim-XR

## Memory Requirements

This application has different memory requirements depending on the configuration:

### Minimal Configuration (Memory-Constrained Deployments)
- **Memory Required**: ~100-200MB
- **Features**: Full interview functionality with fallback confidence scores
- **Whisper**: Disabled (uses default confidence values)

### Full Configuration (Local Development)
- **Memory Required**: ~800MB-1.2GB
- **Features**: Full interview functionality with real Whisper confidence scoring
- **Whisper**: Enabled with local model

## Deployment Options

### Option 1: Memory-Constrained Deployment (Recommended for <512MB RAM)

1. **Install minimal dependencies:**
   ```bash
   pip install -r requirements.minimal.txt
   ```

2. **Set environment variables:**
   ```bash
   export DISABLE_WHISPER=true
   ```
   
   Or copy the production environment file:
   ```bash
   cp .env.production .env
   ```

3. **Deploy:**
   ```bash
   gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT
   ```

### Option 2: Full Featured Deployment (For >1GB RAM)

1. **Install all dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables for smaller model:**
   ```bash
   export WHISPER_MODEL_SIZE=tiny
   export DISABLE_WHISPER=false
   ```

3. **Deploy:**
   ```bash
   gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT
   ```

### Option 3: Hybrid Deployment (Moderate Memory)

1. **Install base + whisper dependencies:**
   ```bash
   pip install -r requirements.minimal.txt
   pip install -r requirements.whisper.txt
   ```

2. **Use tiny model:**
   ```bash
   export WHISPER_MODEL_SIZE=tiny
   export DISABLE_WHISPER=false
   ```

## Environment Variables

| Variable | Default | Options | Description |
|----------|---------|---------|-------------|
| `DISABLE_WHISPER` | `false` | `true`/`false` | Completely disable Whisper to save memory |
| `WHISPER_MODEL_SIZE` | `tiny` | `tiny`, `base`, `small`, `medium`, `large` | Whisper model size |

## Model Memory Usage

| Model Size | Download | RAM Usage | Accuracy |
|------------|----------|-----------|----------|
| `tiny` | ~39MB | ~150MB | Good |
| `base` | ~139MB | ~400MB | Better |
| `small` | ~244MB | ~600MB | Better+ |
| `medium` | ~769MB | ~1.2GB | Best |
| `large` | ~1550MB | ~2.4GB | Best+ |

## Render.com Deployment

For Render.com's 512MB limit, use the minimal configuration:

1. Set environment variable in Render dashboard:
   - `DISABLE_WHISPER` = `true`

2. Use minimal requirements in your deploy command:
   ```bash
   pip install -r requirements.minimal.txt
   ```

## Functionality with Disabled Whisper

When Whisper is disabled:
- ✅ Full interview flow works
- ✅ Audio recording and processing
- ✅ Question generation and TTS
- ✅ Voice metrics calculation (with fallback values)
- ❌ Real confidence scoring (uses safe defaults ~0.7-0.8)

The application gracefully falls back to calculated confidence scores based on audio characteristics when Whisper is unavailable.