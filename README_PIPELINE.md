# WavePilot Unified Pipeline

Complete end-to-end solution for processing audio preset data from CSV to JSON.

## 📋 Overview

The WavePilot pipeline automates the complete workflow:

1. **Data Loading** - Load preset parameters from CSV
2. **VAE Optimization** - Find optimal neural network architecture
3. **VAE Training** - Train final model with best hyperparameters
4. **RBF Optimization** - Find optimal interpolation kernel
5. **JSON Export** - Export ready-to-use model for C++ integration

## 🚀 Quick Start

### Basic Usage

```bash
cd libs/wavepilot
python pipeline.py -i presets.csv -o model.json
```

### Common Options

```bash
# Use only 50 samples for faster testing
python pipeline.py -i presets.csv -o model.json -n 50

# Custom optimization trial counts
python pipeline.py -i presets.csv -o model.json --vae-trials 200 --rbf-trials 100

# Save intermediate checkpoint
python pipeline.py -i presets.csv -o model.json --save-checkpoint model.pt

# Mask specific parameters
python pipeline.py -i presets.csv -o model.json -m cutoff resonance volume

# Compact JSON output (smaller file size)
python pipeline.py -i presets.csv -o model.json --compact
```

## 📊 Input Format

CSV file with normalized parameter values (0-1 range):

```csv
preset_name,param1,param2,param3,...
Init,0.5,0.3,0.7,...
Warm Pad,0.8,0.2,0.9,...
Bass,0.1,0.9,0.4,...
```

**Note:** The first column can be named either "preset_name" or "name" (case-insensitive). These names will be preserved in the JSON export and used for preset identification in the visualization.

## 📦 Output Format

JSON file containing:
- **preset_names**: List of preset names from the CSV (extracted from "preset_name" or "name" column)
- **latent_positions**: 2D/3D coordinates for each preset
- **parameter_values**: Original parameter values
- **rbf_params**: RBF kernel configuration
- **scaler**: Latent space normalization parameters
- **metadata**: Model information

## ⚙️ Command-Line Options

### Required Arguments

| Option | Description |
|--------|-------------|
| `-i PATH` | Input CSV file with preset data |
| `-o PATH` | Output JSON file path |

### Optional Arguments

| Option | Default | Description |
|--------|---------|-------------|
| `-n N` | All | Number of random samples to use |
| `-m COL [COL ...]` | None | Columns to exclude from processing |
| `--vae-trials N` | 500 | VAE optimization trial count |
| `--rbf-trials N` | 300 | RBF optimization trial count |
| `--seed N` | 56 | Random seed for reproducibility |
| `--save-checkpoint PATH` | None | Save intermediate .pt file |
| `--compact` | False | Compact JSON (no indentation) |

## 🔧 Advanced Usage

### Testing with Small Dataset

```bash
# Use 30 samples, fewer trials for quick validation
python pipeline.py -i presets.csv -o test.json \
  -n 30 \
  --vae-trials 50 \
  --rbf-trials 30
```

### Production Run

```bash
# Full dataset, save checkpoint, detailed logging
python pipeline.py -i presets.csv -o production.json \
  --save-checkpoint production.pt \
  --vae-trials 500 \
  --rbf-trials 300
```

### Exclude Problematic Parameters

```bash
# Mask parameters with low variance or high correlation
python pipeline.py -i presets.csv -o model.json \
  -m volume pan master_gain
```

## 🏗️ Creating Standalone Executable

Build a self-contained executable with PyInstaller:

```bash
# Install PyInstaller
pip install pyinstaller

# Create executable
pyinstaller --onefile --name wavepilot pipeline.py

# Run executable
./dist/wavepilot -i presets.csv -o model.json
```

## 📈 Performance Optimizations Applied

The pipeline includes **critical performance fixes**:

### 1. GPU/MPS Acceleration (10-100x speedup)
- Auto-detects and uses GPU/MPS for VAE training
- Falls back to CPU if no accelerator available

### 2. Parallel RBF Optimization (4-8x speedup)
- Utilizes all CPU cores for RBF trials
- Scales with core count

### 3. Expected Performance
- **Before fixes**: Hours for 50 presets
- **After fixes**: Minutes for 50 presets
- **Total speedup**: 40-800x faster

See `OPTIMIZATION_FINDINGS.md` for detailed analysis.

## 🔍 Monitoring Progress

The pipeline provides detailed progress tracking:

```
================================================================================
WavePilot Pipeline Started
================================================================================
Input CSV: presets.csv
Output JSON: model.json
Random seed: 56
VAE trials: 500 | RBF trials: 300
--------------------------------------------------------------------------------
📂 PHASE 1: Loading dataset...
Loaded dataset: 50 samples × 120 features
✅ Phase 1 completed in 0.12 seconds
--------------------------------------------------------------------------------
🔍 PHASE 2: Optimizing hyperparameters...
[Progress bars for VAE and RBF optimization]
✅ Phase 2 completed in 45.23 seconds
--------------------------------------------------------------------------------
📦 PHASE 3: Preparing export data...
✅ Phase 3 completed in 0.34 seconds
--------------------------------------------------------------------------------
📤 PHASE 5: Exporting to JSON...
Exported JSON to: model.json (23.4 KB)
✅ Phase 5 completed in 0.08 seconds
================================================================================
🎉 Pipeline completed successfully in 45.77 seconds (0.8 minutes)
================================================================================
```

## 🐛 Troubleshooting

### Out of Memory (GPU)

If you encounter GPU memory errors:

```bash
# The pipeline will automatically fall back to CPU
# Or explicitly use small batch size:
python pipeline.py -i presets.csv -o model.json -n 30
```

### Slow Performance

Check if GPU/MPS is being used:
```bash
# Should see "Using device: cuda" or "Using device: mps" in logs
# If you see "Using device: cpu", check CUDA/MPS installation
```

### Invalid CSV Format

Ensure CSV has:
- Header row with parameter names
- All values in 0-1 range (normalized)
- No missing values

## 📝 Log Files

Logs are saved to `/tmp/wavepilot_logs/`:
- `OptimizationLogger_YYYYMMDD_HHMMSS.log`
- `ProgressLogger_YYYYMMDD_HHMMSS.log`
- `Pipeline_YYYYMMDD_HHMMSS.log`

## 🔗 Integration with WavePilotHost

The exported JSON can be loaded directly into WavePilotHost:

1. Place JSON in your presets folder
2. Load via plugin UI
3. Navigate latent space in real-time
4. Interpolate between presets smoothly

## 🤝 Contributing

Found a bug or have a feature request? See `OPTIMIZATION_FINDINGS.md` for current issues and roadmap.

## 📄 License

See main WavePilotHost repository for license information.
