#!/usr/bin/env python3
"""
WavePilot Unified Pipeline - CSV to JSON Export

This script provides a complete end-to-end pipeline for processing audio preset data:
1. Load preset data from CSV
2. Optimize VAE and RBF hyperparameters
3. Train final model with best parameters
4. Export to JSON format for C++ integration

This is the recommended way to process preset data for use with the WavePilotHost plugin.

Usage:
    python pipeline.py -i presets.csv -o model.json [options]

Example:
    # Full pipeline with default settings
    python pipeline.py -i my_presets.csv -o output.json

    # Use only 50 samples for faster testing
    python pipeline.py -i my_presets.csv -o output.json -n 50

    # Custom trial counts for optimization
    python pipeline.py -i my_presets.csv -o output.json --vae-trials 200 --rbf-trials 100

    # Mask specific parameters from optimization
    python pipeline.py -i my_presets.csv -o output.json -m cutoff resonance

    # Save intermediate checkpoint
    python pipeline.py -i my_presets.csv -o output.json --save-checkpoint model.pt
"""

import argparse
import threading
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, List

# Force unbuffered output for C++ process capture
os.environ['PYTHONUNBUFFERED'] = '1'

# Detect if running in piped mode (stdout not a TTY)
# This is used to disable verbose progress bars when running from C++
IS_PIPED = not sys.stdout.isatty()

import numpy as np
import torch
from scipy.spatial.distance import pdist

from constants import (
    GLOBAL_SEED,
    N_TRIALS_VAE,
    N_TRIALS_RBF,
    FINAL_EPOCHS,
    RBF_FIXED_EPSILON_KERNELS,
    NORMALISE_LATENT_DEFAULT,
    DEFAULT_KL_BETA,
    GRAD_CLIP_DEFAULT,
    DEFAULT_BATCH_SIZE,
    RECON_ACCURACY_THRESHOLD,
)

from data import DataLoader
from logger import setup_logger
from model import VectorReducer, TrainConfig
from optimize import Optimizer, load_data
from serialization import save_model
from utils import set_global_seeds, LatentScaler

log = setup_logger("Pipeline", file=True)


def numpy_to_python(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_python(item) for item in obj]
    else:
        return obj


def extract_vae_decoder_weights(model) -> Optional[dict]:
    """
    Extract VAE decoder weights from a trained model for C++ inference.

    Args:
        model: Trained DeterministicVAE model

    Returns:
        Dict with decoder weights/biases, or None if extraction fails
    """
    if model is None:
        return None

    try:
        import torch
        state_dict = model.state_dict()

        decoder_weights = []
        decoder_biases = []

        # Extract decoder layers from nn.Sequential.
        # The model defines: self.decoder = nn.Sequential(Linear, SiLU, Linear, SiLU, ..., Linear, Sigmoid)
        # State dict keys: decoder.0.weight, decoder.2.weight, decoder.4.weight, ...
        # (odd indices are activation modules which have no weights)
        layer_indices = []
        for key in state_dict.keys():
            if key.startswith("decoder.") and ".weight" in key:
                idx = int(key.split(".")[1])
                layer_indices.append(idx)

        layer_indices = sorted(set(layer_indices))

        for idx in layer_indices:
            weight_key = f"decoder.{idx}.weight"
            bias_key = f"decoder.{idx}.bias"

            if weight_key in state_dict and bias_key in state_dict:
                weights = state_dict[weight_key].cpu().numpy()
                biases = state_dict[bias_key].cpu().numpy()
                decoder_weights.append(weights)
                decoder_biases.append(biases)

        if not decoder_weights:
            return None

        # Get architecture info from model
        input_dim = decoder_weights[-1].shape[0]  # Output layer gives input_dim
        latent_dim = decoder_weights[0].shape[1]  # First layer input is latent_dim

        # Extract hidden dims from layer shapes
        hidden_dims = []
        for i, w in enumerate(decoder_weights[:-1]):  # Exclude output layer
            hidden_dims.append(w.shape[0])

        return {
            "input_dim": int(input_dim),
            "latent_dim": int(latent_dim),
            "hidden_dims": hidden_dims,
            "decoder_weights": decoder_weights,
            "decoder_biases": decoder_biases,
        }

    except Exception as e:
        log.warning("Failed to extract VAE decoder weights: %s", e)
        return None


def export_to_json(
    vae_params: dict,
    rbf_params: dict,
    Z_std: np.ndarray,
    original_data: np.ndarray,
    scaler: Optional[LatentScaler],
    median_dist: float,
    output_path: str,
    preset_names: Optional[List[str]] = None,
    pretty: bool = True,
    vae_model=None
) -> None:
    """
    Export model data to JSON format for C++ loading.

    Args:
        vae_params: VAE hyperparameters
        rbf_params: RBF hyperparameters
        Z_std: Normalized latent positions (Nx2 or Nx3)
        original_data: Original parameter values (NxP)
        scaler: Latent space scaler (if used)
        median_dist: Median pairwise distance in latent space
        output_path: Path to output JSON file
        preset_names: Optional list of preset names
        pretty: Format with indentation if True
        vae_model: Optional trained VAE model for decoder weight export
    """
    num_presets = Z_std.shape[0]
    latent_dim = Z_std.shape[1]
    num_parameters = original_data.shape[1]

    scaler_data = None
    if scaler is not None:
        scaler_data = {
            "mu": scaler.mu,
            "sigma": scaler.sigma,
            "eps": scaler.eps
        }

    # Extract VAE decoder weights if model is provided
    vae_section = None
    format_version = "1.0"

    if vae_model is not None:
        vae_section = extract_vae_decoder_weights(vae_model)
        if vae_section is not None:
            format_version = "2.0"
            log.info("VAE decoder: %d layers extracted", len(vae_section["decoder_weights"]))

    export_data = {
        "format_version": format_version,
        "source": "WavePilot Pipeline",
        "metadata": {
            "num_presets": num_presets,
            "latent_dim": latent_dim,
            "num_parameters": num_parameters,
            "vae_latent_dim": vae_params.get("latent_dim", latent_dim),
        },
        "rbf_params": {
            "kernel": rbf_params.get("kernel", "thin_plate_spline"),
            "smoothing": rbf_params.get("smoothing", 0.0),
            "epsilon": rbf_params.get("epsilon", None),
            "epsilon_scale": rbf_params.get("epsilon_scale", None),
            "degree": rbf_params.get("degree", None),
            "median_dist": median_dist,
        },
        "scaler": scaler_data,
        "preset_names": preset_names if preset_names else [],
        "latent_positions": Z_std,
        "parameter_values": original_data,
    }

    # Add VAE section if available (format 2.0)
    if vae_section is not None:
        export_data["vae"] = vae_section

    # Convert numpy arrays to Python lists
    export_data = numpy_to_python(export_data)

    # Write JSON
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        if pretty:
            json.dump(export_data, f, indent=2)
        else:
            json.dump(export_data, f)

    file_size_kb = output_file.stat().st_size / 1024
    log.info("Exported JSON (v%s) to: %s (%.1f KB)", format_version, output_file, file_size_kb)


class ProgressWriter:
    """The pipeline's progress for the host, as a small JSON file.

    The host's worker cannot read our stdout while we run (its read
    blocks), so it polls this file instead. One object: phase, a fraction
    of the whole run (or -1 while a phase has no measurable progress) and a
    detail such as "trial 12/500". Rewritten atomically so a reader never
    sees half a file. A None path disables it.
    """

    def __init__(self, path: Optional[str]):
        self.path = path
        self._lock = threading.Lock()

    def write(self, phase: str, progress: float, detail: str = "") -> None:
        if not self.path:
            return
        payload = {"phase": phase, "progress": progress, "detail": detail}
        with self._lock:
            tmp = self.path + ".tmp"
            try:
                with open(tmp, "w") as f:
                    json.dump(payload, f)
                os.replace(tmp, self.path)
            except OSError as e:  # progress is advisory: never fail the run on it
                log.debug("progress file not written: %s", e)

    def span(self, phase: str, start: float, end: float):
        """A per-trial callback that maps k/N onto [start, end] of the run."""
        def report(done: int, total: int) -> None:
            frac = start + (end - start) * (done / total if total else 1.0)
            self.write(phase, frac, f"trial {done}/{total}")
        return report


def run_pipeline(
    input_csv: str,
    output_json: str,
    num_entries: Optional[int] = None,
    mask_columns: Optional[List[str]] = None,
    vae_trials: int = N_TRIALS_VAE,
    rbf_trials: int = N_TRIALS_RBF,
    seed: int = GLOBAL_SEED,
    save_checkpoint: Optional[str] = None,
    compact_json: bool = False,
    quiet: bool = False,
    progress_file: Optional[str] = None
) -> None:
    """
    Execute the complete pipeline from CSV to JSON.

    Args:
        input_csv: Path to input CSV file with preset data
        output_json: Path to output JSON file
        num_entries: Number of random samples to use (None = use all)
        mask_columns: List of column names to exclude
        vae_trials: Number of VAE optimization trials
        rbf_trials: Number of RBF optimization trials
        seed: Random seed for reproducibility
        save_checkpoint: Optional path to save .pt checkpoint
        compact_json: Use compact JSON format (no indentation)
        quiet: Suppress progress bars (auto-enabled when piped)
    """
    pipeline_start = time.time()
    set_global_seeds(seed)

    # Determine if we should show progress bars
    # Disable if quiet mode or if output is piped (not a TTY)
    show_progress = not (quiet or IS_PIPED)

    log.info("=" * 80)
    log.info("WavePilot Pipeline Started")
    log.info("=" * 80)
    log.info("Input CSV: %s", input_csv)
    log.info("Output JSON: %s", output_json)
    log.info("Random seed: %d", seed)
    log.info("VAE trials: %d | RBF trials: %d", vae_trials, rbf_trials)
    if num_entries:
        log.info("Using %d random samples from dataset", num_entries)
    if mask_columns:
        log.info("Masking columns: %s", mask_columns)
    log.info("-" * 80)

    progress = ProgressWriter(progress_file)

    # ========== PHASE 1: Load Data ==========
    log.info("📂 PHASE 1: Loading dataset...")
    phase_start = time.time()
    progress.write("Loading data", 0.0)

    # Load data using DataLoader to preserve preset names
    data_loader = DataLoader(input_csv, mask_columns)
    df = data_loader.load_presets()

    # Extract preset names
    preset_names = data_loader.get_preset_names(n_items=df.shape[0])
    log.info("Loaded dataset: %d samples × %d features", df.shape[0], df.shape[1])
    log.info("Extracted %d preset names", len(preset_names))

    # Handle random sampling if requested
    if num_entries and num_entries < df.shape[0]:
        selected_idx = np.random.choice(df.shape[0], size=num_entries, replace=False)
        df = df[selected_idx]
        preset_names = [preset_names[i] for i in selected_idx]
        log.info("Randomly selected %d entries from the dataset", num_entries)

    log.info("✅ Phase 1 completed in %.2f seconds", time.time() - phase_start)
    log.info("-" * 80)

    # ========== PHASE 2: Optimize Hyperparameters ==========
    log.info("🔍 PHASE 2: Optimizing hyperparameters...")
    phase_start = time.time()

    optimizer = Optimizer(df)

    # VAE optimization (the long part: 2 % -> 55 % of the run)
    log.info("Optimizing VAE hyperparameters (%d trials)...", vae_trials)
    progress.write("VAE search", 0.02, f"trial 0/{vae_trials}")
    optimizer.progress_cb = progress.span("VAE search", 0.02, 0.55)
    best_vae_params = optimizer.optimize_vae(n_trials=vae_trials, show_progress=show_progress)
    log.info("Best VAE params: %s", best_vae_params)
    progress.write("Training VAE", -1.0)  # one fit, no measurable progress

    # Train final VAE with best params
    log.info("Training final VAE with best parameters...")
    reducer = VectorReducer(df, latent_dim=best_vae_params["latent_dim"])

    cfg = TrainConfig(
        epochs=FINAL_EPOCHS,
        lr=best_vae_params["learning_rate"],
        kl_beta=best_vae_params.get("kl_beta", DEFAULT_KL_BETA),
        deterministic=True,
        hidden_dims=None,
        grad_clip=best_vae_params.get("grad_clip", GRAD_CLIP_DEFAULT),
        batch_size=best_vae_params.get("batch_size", DEFAULT_BATCH_SIZE),
    )
    setattr(cfg, "width_scale", best_vae_params["width_scale"])
    setattr(cfg, "depth", best_vae_params["depth"])
    setattr(cfg, "round_to", best_vae_params["round_to"])

    reducer.fit(cfg, show_progress=show_progress)

    # Compute metrics
    accuracy = reducer.reconstruction_accuracy(threshold=RECON_ACCURACY_THRESHOLD)
    latent_points = reducer.transform()

    log.info("VAE reconstruction accuracy: %.2f%%", accuracy)
    log.info("VAE reconstruction MSE: %.6f", optimizer.study_vae.best_value)

    # RBF optimization (65 % -> 95 %)
    log.info("Optimizing RBF hyperparameters (%d trials)...", rbf_trials)
    progress.write("RBF search", 0.65, f"trial 0/{rbf_trials}")
    optimizer.progress_cb = progress.span("RBF search", 0.65, 0.95)
    best_rbf_params = optimizer.optimize_rbf(
        original_data=df,
        reduced_data=latent_points,
        n_trials=rbf_trials,
        normalise_latent=NORMALISE_LATENT_DEFAULT,
        show_progress=show_progress
    )
    log.info("Best RBF params: %s", best_rbf_params)
    log.info("RBF validation distance: %.6f", optimizer.study_rbf.best_value)

    log.info("✅ Phase 2 completed in %.2f seconds", time.time() - phase_start)
    log.info("-" * 80)
    progress.write("Exporting", 0.95)

    # ========== PHASE 3: Prepare Export Data ==========
    log.info("📦 PHASE 3: Preparing export data...")
    phase_start = time.time()

    # Get normalized latent space and scaler from optimizer
    scaler = optimizer.rbf_latent_scaler_
    Z_std = latent_points

    if scaler is not None:
        Z_std = scaler.transform(latent_points)
        log.info("Applied latent space normalization")

    median_dist = optimizer.rbf_median_dist_
    log.info("Latent space median distance: %.6f", median_dist)

    # Compute final epsilon if needed
    kernel = best_rbf_params["kernel"]
    epsilon = best_rbf_params.get("epsilon")
    if epsilon is None:
        epsilon_scale = best_rbf_params["epsilon_scale"]
        if kernel in RBF_FIXED_EPSILON_KERNELS:
            epsilon = 1.0
        else:
            epsilon = max(1e-12, float(epsilon_scale) * median_dist)
        best_rbf_params["epsilon"] = epsilon
        log.info("Computed epsilon: %.6e (scale: %.2f)", epsilon, epsilon_scale)

    log.info("✅ Phase 3 completed in %.2f seconds", time.time() - phase_start)
    log.info("-" * 80)

    # ========== PHASE 4: Save Checkpoint (Optional) ==========
    if save_checkpoint:
        log.info("💾 PHASE 4: Saving checkpoint...")
        phase_start = time.time()

        save_model(
            reducer=reducer,
            vae_params=best_vae_params,
            rbf_params=best_rbf_params,
            filepath=save_checkpoint,
            latent_scaler=scaler,
            rbf_interpolator=None,
            Z_std=Z_std,
            original_data=df,
            median_dist=median_dist
        )

        log.info("Saved checkpoint to: %s", save_checkpoint)
        log.info("✅ Phase 4 completed in %.2f seconds", time.time() - phase_start)
        log.info("-" * 80)

    # ========== PHASE 5: Export to JSON ==========
    log.info("📤 PHASE 5: Exporting to JSON...")
    phase_start = time.time()

    export_to_json(
        vae_params=best_vae_params,
        rbf_params=best_rbf_params,
        Z_std=Z_std,
        original_data=df,
        scaler=scaler,
        median_dist=median_dist,
        output_path=output_json,
        preset_names=preset_names,
        pretty=not compact_json,
        vae_model=reducer.model  # Pass trained VAE model for decoder weight export
    )

    progress.write("Done", 1.0)
    log.info("✅ Phase 5 completed in %.2f seconds", time.time() - phase_start)
    log.info("=" * 80)

    total_time = time.time() - pipeline_start
    log.info("🎉 Pipeline completed successfully in %.2f seconds (%.1f minutes)",
             total_time, total_time / 60.0)
    log.info("=" * 80)


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="wavepilot-pipeline",
        description="WavePilot Unified Pipeline - Complete CSV to JSON processing",
        epilog="""
Examples:
  # Basic usage with default settings
  python pipeline.py -i presets.csv -o model.json

  # Use 50 samples for faster testing
  python pipeline.py -i presets.csv -o model.json -n 50

  # Custom optimization trials
  python pipeline.py -i presets.csv -o model.json --vae-trials 200 --rbf-trials 100

  # Save intermediate checkpoint
  python pipeline.py -i presets.csv -o model.json --save-checkpoint model.pt

  # Mask specific parameters
  python pipeline.py -i presets.csv -o model.json -m cutoff resonance volume

  # Compact JSON output
  python pipeline.py -i presets.csv -o model.json --compact

For PyInstaller executable:
  pyinstaller --onefile --name wavepilot pipeline.py
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "-i", "--input",
        dest="input_csv",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to input CSV file with preset data (required)"
    )

    parser.add_argument(
        "-o", "--output",
        dest="output_json",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to output JSON file (required)"
    )

    parser.add_argument(
        "-n", "--num-entries",
        dest="num_entries",
        type=int,
        default=None,
        metavar="N",
        help="Number of random samples to use from dataset (default: use all)"
    )

    parser.add_argument(
        "-m", "--mask-columns",
        dest="mask_columns",
        nargs="*",
        default=None,
        metavar="COL",
        help="Space-separated list of column names to exclude from processing"
    )

    parser.add_argument(
        "--vae-trials",
        dest="vae_trials",
        type=int,
        default=N_TRIALS_VAE,
        metavar="N",
        help=f"Number of VAE optimization trials (default: {N_TRIALS_VAE})"
    )

    parser.add_argument(
        "--rbf-trials",
        dest="rbf_trials",
        type=int,
        default=N_TRIALS_RBF,
        metavar="N",
        help=f"Number of RBF optimization trials (default: {N_TRIALS_RBF})"
    )

    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=GLOBAL_SEED,
        metavar="N",
        help=f"Random seed for reproducibility (default: {GLOBAL_SEED})"
    )

    parser.add_argument(
        "--save-checkpoint",
        dest="save_checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional path to save intermediate .pt checkpoint file"
    )

    parser.add_argument(
        "--compact",
        dest="compact_json",
        action="store_true",
        help="Use compact JSON format (no indentation, smaller file size)"
    )

    parser.add_argument(
        "--progress-file",
        dest="progress_file",
        metavar="PATH",
        default=None,
        help="Write the pipeline's progress (phase, fraction, trial k/N) to this JSON file for the host"
    )
    parser.add_argument(
        "--quiet", "-q",
        dest="quiet",
        action="store_true",
        help="Suppress progress bars (auto-enabled when output is piped)"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    try:
        run_pipeline(
            input_csv=args.input_csv,
            output_json=args.output_json,
            num_entries=args.num_entries,
            mask_columns=args.mask_columns,
            vae_trials=args.vae_trials,
            rbf_trials=args.rbf_trials,
            seed=args.seed,
            save_checkpoint=args.save_checkpoint,
            compact_json=args.compact_json,
            quiet=args.quiet,
            progress_file=args.progress_file
        )
    except FileNotFoundError as e:
        log.error("❌ File not found: %s", e)
        sys.exit(1)
    except ValueError as e:
        log.error("❌ Invalid data: %s", e)
        sys.exit(1)
    except Exception as e:
        log.error("❌ Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Required for PyInstaller frozen executables with multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()

    main()
