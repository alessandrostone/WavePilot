#!/usr/bin/env python3
"""
Export WavePilot .pt model to JSON format for C++ loading.

This script extracts the essential RBF interpolation data and VAE decoder
weights from a WavePilot checkpoint (.pt file) and exports them to a JSON
format that can be loaded by the C++ RBFInterpolator and VAEInference classes.

Usage:
    poetry run python export_to_json.py -i model.pt -o model.json

The exported JSON contains:
    - latent_positions: Normalized 3D coordinates for each preset (Nx3 matrix)
    - parameter_values: Normalized parameter values for each preset (NxP matrix)
    - rbf_params: RBF kernel type, smoothing, epsilon, degree
    - scaler: Mean/std for latent space normalization
    - metadata: Version, number of presets, number of parameters
    - vae (optional): VAE decoder weights for smoother interpolation (format 2.0+)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from serialization import load_model
from logger import setup_logger


log = setup_logger("Export to JSON")


def numpy_to_python(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
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


def extract_vae_decoder_weights(state_dict: Dict[str, Any], model_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract VAE decoder weights from PyTorch state_dict for C++ inference.

    The DeterministicVAE decoder architecture is:
        z -> decoder_backbone (Linear+SiLU layers) -> fc_out (Linear+Sigmoid) -> params

    Args:
        state_dict: PyTorch model state_dict
        model_meta: Model metadata containing architecture info

    Returns:
        Dict containing decoder_weights and decoder_biases arrays for each layer
    """
    # Collect decoder layers in order.
    # The model defines: self.decoder = nn.Sequential(Linear, SiLU, Linear, SiLU, ..., Linear, Sigmoid)
    # State dict keys: decoder.0.weight, decoder.2.weight, decoder.4.weight, ...
    # (odd indices are activation modules which have no weights)

    decoder_weights = []
    decoder_biases = []

    # Extract all Linear layers from the decoder Sequential
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

    return {
        "input_dim": int(model_meta.get("input_dim", decoder_weights[-1].shape[0])),
        "latent_dim": int(model_meta.get("latent_dim", decoder_weights[0].shape[1])),
        "hidden_dims": list(model_meta.get("hidden_dims", [])),
        "decoder_weights": decoder_weights,
        "decoder_biases": decoder_biases,
    }


def export_model_to_json(
    pt_filepath: str,
    json_filepath: str,
    pretty: bool = True
) -> None:
    """
    Load a WavePilot .pt checkpoint and export RBF interpolation data to JSON.

    Args:
        pt_filepath: Path to input .pt file
        json_filepath: Path to output .json file
        pretty: If True, format JSON with indentation for readability
    """
    log.info("Loading model from: %s", pt_filepath)

    # Load the checkpoint directly (bypassing model instantiation which may fail on dim validation)
    ckpt = torch.load(pt_filepath, map_location=torch.device("cpu"), weights_only=False)

    if 'params' not in ckpt or 'rbf_bundle' not in ckpt or 'model_meta' not in ckpt:
        raise ValueError("Invalid checkpoint format - missing required keys")

    vae_params = ckpt['params'].get('vae', {})
    rbf_params = ckpt['params'].get('rbf', {})
    rbf_bundle = ckpt['rbf_bundle']
    model_meta = ckpt['model_meta']

    # Extract latent scaler (for coordinate normalization)
    scaler_data = rbf_bundle.get("scaler", None)
    scaler = None
    if scaler_data is not None:
        scaler = {
            "mu": scaler_data.get("mu"),
            "sigma": scaler_data.get("sigma"),
            "eps": scaler_data.get("eps", 1e-12)
        }

    # Extract latent positions (normalized coordinates)
    Z_std = rbf_bundle.get("Z_std", None)
    if Z_std is None:
        raise ValueError("Checkpoint does not contain Z_std (normalized latent positions)")

    # Extract original parameter values
    original_data = rbf_bundle.get("original_data", None)
    if original_data is None:
        raise ValueError("Checkpoint does not contain original_data (parameter values)")

    num_presets = Z_std.shape[0]
    latent_dim = Z_std.shape[1]
    num_parameters = original_data.shape[1]

    log.info("Model contains:")
    log.info("  - %d presets", num_presets)
    log.info("  - %d latent dimensions", latent_dim)
    log.info("  - %d parameters per preset", num_parameters)

    # Extract VAE decoder weights if available
    state_dict = ckpt.get('state_dict', None)
    vae_section = None
    format_version = "1.0"

    if state_dict is not None:
        vae_section = extract_vae_decoder_weights(state_dict, model_meta)
        if vae_section is not None:
            format_version = "2.0"  # Bump version when VAE is included
            log.info("  - VAE decoder: %d layers", len(vae_section["decoder_weights"]))
            log.info("    Architecture: %d -> %s -> %d",
                     vae_section["latent_dim"],
                     " -> ".join(str(d) for d in vae_section["hidden_dims"]),
                     vae_section["input_dim"])
        else:
            log.warning("  - VAE decoder weights not found in checkpoint")

    # Build JSON structure
    export_data = {
        "format_version": format_version,
        "source": "WavePilot",
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
            "median_dist": rbf_bundle.get("median_dist", None),
        },
        "scaler": scaler,
        "latent_positions": Z_std,  # Nx3 (or Nx2 or Nx4) normalized coordinates
        "parameter_values": original_data,  # NxP normalized parameter values [0,1]
    }

    # Add VAE section if available (format 2.0+)
    if vae_section is not None:
        export_data["vae"] = vae_section

    # Convert numpy arrays to Python lists
    export_data = numpy_to_python(export_data)

    # Write JSON
    output_path = Path(json_filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        if pretty:
            json.dump(export_data, f, indent=2)
        else:
            json.dump(export_data, f)

    file_size_kb = output_path.stat().st_size / 1024
    log.info("Exported to: %s (%.1f KB)", output_path, file_size_kb)
    log.info("✅ Export successful!")


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="export_to_json",
        description="Export WavePilot .pt model to JSON for C++ loading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export model to JSON with pretty formatting
  poetry run python export_to_json.py -i model.pt -o model.json

  # Export to compact JSON (smaller file size)
  poetry run python export_to_json.py -i model.pt -o model.json --compact

  # Export with automatic output naming
  poetry run python export_to_json.py -i /path/to/Freak_2.pt
  # Creates: /path/to/Freak_2.json
        """
    )

    parser.add_argument(
        "-i", "--input",
        dest="input_path",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to input .pt model file (required)"
    )

    parser.add_argument(
        "-o", "--output",
        dest="output_path",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to output .json file (default: same as input with .json extension)"
    )

    parser.add_argument(
        "--compact",
        dest="compact",
        action="store_true",
        help="Compact JSON output (no indentation, smaller file size)"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Determine output path
    output_path = args.output_path
    if output_path is None:
        output_path = Path(args.input_path).with_suffix('.json')

    try:
        export_model_to_json(
            pt_filepath=args.input_path,
            json_filepath=output_path,
            pretty=not args.compact
        )
    except FileNotFoundError as e:
        log.error("File not found: %s", e)
        exit(1)
    except ValueError as e:
        log.error("Invalid checkpoint: %s", e)
        exit(1)
    except Exception as e:
        log.error("Export failed: %s", e, exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
