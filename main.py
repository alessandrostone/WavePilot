"""
WavePilot Main Entry Point - Unified interface for optimization and training pipelines
"""
import argparse
import asyncio
from optimize import main as optimize_main
from train import main as train_main
from logger import setup_logger


log = setup_logger("Main")

def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="wavepilot",
        description="WavePilot - Audio preset dimensionality reduction using VAE and RBF interpolation",
        epilog="""
Examples:
  # Optimize hyperparameters on a preset dataset
  python main.py optimize -f presets.csv

  # Optimize with 100 random samples and exclude specific columns
  python main.py optimize -f presets.csv -n 100 -m cutoff resonance

  # Train a model from scratch
  python main.py train -f presets.csv

  # Train using optimized hyperparameters from a previous session
  python main.py train -f presets.csv -o optimizer_log.txt

  # Train and save the model
  python main.py train -f presets.csv -s model.pth

  # Load a pretrained model and continue training
  python main.py train -f presets.csv -p model.pth -s model_v2.pth

For detailed help on each mode:
  python main.py optimize --help
  python main.py train --help
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(
        dest="mode",
        required=True,
        help="Operation mode"
    )

    # ========== OPTIMIZE SUBCOMMAND ==========
    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Run hyperparameter optimization for VAE and RBF",
        description="Optimize VAE and RBF hyperparameters using Optuna",
        epilog="""
This mode performs:
  1. VAE hyperparameter search (learning rate, KL-beta, latent dimensions, architecture)
  2. Final VAE training with best parameters
  3. RBF hyperparameter search (kernel type, smoothing, epsilon, degree)
  4. Outputs best parameters and reconstruction accuracy metrics
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    optimize_parser.add_argument(
        "-f", "--filepath",
        dest="filepath",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to preset dataset CSV file (required)"
    )

    optimize_parser.add_argument(
        "-n", "--num-entries",
        dest="num_entries",
        type=int,
        default=None,
        metavar="N",
        help="Number of random entries to use from dataset (default: use all)"
    )

    optimize_parser.add_argument(
        "-m", "--mask-columns",
        dest="mask_columns",
        type=str,
        nargs="+",
        default=None,
        metavar="COL",
        help="Space-separated list of column names to exclude from optimization"
    )

    # ========== TRAIN SUBCOMMAND ==========
    train_parser = subparsers.add_parser(
        "train",
        help="Train or continue training a VAE model",
        description="Train a VAE model for preset dimensionality reduction",
        epilog="""
Training modes:
  1. From scratch: Only specify dataset (-f)
  2. From optimized hyperparameters: Use -o with optimizer log file
  3. Continue training: Use -p to load pretrained model
  4. Save trained model: Use -s to specify output path

The trained model can be used for preset interpolation and exploration.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    train_parser.add_argument(
        "-f", "--filepath",
        dest="filepath",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to preset dataset CSV file (required)"
    )

    train_parser.add_argument(
        "-p", "--pretrained-model",
        dest="pretrained_model",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to pretrained model file to continue training from"
    )

    train_parser.add_argument(
        "-o", "--optimizer-session",
        dest="optimizer_session",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to optimizer log file from a previous optimization session"
    )

    train_parser.add_argument(
        "-s", "--save-model-path",
        dest="save_model_path",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to save the trained model after training (optional)"
    )

    return parser.parse_args()



async def main():
    args = parse_arguments()

    if args.mode == "optimize":
        optimize_main(
            filepath=args.filepath,
            num_entries=args.num_entries,
            mask_columns=args.mask_columns)

    elif args.mode == "train":
        await train_main(
            filepath=args.filepath,
            pretrained_model_path=args.pretrained_model,
            optimizer_session=args.optimizer_session,
            save_model_path=args.save_model_path)

    else:
        log.error("Invalid mode selected.")


if __name__ == "__main__":
    asyncio.run(main())
