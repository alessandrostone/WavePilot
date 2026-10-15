import argparse
import sys
import threading
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
import optuna
#from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
#from optuna.storages import RDBStorage

import torch  # to decide n_jobs when GPU/MPS is present
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import mahalanobis, pdist
from tqdm import tqdm

from constants import (
    N_TRIALS_RBF,
    N_TRIALS_VAE,
    GLOBAL_SEED,
    SEARCH_EPOCHS,
    FINAL_EPOCHS,
    RBF_FIXED_EPSILON_KERNELS,
    RBF_MIN_DEGREE,
    RBF_DEGREE_LOCK,
    RBF_PARAM_RANGES,
    RECON_ACCURACY_THRESHOLD,
    VAE_PARAM_RANGES,
)

from data import DataLoader
from dispatcher import SUGGEST_DISPATCH
from logger import setup_logger
from model import VectorReducer, TrainConfig
from utils import set_global_seeds, space_fingerprint, LatentScaler  # external latent z-score

optuna.logging.set_verbosity(optuna.logging.WARNING)

log = setup_logger('OptimizationLogger', file=True)
log_progress = setup_logger('ProgressLogger', file=True)


# -----------------------------
# Argument parsing
# -----------------------------
def get_arguments():
    parser = argparse.ArgumentParser(
        description="WavePilot Hyperparameter Optimizer - Optimize VAE and RBF parameters for preset dimensionality reduction",
        epilog="""
Examples:
  # Basic optimization on full dataset
  python optimize.py -f presets.csv

  # Use only 100 random samples for faster testing
  python optimize.py -f presets.csv -n 100

  # Custom number of trials for VAE and RBF optimization
  python optimize.py -f presets.csv --vae-trials 50 --rbf-trials 30

  # Mask specific columns from the dataset
  python optimize.py -f presets.csv -m cutoff resonance volume

The script performs:
  1. VAE hyperparameter search (learning rate, KL-beta, latent dimensions, architecture)
  2. Final VAE training with best parameters
  3. RBF hyperparameter search (kernel type, smoothing, epsilon, degree)
  4. Outputs best parameters and reconstruction accuracy metrics

Configuration defaults are loaded from constants.py
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("-f", "--filepath",
                        dest="filepath",
                        type=str,
                        required=True,
                        metavar="PATH",
                        help="Path to input dataset CSV file (required)")

    parser.add_argument("-n", "--num-entries",
                        dest="num_entries",
                        type=int,
                        default=None,
                        metavar="N",
                        help="Number of random entries to use from dataset (default: use all)")

    parser.add_argument("-m", "--mask-columns",
                        dest="mask_columns",
                        nargs="*",
                        default=None,
                        metavar="COL",
                        help="Space-separated list of columns to exclude from optimization")

    parser.add_argument("--vae-trials",
                        dest="vae_trials",
                        type=int,
                        default=N_TRIALS_VAE,
                        metavar="N",
                        help=f"Number of VAE optimization trials (default: {N_TRIALS_VAE})")

    parser.add_argument("--rbf-trials",
                        dest="rbf_trials",
                        type=int,
                        default=N_TRIALS_RBF,
                        metavar="N",
                        help=f"Number of RBF optimization trials (default: {N_TRIALS_RBF})")

    parser.add_argument("--seed",
                        dest="seed",
                        type=int,
                        default=GLOBAL_SEED,
                        metavar="N",
                        help=f"Random seed for reproducibility (default: {GLOBAL_SEED})")

    return parser.parse_args()


# -----------------------------
# Data loading
# -----------------------------
def load_data(filepath, num_entries=None, mask_columns=None):
    loader = DataLoader(filepath, mask_columns)
    df = loader.load_presets()

    if num_entries:
        selected_idx = np.random.choice(df.shape[0], size=num_entries, replace=False)
        df = df[selected_idx]

        log_progress.info("Randomly selected %d entries from the dataset", num_entries)
        log_progress.info("Selected indices from dataset: %s", selected_idx)
    else:
        log_progress.info("Using the entire dataset!")

    return df


# -----------------------------
# VAE: training + evaluation
# -----------------------------
def train_and_validate(params, df, trial_number):
    """
    Esegue un singolo trial VAE in un worker di ProcessPool in modo deterministico.
    - Fissa il seed per-trial: GLOBAL_SEED + trial_number
    - Allena su CPU
    - Ritorna la metrica di ricostruzione (float)
    """
    set_global_seeds(GLOBAL_SEED + int(trial_number))

    lr          = params["learning_rate"]
    kl_beta     = params["kl_beta"]
    latent_dim  = params["latent_dim"]
    width_scale = params["width_scale"]
    depth       = params["depth"]
    round_to    = params["round_to"]
    grad_clip   = params.get("grad_clip", 1.0)
    batch_size  = params.get("batch_size", 0)  # 0 = full-batch

    reducer = VectorReducer(df, latent_dim=latent_dim)  # Auto-detect GPU/MPS/CPU

    cfg = TrainConfig(
        epochs=SEARCH_EPOCHS,
        lr=lr,
        kl_beta=kl_beta,
        deterministic=True,
        hidden_dims=None,
        grad_clip=grad_clip,
        batch_size=batch_size,
    )
    setattr(cfg, "width_scale", width_scale)
    setattr(cfg, "depth",       depth)
    setattr(cfg, "round_to",    round_to)

    reducer.fit(cfg)  # No pruning (deterministic)
    return reducer.reconstruction_mse()



# -----------------------------
# RBF: fitting + validation
# -----------------------------
def interpolate_and_validate(
    params,
    original_data,
    Z_std,                # normalised latent coordinates (or raw if you choose not to normalise)
    median_dist,          # median pairwise distance in the same space as Z_std
    min_degree,
    fixed_epsilon_kernels,
    degree_lock,
    trial_number
):
    """
    Evaluate an RBF configuration by computing the mean Mahalanobis distance on training points,
    plus a penalty for out-of-bounds outputs in [0,1].
    """
    set_global_seeds(GLOBAL_SEED + int(trial_number))

    try:
        smoothing     = params['smoothing']
        kernel        = params['kernel']
        degree        = params['degree']
        epsilon_scale = params['epsilon_scale']  # new: scale relative to median geometry

        # Epsilon: fixed for some kernels, relative otherwise
        if kernel in fixed_epsilon_kernels:
            epsilon = 1.0
        else:
            md = float(median_dist if median_dist is not None else 1.0)
            epsilon = max(1e-12, float(epsilon_scale) * md)

        # Degree lock for SPD kernels
        if degree_lock and kernel in degree_lock:
            degree = degree_lock[kernel]

        # Minimal degree requirement for CPD kernels
        if kernel in min_degree and degree < min_degree[kernel]:
            log_progress.warning("Skipping configuration: kernel=%s, degree=%d below minimum", kernel, degree)
            return float('inf'), params

        # Check polynomial terms vs N
        num_poly_terms = 0 if degree == -1 else (degree + 1) * (degree + 2) // 2
        if original_data.shape[0] < num_poly_terms:
            log_progress.warning("Skipping: insufficient N for degree=%d", degree)
            return float('inf'), params

        # Fit and evaluate on the same points (we want faithful interpolation on anchors)
        interpolator = RBFInterpolator(
            Z_std,
            original_data,
            smoothing=smoothing,
            kernel=kernel,
            epsilon=epsilon,
            degree=degree
        )

        interpolated_data = interpolator(Z_std)
        if not np.all(np.isfinite(interpolated_data)):
            log_progress.warning("Non-finite interpolated_data; returning inf for this trial!")
            return float('inf'), params

        # Mahalanobis with robust fallbacks
        cov_matrix = np.cov(original_data, rowvar=False)
        cov_matrix = np.nan_to_num(cov_matrix, copy=False)
        cov_matrix = cov_matrix + 1e-12 * np.eye(cov_matrix.shape[1])
        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            try:
                inv_cov = np.linalg.pinv(cov_matrix)
            except Exception as e:
                log_progress.warning("Fallback to identity matrix: %s", str(e))
                inv_cov = np.eye(original_data.shape[1])

        distances = []
        for i in range(len(original_data)):
            try:
                d = mahalanobis(original_data[i], interpolated_data[i], inv_cov)
            except Exception as e:
                log_progress.warning("Mahalanobis failed, using Euclidean: %s", str(e))
                d = np.linalg.norm(original_data[i] - interpolated_data[i])
            distances.append(d)

        # Penalty for out-of-bounds wrt [0,1]
        out_of_bounds = np.sum((interpolated_data < 0) | (interpolated_data > 1))
        penalty = out_of_bounds / interpolated_data.size
        mean_distance = np.mean(distances)

        validation_distance = mean_distance + 30 * (penalty ** 1.5)
        if not np.isfinite(validation_distance):
            log_progress.warning("Non-finite validation_distance; returning inf for this trial.")
            return float('inf'), params
        return validation_distance, params

    except np.linalg.LinAlgError:
        log_progress.warning("Singular matrix error with kernel=%s, degree=%d", kernel, degree)
        return float('inf'), params

    except ValueError as e:
        if "At least" in str(e):
            log_progress.warning("Insufficient data points: kernel=%s, degree=%d", kernel, degree)
            return float('inf'), params
        else:
            log_progress.error("Unexpected ValueError: %s", e, exc_info=True)
            raise e

    except Exception as e:
        log_progress.error("Unexpected error in interpolate_and_validate: %s", e, exc_info=True)
        return float('inf'), params


# -----------------------------
# Optimizer orchestrator
# -----------------------------
class Optimizer:
    """
    Since datasets are tiny, we train on the full set (no train/test split).
    Evaluation is computed on the full set as well (overfitting is desired).
    """
    def __init__(self, df):
        self.df = df
        self.study_vae = None
        self.study_rbf = None
        # optional artefacts for later reuse
        self.rbf_latent_scaler_ = None
        self.rbf_median_dist_ = None

    def objective_vae(self, trial):
        """Optuna objective for VAE: minimise reconstruction MSE."""
        params = {
            name: SUGGEST_DISPATCH[config["type"]](trial, name, config)
            for name, config in VAE_PARAM_RANGES.items()
        }
        try:
            val_mse = train_and_validate(params, self.df, trial.number)
            return val_mse
        except optuna.TrialPruned:
            raise

    # Set by the pipeline: called with (done, total) after every finished
    # trial so the host can show real progress. Advisory: never raises.
    progress_cb = None

    def _report_progress(self, done, total):
        cb = getattr(self, "progress_cb", None)
        if cb is None:
            return
        try:
            cb(done, total)
        except Exception as e:  # noqa: BLE001 - progress must not break the search
            log.debug("progress callback failed: %s", e)

    def optimize_vae(self, n_trials=N_TRIALS_VAE, show_progress=True):
        # Reproducible sampler
        sampler = TPESampler(
            seed=GLOBAL_SEED,
            n_startup_trials=10,
            consider_prior=True,
            prior_weight=1.0
        )

        try:
            suffix = space_fingerprint(VAE_PARAM_RANGES)
            study_name = f"wavepilot_vae_{suffix}"
        except Exception:
            study_name = "wavepilot_vae_study"

        self.study_vae = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            storage=None,
            study_name=study_name
        )

        set_global_seeds(GLOBAL_SEED)

        # Batch parameters
        # CRITICAL: Only use parallel workers if no GPU/MPS available
        # Multiple processes trying to use GPU/MPS simultaneously will deadlock
        # Also check for frozen executable (PyInstaller) which doesn't support multiprocessing well
        is_frozen = getattr(sys, 'frozen', False)
        if is_frozen:
            max_workers = 1  # Frozen executables don't support multiprocessing
            log.info("Frozen executable detected - using sequential execution for VAE optimization")
        elif torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            max_workers = 1  # Sequential execution when GPU/MPS available
            log.info("GPU/MPS detected - using sequential execution for VAE optimization")
        else:
            max_workers = max(1, cpu_count())  # Parallel execution on CPU
            log.info("CPU-only mode - using %d parallel workers for VAE optimization", max_workers)

        total = n_trials
        i = 0
        pbar = tqdm(total=total, desc="DVAE HPO", leave=True, disable=not show_progress)
        done = 0  # own count: a disabled tqdm never advances pbar.n

        try:
            while i < total:
                # Ask for a batch (deterministic order)
                batch = []
                for _ in range(min(max_workers, total - i)):
                    trial = self.study_vae.ask()
                    params = {
                        name: SUGGEST_DISPATCH[config["type"]](trial, name, config)
                        for name, config in VAE_PARAM_RANGES.items()
                    }
                    batch.append((trial, params))

                results = [None] * len(batch)

                if max_workers == 1:
                    # Sequential execution - avoid ProcessPoolExecutor entirely
                    for idx, (trial, params) in enumerate(batch):
                        results[idx] = train_and_validate(params, self.df, trial.number)
                        pbar.update(1)
                        done += 1
                        self._report_progress(done, total)
                else:
                    # Parallel execution
                    with ProcessPoolExecutor(max_workers=max_workers) as ex:
                        future_to_idx = {}
                        for idx, (trial, params) in enumerate(batch):
                            fut = ex.submit(
                                train_and_validate,
                                params,
                                self.df,
                                trial.number
                            )
                            future_to_idx[fut] = idx

                        # Progress bar when a worker ends
                        for fut in as_completed(future_to_idx.keys()):
                            idx = future_to_idx[fut]
                            results[idx] = fut.result()
                            pbar.update(1)
                            done += 1
                            self._report_progress(done, total)

                # tell in batch order
                for (trial, _), value in zip(batch, results):
                    self.study_vae.tell(trial, value)

                i += len(batch)

        finally:
            pbar.close()

        return self.study_vae.best_params 


    def objective_rbf(self, trial, original_data, Z_std, median_dist):
        """Optuna objective for RBF: minimise validation distance (Mahalanobis + penalty)."""
        params = {
            name: SUGGEST_DISPATCH[config["type"]](trial, name, config)
            for name, config in RBF_PARAM_RANGES.items()
        }

        validation_distance, _ = interpolate_and_validate(
            params=params,
            original_data=original_data,
            Z_std=Z_std,
            median_dist=median_dist,
            min_degree=RBF_MIN_DEGREE,
            fixed_epsilon_kernels=RBF_FIXED_EPSILON_KERNELS,
            degree_lock=RBF_DEGREE_LOCK,
            trial_number=trial.number
        )
        return validation_distance

    def optimize_rbf(self, original_data, reduced_data, n_trials=N_TRIALS_RBF, normalise_latent=True, show_progress=True):
        """
        RBF search on latent space:
          - optionally z-score the latent ONCE,
          - precompute the median pairwise distance in that space,
          - search epsilon via epsilon_scale × median_dist.
        """
        set_global_seeds(GLOBAL_SEED)

        sampler = TPESampler(
            seed=GLOBAL_SEED,
            n_startup_trials=5,
            consider_prior=True,
            prior_weight=1.0
        )

        try:
            suffix = space_fingerprint(RBF_PARAM_RANGES)
            study_name = f"wavepilot_rbf_{suffix}"
        except Exception:
            study_name = "wavepilot_rbf_study"

        self.study_rbf = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            storage=None,
            study_name=study_name
        )


        # --- Prepare the latent space used for RBF fitting ---
        if normalise_latent:
            scaler = LatentScaler().fit(reduced_data)
            Z_std = scaler.transform(reduced_data)
            self.rbf_latent_scaler_ = scaler
        else:
            Z_std = reduced_data
            self.rbf_latent_scaler_ = None

        if Z_std.shape[0] >= 2:
            dvec = pdist(Z_std, metric="euclidean")
            median_dist = float(np.median(dvec)) if dvec.size > 0 else 1.0
        else:
            median_dist = 1.0
        self.rbf_median_dist_ = median_dist


        # Completed-trial count for the host's progress (trials finish on
        # worker threads, in any order)
        finished = itertools.count(1)
        count_lock = threading.Lock()

        def _on_trial_done(study, trial):
            with count_lock:
                done = next(finished)
            self._report_progress(done, n_trials)

        # CPU-bound, safe to parallelise across all cores
        self.study_rbf.optimize(
            lambda trial: self.objective_rbf(trial, original_data, Z_std, median_dist),
            n_trials=n_trials,
            n_jobs=-1,  # Use all available CPU cores
            show_progress_bar=show_progress,
            callbacks=[_on_trial_done]
        )

        best_params = self.study_rbf.best_params
        return best_params


# -----------------------------
# Run final training with best VAE params
# -----------------------------
def run_training(best_params_train, df_train):
    """
    Train the final model with the best hyperparameters and return the fitted reducer.
    """
    reducer = VectorReducer(df_train, latent_dim=best_params_train["latent_dim"])

    cfg = TrainConfig(
        epochs=FINAL_EPOCHS,
        lr=best_params_train["learning_rate"],
        kl_beta=best_params_train["kl_beta"],
        deterministic=True,
        hidden_dims=None,
        grad_clip=best_params_train["grad_clip"],
        batch_size=best_params_train.get("batch_size", 0),
    )
    setattr(cfg, "width_scale", best_params_train["width_scale"])
    setattr(cfg, "depth",       best_params_train["depth"])
    setattr(cfg, "round_to",    best_params_train["round_to"])

    reducer.fit(cfg, show_progress=True)
    return reducer


# -----------------------------
# Main orchestration
# -----------------------------
def main(filepath, num_entries=None, mask_columns=None, vae_trials=N_TRIALS_VAE, rbf_trials=N_TRIALS_RBF, seed=GLOBAL_SEED):
    set_global_seeds(seed)
    df = load_data(filepath, num_entries, mask_columns)

    optimizer = Optimizer(df)

    # ---- VAE hyperparameter search (minimise MSE) ----
    best_vae_params = optimizer.optimize_vae(n_trials=vae_trials)

    # ---- Train final VAE with best params ----
    reducer = run_training(best_vae_params, df)

    # Deterministic metrics and artefacts
    accuracy = reducer.reconstruction_accuracy(threshold=RECON_ACCURACY_THRESHOLD)  # already in percentage
    latent_points = reducer.transform()                         # μ to feed RBF

    # ---- RBF hyperparameter search on (optionally normalised) latent space ----
    best_rbf_params = optimizer.optimize_rbf(
        original_data=df,
        reduced_data=latent_points,
        n_trials=rbf_trials,
        normalise_latent=True  # keep True to match the rest of the pipeline
    )

    # Logging
    log_progress.info("VAE Reconstruction Accuracy: %.2f%%", accuracy)
    log_progress.info("Best VAE Parameters: %s", best_vae_params)
    log_progress.info("Best RBF Parameters: %s", best_rbf_params)

    log.info("VAE Reconstruction MSE: %.6f | Acc: %.2f%%", optimizer.study_vae.best_value, accuracy)
    log.info("Best VAE Parameters: %s", best_vae_params)
    log.info("Best RBF Parameters: %s | Validation distance: %.6f", best_rbf_params, optimizer.study_rbf.best_value)


if __name__ == "__main__":
    args = get_arguments()
    main(
        filepath=args.filepath,
        num_entries=args.num_entries,
        mask_columns=args.mask_columns,
        vae_trials=args.vae_trials,
        rbf_trials=args.rbf_trials,
        seed=args.seed
    )

