import argparse
import itertools
import numpy as np
import torch

from data import DataLoader
from logger import setup_logger
from model import VectorReducer
from multiprocessing import Pool, cpu_count, Manager, Process
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import *




def get_arguments():
    parser = argparse.ArgumentParser()

    # (Small) dataset of the presets to be reduced (Mandatory)
    parser.add_argument('-f', '--filepath',
                        dest='filepath',
                        type=str,
                        required=True,
                        help='Dataset of the presets to be reduced.')
    
    parser.add_argument('-n', '--num_entries',
                        dest='num_entries',
                        type=int,
                        default=None,
                        help='Number of random entries to select from the dataset.')
    
    # Large dataset of presets to pretrain the model (Optional)
    parser.add_argument('-F', '--filepath_pretrain',
                        dest='filepath_pretrain',
                        type=str,
                        default=None,
                        help='Large dataset to pretrain the model.')
    
    # Filepath where to save the pretrained model, only necessary if -F is passed
    parser.add_argument('-s', '--filepath_save_pretrain',
                        dest='filepath_save_pretrain',
                        type=str,
                        default=None)
    
    parser.add_argument('-d', '--disable_split',
                        dest='disable_split',
                        action='store_false',
                        help='Disable train/test split and use the entire dataset for both training and validation. Default split enabled.')
    
    return parser.parse_args()


log = setup_logger('Opmization Session', file=True)



# Load data
def load_data(filepath, num_entries=None):
    loader = DataLoader(filepath)
    df = loader.load_presets()

    if num_entries:
        df = df.sample(n=num_entries, random_state=42)
        print(f'Randomly selected {num_entries} entries from ther dataset.')
    else:
        print(f'Using the entire dataset.')
    
    return df


# Compute KL divergence
def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


# Calculate validation error
def compute_validation_error(reducer, data):
    data_tensor = torch.tensor(data.values).float().to(reducer.device)
    with torch.no_grad():
        return reducer.compute_loss(data_tensor, compute_gradients=False)
    

# Progress listener process
def progress_listener(queue, total):
    with tqdm(total=total) as pbar:
        while True:
            msg = queue.get()
            if msg == 'DONE':
                break
            pbar.update(1)


def train_and_validate(queue, n_epochs, params, df_train, df_test, pretrained_model=None):
    try:
        # unpack params
        learning_rate, weight_decay, n_layers, layer_dim, activation_name, kl_beta, mse_beta = params
        activation = get_activation_function(activation_name)

        # Costruzione e addestramento del modello
        reducer = VectorReducer(df_train, learning_rate, weight_decay, n_layers, layer_dim, activation, kl_beta, mse_beta, pretrained_model)
        reducer.train_vae(n_epochs)  # Sposta i dati e il modello sulla GPU/CPU corretta

        # Validazione
        validation_error = compute_validation_error(reducer, df_test)
        reducer.move_to_cpu()
        return validation_error, params, reducer.model

    except Exception as e:
        print(f"Error during VAE optimization: {e}")
        return float('inf'), params, None  # Ritorna un valore alto per continuare l'ottimizzazione
    finally:
        queue.put(1) # Notify progress


def optimize_vae(df_train, df_test, log_prefix, save_pretrained_model=False, save_filepath=None, pretrained_model=None):
    print(f"{log_prefix} Starting VAE optimization...")

    # VAE's params' grid
    # vae_grid = {
    #     'n_epochs': [5, 10, 25, 50, 100, 150, 200],
    #     'learning_rate': np.logspace(-5, -2, num=3),
    #     'weight_decay': np.logspace(-5, -2, num=3),
    #     'n_layers': list(range(1, 5)),
    #     'layer_dim': [32, 64, 128, 256],
    #     'activation': ['ReLU', 'LeakyReLU', 'Sigmoid'],
    #     'kl_beta': np.linspace(0.05, 0.5, num=10),
    #     'mse_beta': np.linspace(0.3, 1.0, num=10)
    # }

        # VAE's params' grid
    vae_grid = {
        'n_epochs': [5, 10, 25, 50],
        'learning_rate': np.logspace(-5, -2, num=3),
        'weight_decay': np.logspace(-5, -2, num=3),
        'n_layers': list(range(1, 3)),
        'layer_dim': [64, 128],
        'activation': ['ReLU', 'LeakyReLU'],
        'kl_beta': np.linspace(0.05, 0.5, num=5),
        'mse_beta': np.linspace(0.3, 1.0, num=5)
    }


    # Get all combinations of hyperparameters
    param_combinations = list(itertools.product(*vae_grid.values()))
    total_combinations = len(param_combinations)
    print(f"{log_prefix} Total hyperparameter combinations: {len(param_combinations)}")

    # Prepare for multiprocessing
    manager = Manager()
    queue = manager.Queue()
    listener = Process(target=progress_listener, args=(queue, total_combinations))
    listener.start()

    input_data = [(queue, params[0], params[1:], df_train, df_test, pretrained_model) for params in param_combinations]


    # Find the best result
    best_validation_error = float('inf')
    best_params = None
    best_model = None

    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(train_and_validate, input_data)


    for _, result in enumerate(results):
        validation_error, params, model = result

        print(f"{log_prefix} Validation Error: {validation_error:.4f} | Params: {params}")

        if validation_error < best_validation_error:
            best_validation_error = validation_error
            best_params = params
            best_model = model

    log.info(f'Best VAE hyperparams: {best_params} with a validation error of {best_validation_error}')

    if save_pretrained_model:
        torch.save(best_model, f'{save_filepath}.pt')

    queue.put('DONE')
    listener.join()

    return best_params, best_model




def main():
    try:
        args = get_arguments()
        torch.manual_seed(42)

        device = get_device()
        print(f"Using device: {device}")

        filepath = args.filepath
        num_entries = args.num_entries
        filepath_pretrain = args.filepath_pretrain
        filepath_save_pretrain = args.filepath_save_pretrain
        disable_split = args.disable_split

        df = load_data(filepath, num_entries)

        if disable_split:
            df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
        else:
            df_train, df_test = df, df

        if filepath_pretrain:
            df_pretrain = load_data(filepath_pretrain)
            df_train_pretrain, df_test_pretrain = train_test_split(df_pretrain, test_size=0.2, random_state=42)
            best_params_pretrain, best_model_pretrain = optimize_vae(df_train_pretrain, df_test_pretrain, 'Pretrain', save_pretrained_model=True, save_filepath=filepath_save_pretrain)
            best_params_train, _ = optimize_vae(df_train, df_test, 'Train', pretrained_model=best_model_pretrain)
        else:
            best_params_train, _ = optimize_vae(df_train, df_test, 'Train')

    except Exception as e:
        print(f'Error in main: {e}')


if __name__ == '__main__':
    main()