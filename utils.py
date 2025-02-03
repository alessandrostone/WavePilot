"""Utils"""

from ast import literal_eval
import pandas as pd
import numpy as np
#import plotly.express as px
import plotly.graph_objects as go
import torch
from torch import nn

# Set GPU device if available according to OS
# Funzione per rilevare il dispositivo migliore
def get_device() -> torch.device:
    """Set GPU device if available according to OS

    Returns:
        torch.device: device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device: torch.device = get_device()
print(f"Using device: {device}")


# Read and get best hyperparams from log session
def get_hyperparams_from_log(log_file):
    """get_hyperparams_from_log

    Args:
        log_file (_type_): _description_

    Returns:
        dict: params
    """
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    params = {}
    for line in lines:
        if "Best VAE hyperparams" in line:
            # Get autoencoder's best hyperp fomr log
            params_str = line.split("Best VAE hyperparams: ")[1].split(" with")[0]
            params_tuple = literal_eval(params_str)
            params['vae'] = {'n_epochs': params_tuple[0], 'learning_rate': params_tuple[1],
                              'weight_decay': params_tuple[2], 'n_layers': params_tuple[3],
                              'activation': params_tuple[4], 'beta': params_tuple[5]}
        elif "Best RBF params" in line:
            # Get interpolator's best hyperp fomr log
            params_str = line.split("Best RBF params: ")[1].split(" with")[0]
            params_tuple = literal_eval(params_str)
            params['rbf'] = {'smoothing': params_tuple[0],
                              'kernel': params_tuple[1],
                              'epsilon': params_tuple[2],
                              'degree': params_tuple[3]}

    return params

# Get activation function module from string
def get_activation_function(activation_name):
    """get_activation_function

    Args:
        activation_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    activation_functions = {
        'ReLU': nn.ReLU(),
        'LeakyReLU': nn.LeakyReLU(),
        'Sigmoid': nn.Sigmoid()
    }
    return activation_functions.get(activation_name, None)


# # Compute loss for optimization
# def compute_validation_error(model, criterion, val_data):
#     with torch.no_grad():
#         val_data = val_data.drop(['ID', 'PRESET_NAME'], axis=1)
#         val_data = torch.tensor(val_data.values).float()
#         _, output = model(val_data)
#         loss = criterion(output, val_data)
#     return loss.item()

def select_random_entries(input_csv, ouput_csv, n):
    """select_random_entries

    Args:
        input_csv (_type_): _description_
        ouput_csv (_type_): _description_
        n (_type_): _description_
    """
    df = pd.read_csv(input_csv)
    df['ID'] = range(1, len(df) + 1)
    df.to_csv(input_csv, index=False)

    df_sample = df.sample(n)
    df_sample.to_csv(ouput_csv, index=False)


def plot_reconstruction_error(original_data, reduced_data, reconstructed_data):
    """PLOT RECONSTRUCTION ERROR

    Args:
        original_data (_type_): _description_
        reduced_data (_type_): _description_
        reconstructed_data (_type_): _description_
    """
    reconstruction_error = np.mean(np.square(original_data - reconstructed_data), axis=1)
    average_error = np.mean(reconstruction_error)
    print(average_error)

    # Estrai le coordinate x, y, z dai dati ridotti
    x, y, z = reduced_data.T

    # Crea un grafico a dispersione 3D dell'errore di ricostruzione
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                   marker=dict(size=5,
                                                color=reconstruction_error,
                                                  colorscale='Viridis'))])

    fig.update_layout(title='3D Scatter Plot of Reconstruction Error',
                  scene=dict(xaxis_title='X',
                             yaxis_title='Y',
                             zaxis_title='Reconstruction Error'))
    fig.show()
