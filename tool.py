import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import pickle
from model import Condiitonal_gmms
import pickle
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def text_to_input(text_input):
    """Convert the input text to the input of the model
    text: str
    
    conver the text to the input of the model
    for example 
    text_input = [100,'2 onder 1 kap', '1940-1979', '1 Alleenstaande']
    input_x = [100, 0, 0, 0]
    """
    input_x = []
    input_x.append(text_input[0])
    
    if text_input[1] == '2 onder 1 kap':
        input_x.append(0)
    elif text_input[1] == 'Appartement':
        input_x.append(1)
    elif text_input[1] == 'Overige':
        input_x.append(2)
    elif text_input[1] == 'Rijtjeswoning':
        input_x.append(3)
    elif text_input[1] == 'Vrijstaande woning':
        input_x.append(4)
    
    if text_input[2] == '1940-1979':
        input_x.append(0)
    elif text_input[2] == '1980-heden':
        input_x.append(1)
    elif text_input[2] == 'voor 1940':
        input_x.append(2)
        
    if text_input[3] == '1 Alleenstaande':
        input_x.append(0)
    elif text_input[3] == '2 Gezin met kinderen':
        input_x.append(1)
    elif text_input[3] == '3 Paar zonder kinderen':
        input_x.append(2)
        
    return input_x
    
    
def input_encode(input_x):
    """x has 12 columns but it is the hot code of the condition
    give 0,1,2.. which represent the typs, transform it to the hot code
    input_x: list
    first 4 elements: [1000, 0, 0, 0]
    first element: real number, annual energy consumption
    second element: 
        5 types of building 
        0: 2 onder 1 kap
        1: Appartement
        2: Overige
        3: Rijtjeswoning
        4：Vrijstaande woning
    third element:
        3 types of building age
        0: 1940-1979
        1: 1980-heden
        2: voor 1940
    forth element:
        3 types of family
        0: 1 Alleenstaande
        1: 2 Gezin met kinderen
        2: 3 Paar zonder kinderen
    """
    give_x = []
    
    # annual energy consumption
    give_x.append(input_x[0])
    
    # building type
    if input_x[1] == 0:
        give_x.extend([1, 0, 0, 0, 0])
    elif input_x[1] == 1:
        give_x.extend([0, 1, 0, 0, 0])
    elif input_x[1] == 2:
        give_x.extend([0, 0, 1, 0, 0])
    elif input_x[1] == 3:
        give_x.extend([0, 0, 0, 1, 0])
    elif input_x[1] == 4:
        give_x.extend([0, 0, 0, 0, 1])
        
    # building age
    if input_x[2] == 0:
        give_x.extend([1, 0, 0])
    elif input_x[2] == 1:
        give_x.extend([0, 1, 0])
    elif input_x[2] == 2:
        give_x.extend([0, 0, 1])
        
    # family
    if input_x[3] == 0:
        give_x.extend([1, 0, 0])
    elif input_x[3] == 1:
        give_x.extend([0, 1, 0])
    elif input_x[3] == 2:
        give_x.extend([0, 0, 1])
        
    return np.array(give_x)
    

# load the model
def sample_and_plot(given_x, number_samples=1000):
    # Load the Conditional Gaussian Mixture Model
    with open('c_gmm.pkl', 'rb') as f:
        c_gmm = pickle.load(f)
    
    # print(given_x)
    # Generate samples
    samples = c_gmm.sample(given_x, n_samples=number_samples)
    
    # Replace values smaller than 0 with 0
    samples[samples < 0] = 0
    
    # Calculate the sum of each sample for coloring
    sample_sums = samples.sum(axis=1)
    
    # Normalize sample sums for color mapping
    norm = Normalize(vmin=sample_sums.min(), vmax=sample_sums.max())
    smappable = ScalarMappable(cmap='viridis', norm=norm)
    smappable.set_array([])
    
    # Time labels for the x-axis
    time_labels = [f'{hour:02d}:{minute:02d}' for hour in range(24) for minute in [0, 15, 30, 45]]
    
    # Plot the samples with colors based on their sum
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(samples.shape[0]):
        ax.plot(time_labels, samples[i, :], color=smappable.to_rgba(sample_sums[i]), alpha=0.3)
    
    # Configure x-axis to display time labels every 2 hours
    ax.set_xticks(time_labels[::8])
    ax.set_xticklabels(time_labels[::8], rotation=45, ha='right', fontsize=12)  # Larger tick labels
    
    # Title and labels with increased font sizes
    ax.set_title('Generated load profiles based on input conditions', fontsize=20)
    ax.set_xlabel('Time of day [Hour]', fontsize=20)
    ax.set_ylabel('Energy consumption (Wh)', fontsize=20)
    
    # plt.tight_layout(pad=3.0) 
    
    # Adjust tick size for both axes
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    # Colorbar with larger label size
    cbar = plt.colorbar(smappable, ax=ax)
    cbar.set_label('Daily energy consumption (Wh)', fontsize=20)
    cbar.ax.tick_params(labelsize=15)  # Larger colorbar labels
    
    # plt.subplots_adjust(top=0.85)  # You can adjust this value as needed to fit the title
    
    plt.tight_layout()
    plt.show()
    
    return fig, samples

# text_input = [100,'2 onder 1 kap', '1940-1979', '1 Alleenstaande']
# input_x = text_to_input(text_input)
# give_x = input_encode(input_x)
# fig, sample = sample_and_plot(give_x, number_samples=365)
