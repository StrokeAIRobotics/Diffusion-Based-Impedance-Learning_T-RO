import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gc
from torch.utils.data import DataLoader

from models import NoisePredictorTransformerWithCrossAttentionTime
from data import ImpedanceDatasetDiffusion, load_robot_data, compute_statistics_per_axis, normalize_data_per_axis
from train_val_test import train_model_diffusion, test_model, inference_simulation
from utils import set_seed
from datetime import datetime

set_seed(42)  # Set seed for reproducibility

def main():
    """
    Example main function to execute the training and inference simulation for diffusion based impedance adaptation.
    """ 

    # Clear any previous GPU memory
    torch.cuda.empty_cache()
    gc.collect()

    # Definition of parameters
    seq_length = 16 #seq len of data
    input_dim = seq_length * 3  # Flattened input dimension
    hidden_dim = 512 #hidden dim of the model
    batch_size =64 #batch size
    num_epochs = 20 #number of epochs
    learning_rate = 1e-4 #learning rate
    noiseadding_steps = 5 # Number of steps to add noise
    use_forces = True  # Set this to True if you want to use forces as input to the model
    noise_with_force = False # Set this to True if you want to use forces as the noise
    #if force is used as noise, then force should not be used as input
    if noise_with_force:
            use_forces = False

    beta_start = 0.0001 #for the noise diffusion model
    beta_end = 0.04 #for the noise diffusion model
    max_grad_norm=7.0 #max grad norm for gradient clipping 
    add_gaussian_noise = False # to add additional guassian noise
    early_stop_patience = 8 #for early stopping
    save_interval = 20
    save_path = "save_checkpoints/Test"
    timestamp = datetime.now().strftime("%Y-%"
    "m-%d_%H-%M-%S")

    hyperparams = {
    "seq_length": seq_length,
    "hidden_dim": hidden_dim,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "learning_rate": learning_rate,
    "noiseadding_steps": noiseadding_steps,
    "use_forces": use_forces,
    "noise_with_force": noise_with_force,
    "beta_start": beta_start,
    "beta_end": beta_end,
    "max_grad_norm": max_grad_norm,
    "add_gaussian_noise": add_gaussian_noise,
    "early_stop_patience": early_stop_patience
    }

    file_path = "Your_Path\\Data\\Parkour" #path to data txt files for training
    file_path_application = "Your_Path\\Data\\Parkour\\ApplicationData" #path to data for inference simulation/whole sequence

    # Load real data
    #data = load_robot_data(file_path, seq_length, use_overlap=True)
    data = load_robot_data(file_path, seq_length, use_overlap=True)
    data_application = load_robot_data(file_path_application, seq_length, use_overlap=False)
    # Compute per-axis normalization statistics
    stats = compute_statistics_per_axis(data)
    # Normalize data per axis
    normalized_data = normalize_data_per_axis(data, stats)
    normalized_data_application = normalize_data_per_axis(data_application, stats)
    # Define split ratios
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    # Compute split indices
    total_size = len(normalized_data)
    train_split = int(total_size * train_ratio)
    val_split = train_split + int(total_size * val_ratio)
    test_split = val_split + int(total_size * test_ratio)
    # Split data
    train_data = normalized_data[:train_split]
    val_data = normalized_data[train_split:val_split]
    test_data = normalized_data[val_split:test_split]

    # Create datasets with per-axis normalization
    train_dataset = ImpedanceDatasetDiffusion(train_data, stats)
    val_dataset = ImpedanceDatasetDiffusion(val_data, stats)
    test_dataset = ImpedanceDatasetDiffusion(test_data, stats)
    application_dataset = ImpedanceDatasetDiffusion(normalized_data_application, stats)
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    application_loader = DataLoader(application_dataset, batch_size=1, shuffle=False)

    # Model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "DM"
    model = NoisePredictorTransformerWithCrossAttentionTime(seq_length, hidden_dim, use_forces=use_forces).to(device)

    #choose optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion=nn.SmoothL1Loss()


    print("_____________________________________________")
    print("-----Training and Validation-----")
    # Train and validate
    train_losses, val_loss = train_model_diffusion(
        model,
        train_loader, 
        val_loader,
        optimizer, 
        criterion, 
        device, 
        num_epochs, 
        noiseadding_steps, 
        beta_start, 
        beta_end, 
        use_forces,
        noise_with_force, 
        max_grad_norm,
        add_gaussian_noise,
        save_interval, 
        save_path,
        early_stop_patience)
    

    print("_____________________________________________")
    print("-----Test model-----")
    #generate save path test folder
    save_path_test = os.path.join(save_path, f"{model_name}_{timestamp}_test")
    os.makedirs(save_path_test, exist_ok=True)

    # Clear GPU memory after training
    del train_loader, val_loader, train_dataset, val_dataset
    torch.cuda.empty_cache()

    # Load best model
    best_model_path = os.path.join(save_path, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.to(device)
    print("best model loaded")

    #test model
    test_model(model, test_loader, test_dataset, device, use_forces, save_path = save_path_test, num_denoising_steps=noiseadding_steps, num_samples=len(test_loader), postprocessing=True)


    #Inference application
    print("_____________________________________________")
    print("-----Inference application-----")
    # Clear GPU memory after testing
    del test_loader, test_dataset
    torch.cuda.empty_cache()

    # generate save path inference simulation
    save_path_application = os.path.join(save_path, f"{model_name}_{timestamp}_inference_application")
    os.makedirs(save_path_application, exist_ok=True)

    
    # Run inference on application data
    inference_simulation(
        model,
        application_loader,
        application_dataset,
        device,
        use_forces=use_forces,
        save_path=save_path_application,
        num_sequences=len(application_loader),
        num_denoising_steps=noiseadding_steps,
        postprocessing=True  
    )

if __name__ == "__main__":
    main()