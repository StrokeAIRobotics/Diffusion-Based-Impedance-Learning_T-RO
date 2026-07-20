import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.ndimage import uniform_filter1d
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from scipy.ndimage import uniform_filter1d
import math
import torch.nn.functional as F
from utils import (add_noise, quaternion_multiply, quaternion_inverse, quaternion_loss, smooth_quaternions_slerp, quaternion_to_axis, smooth_stiffness, estimate_stiffness_per_window, quat_to_axis_angle_stiff)
from utils import set_seed


def train_model_diffusion(model, traindataloader, valdataloader,optimizer, criterion, device, num_epochs, noiseadding_steps, beta_start, 
                          beta_end, use_forces=False, noise_with_force=False, max_grad_norm=7.0, add_gaussian_noise=False, save_interval = 20, 
                          save_path = "save_checkpoints",early_stop_patience = 25):
    """
    Trains the NoisePredictor model using diffusion-based noisy trajectories.

    Args:
        model (nn.Module): The NoisePredictor model.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (optim.Optimizer): Optimizer for parameter updates.
        criterion (nn.Module): Loss function.
        device (torch.device): Device for training (CPU or GPU).
        num_epochs (int): Number of training epochs.
        noiseadding_steps (int): Number of steps to add noise.
        use_forces (bool): Whether to use forces as additional input to the model.

    Returns:
        list: List of average losses for each epoch.
    """
    train_epoch_losses = []
    val_epoch_losses = []

    os.makedirs(save_path, exist_ok=True)  # Ensure save directory exists
    best_val_loss = float('inf')  # Track best validation loss
    early_stopping_counter = 0  # Count epochs since last improvement


    # Initialize ReduceLROnPlateau
    lr_scheduler_patience = min(5, int(early_stop_patience * 0.32))  # Reduce LR after 1/3 of early stopping patience
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)#, verbose=True)



    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Use tqdm to create a progress bar
        for batch_idx, (pos_0, pos, q_0, q, force, moment, dx, omega, lambda_matrix, lambda_w_matrix) in enumerate(tqdm(traindataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)):
            
            # Move data to device
            clean_pos = pos_0.to(device)
            complete_noisy_pos = pos.to(device)
            clean_q = q_0.to(device)
            complete_noisy_q = q.to(device)
            force = force.to(device)
            moment = moment.to(device)

            # Dynamically add noise
            noisy_pos, noisy_q, noise_scale, t = add_noise(clean_pos, complete_noisy_pos, 
                                        clean_q, complete_noisy_q,
                                        force, moment,
                                        noiseadding_steps, beta_start, 
                                        beta_end, noise_with_force, add_gaussian_noise)
            
            # Compute the max noise (actual noise) based on the flag
            if noise_with_force:
                actual_noise_pos = force  # Use force as the max noise
                use_forces = False #then force should not be used as input
            else:
                actual_noise_pos = noisy_pos - clean_pos  # Default noise

            #Calc actual noise for q: actual_noise_q = noisy_q * clean_q^-1
            # Compute actual noise in quaternion space
            actual_noise_q = quaternion_multiply(noisy_q, quaternion_inverse(clean_q))

            optimizer.zero_grad()

            # Predict the noise from the noisy pos
            if use_forces:
                predicted_noise = model(noisy_pos, noisy_q, t, force, moment)
            
            else:
                predicted_noise = model(noisy_pos, noisy_q, t)
            
            loss = criterion(predicted_noise[:,:,0:3], actual_noise_pos) + 4*  quaternion_loss(predicted_noise[:,:,3:], actual_noise_q)

            loss = loss / torch.clamp(noise_scale, min=1e-6) * 10000  # Normalize loss by noise scale
            loss.backward()


            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(traindataloader)
        train_epoch_losses.append(avg_train_loss)


        #Validation after each epoch
        model.eval()  # Switch to evaluation mode
        with torch.no_grad():
            val_loss = validate_model_diffusion(
                model, valdataloader, criterion, device, noiseadding_steps, beta_start, beta_end, 
                use_forces, noise_with_force, add_gaussian_noise
            )
        val_epoch_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        last_lr = optimizer.param_groups[0]['lr']
        # Reduce LR if no improvement for `lr_scheduler_patience` epochs
        scheduler.step(val_loss)

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != last_lr:
            print(f"Epoch {epoch+1}: Learning Rate dropped to = {current_lr:.6e}")
  

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0  # Reset counter
            best_model_path = os.path.join(save_path, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at {best_model_path} after epoch {epoch+1}")
        else:
            early_stopping_counter += 1
            print(f"Early stopping patience: {early_stopping_counter}/{early_stop_patience}")

        # If no improvement for `patience` epochs, stop training
        if early_stopping_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs. Restoring best model.")
            model.load_state_dict(torch.load(best_model_path))  # Restore best model
            break  # Exit training loop

        # Save model every 'save_interval' epochs
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_path, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

    

    return train_epoch_losses, val_epoch_losses



def validate_model_diffusion(model, dataloader, criterion, device, max_noiseadding_steps, 
                             beta_start, beta_end, use_forces=False, noise_with_force=False, add_gaussian_noise=False):
    """
    Validates the NoisePredictor model on unseen data using diffusion-based noisy trajectories.

    Args:
        model (nn.Module): The NoisePredictor model.
        dataloader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device for validation (CPU or GPU).
        max_noiseadding_steps (int): Maximum number of steps to add noise.
        use_forces (bool): Whether to use forces as additional input to the model.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    total_loss = 0

    # Use tqdm to create a progress bar
    with torch.no_grad():
        for batch_idx, (pos_0, pos, q_0, q, force, moment, dx, omega, lambda_matrix, lambda_w_matrix) in enumerate(tqdm(dataloader, desc="Validating", leave=True)):
            clean_pos = pos_0.to(device)
            noisy_pos = pos.to(device)
            clean_q = q_0.to(device)
            noisy_q = q.to(device)
            force = force.to(device)
            moment = moment.to(device)

            # Dynamically add noise
            noisy_pos, noisy_q, noise_scale,  t = add_noise(clean_pos, noisy_pos, clean_q, noisy_q,
                                        force, moment,
                                        max_noiseadding_steps, beta_start, beta_end, noise_with_force, add_gaussian_noise)

            # Compute the max noise (actual noise) based on the flag
            if noise_with_force:
                actual_noise_pos = force  # Use force as the max noise
                use_forces = False #then force should not be used as input
            else:
                actual_noise_pos = noisy_pos - clean_pos  # Default: noise is the diff


            #Calc actual noise for q: actual_noise_q = noisy_q * clean_q^-1
            # Compute actual noise in quaternion space
            actual_noise_q = quaternion_multiply(noisy_q, quaternion_inverse(clean_q))

            # Predict the noise from the noisy pos
            if use_forces:
                predicted_noise = model(noisy_pos, noisy_q, t, force, moment) #with timestep

            else:
                predicted_noise = model(noisy_pos, noisy_q, t)

            # Calculate loss
            loss = criterion(predicted_noise[:,:,0:3], actual_noise_pos) + 4*  quaternion_loss(predicted_noise[:,:,3:], actual_noise_q)
            loss = loss / torch.clamp(noise_scale, min=1e-6) * 10000
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


def test_model(model, val_loader, val_dataset, device, use_forces, save_path, num_denoising_steps=1, num_samples=5, postprocessing = False):
    """
    Function to evaluate the model by predicting noise, performing iterative denoising,
    and visualizing the results.

    Args:
        model (torch.nn.Module): The trained noise predictor model.
        val_loader (DataLoader): DataLoader for the validation set.
        val_dataset (Dataset): Validation dataset (for denormalization).
        device (torch.device): Device (CPU/GPU).
        use_forces (bool): Whether forces are used as an input to the model.
        num_denoising_steps (int): Number of denoising steps.
        num_samples (int): Number of samples to visualize.
    """

    model.eval()  # Set model to evaluation mode

    # Convert validation dataset into a list (ensures correct indexing)
    val_data = list(val_loader.dataset)

    # Initialize lists for mean absolute differences
    mean_diffs_pos_x, mean_diffs_pos_y, mean_diffs_pos_z, overall_mean_diffs_pos = [], [], [], []
    mean_diffs_theta = []
    mean_diffs_axis_alpha = []
    mean_q_diffs = []
    with torch.no_grad():
        for sample_idx, (pos_0, pos, q_0, q, force, moment, dx, omega, lambda_matrix, lambda_w_matrix) in enumerate(tqdm(val_loader, desc="Testing", leave=True)):

            # Move data to the correct device
            clean_pos = pos_0.to(device)
            noisy_pos = pos.to(device)
            clean_q = q_0.to(device)
            noisy_q = q.to(device)
            force = force.to(device)
            moment = moment.to(device)

            # Start iterative denoising
            denoised_pos = noisy_pos.clone()
            denoised_q = noisy_q.clone()

            for i in range(num_denoising_steps):
                t = torch.tensor(i).long().to(denoised_pos.device)
                predicted_noise = model(denoised_pos, denoised_q, t, force, moment) if use_forces else model(denoised_pos, denoised_q, i)
                denoised_pos = denoised_pos - predicted_noise[:, :, 0:3]
                denoised_q = quaternion_multiply(denoised_q, quaternion_inverse(predicted_noise[:, :, 3:]))

            # Denormalize trajectories
            noisy_pos_np = val_dataset.denormalize(noisy_pos.detach().cpu(), "pos").numpy()
            clean_pos_np = val_dataset.denormalize(clean_pos.detach().cpu(), "pos_0").numpy()
            denoised_pos_np = val_dataset.denormalize(denoised_pos.detach().cpu(), "pos_0").numpy()

            noisy_q_np = val_dataset.denormalize(noisy_q.detach().cpu(), "q").numpy()
            clean_q_np = val_dataset.denormalize(clean_q.detach().cpu(), "q_0").numpy()
            denoised_q_np = val_dataset.denormalize(denoised_q.detach().cpu(), "q_0").numpy()

            force_np = val_dataset.denormalize(force.detach().cpu(), "force").numpy()
            moment_np = val_dataset.denormalize(moment.detach().cpu(), "force").numpy()

            if postprocessing:
                # Apply smoothing using a moving average filter
                window_size = 4
                denoised_pos_np = uniform_filter1d(denoised_pos_np, size=window_size, axis=1, mode='nearest')
                denoised_q_np = smooth_quaternions_slerp(torch.tensor(denoised_q_np), window_size=window_size, smoothing_factor=0.5).numpy()

                # Remove offsets
                offset_pos = np.mean(clean_pos_np[:, :1, :] - denoised_pos_np[:, :1, :], axis=1)
                denoised_pos_np += offset_pos[:, np.newaxis, :]

                q_offset = quaternion_multiply(clean_q[:, 0, :], quaternion_inverse(denoised_q[:, 0, :]))
                for t in range(denoised_q.shape[1]):
                    denoised_q[:, t, :] = quaternion_multiply(q_offset, denoised_q[:, t, :])

                denoised_q_np = denoised_q.detach().cpu().numpy()

            # Compute mean absolute differences in position
            mean_diff_x = np.mean(np.abs(clean_pos_np[:, :, 0] - denoised_pos_np[:, :, 0]))
            mean_diff_y = np.mean(np.abs(clean_pos_np[:, :, 1] - denoised_pos_np[:, :, 1]))
            mean_diff_z = np.mean(np.abs(clean_pos_np[:, :, 2] - denoised_pos_np[:, :, 2]))
            overall_mean_diff_pos = np.mean(np.abs(clean_pos_np - denoised_pos_np))

            mean_diffs_pos_x.append(mean_diff_x)
            mean_diffs_pos_y.append(mean_diff_y)
            mean_diffs_pos_z.append(mean_diff_z)
            overall_mean_diffs_pos.append(overall_mean_diff_pos)

            # Compute relative quaternion: q_rel = clean_q * inv(denoised_q)
            clean_q = clean_q / (clean_q.norm(dim=-1, keepdim=True) + 1e-8)
            denoised_q = denoised_q / (denoised_q.norm(dim=-1, keepdim=True) + 1e-8)

            q_rel = quaternion_multiply(clean_q, quaternion_inverse(denoised_q))
            q_rel = q_rel / q_rel.norm(dim=-1, keepdim=True).clamp(min=1e-8)

            # ---- Theta: rotation angle difference (aligned with loss function) ----
            # Flip predicted quaternion to same hemisphere as target
            dot = torch.sum(denoised_q * clean_q, dim=-1, keepdim=True)
            denoised_q_flipped = torch.where(dot < 0, -denoised_q, denoised_q)

            # Compute rotation angles
            theta_clean = 2 * torch.acos(torch.clamp(clean_q[..., 0], -1.0, 1.0))
            theta_denoised = 2 * torch.acos(torch.clamp(denoised_q_flipped[..., 0], -1.0, 1.0))

            # Angle difference with wrap-around
            theta_diff = torch.abs(theta_clean - theta_denoised)
            theta_diff = torch.minimum(theta_diff, 2 * torch.pi - theta_diff)

            # Degrees
            theta_diff_deg = torch.rad2deg(theta_diff)
            mean_theta_error = theta_diff_deg.mean().item()
            mean_diffs_theta.append(mean_theta_error)

            # ---- Alpha: axis alignment error ----
            clean_axis = quaternion_to_axis(clean_q.detach().cpu().numpy())
            denoised_axis = quaternion_to_axis(denoised_q.detach().cpu().numpy())
            clean_axis = torch.tensor(clean_axis, device=device, dtype=torch.float32)
            denoised_axis = torch.tensor(denoised_axis, device=device, dtype=torch.float32)

            clean_axis = clean_axis / (clean_axis.norm(dim=-1, keepdim=True) + 1e-8)
            denoised_axis = denoised_axis / (denoised_axis.norm(dim=-1, keepdim=True) + 1e-8)

            dot_product = torch.sum(clean_axis * denoised_axis, dim=-1)
            dot_product = torch.clamp(torch.abs(dot_product), -1.0, 1.0)
            alpha_error = torch.acos(dot_product)
            alpha_error_deg = torch.rad2deg(alpha_error)
            mean_alpha_error = alpha_error_deg.mean().item()
            mean_diffs_axis_alpha.append(mean_alpha_error)

            # ---- Quaternion difference (geodesic loss) ----
            clean_q_norm = clean_q / (clean_q.norm(dim=-1, keepdim=True) + 1e-8)
            denoised_q_norm = denoised_q / (denoised_q.norm(dim=-1, keepdim=True) + 1e-8)
            dot = torch.sum(clean_q_norm * denoised_q_norm, dim=-1, keepdim=True)
            denoised_q_norm = torch.where(dot < 0, -denoised_q_norm, denoised_q_norm)

            dot_product = torch.sum(clean_q_norm * denoised_q_norm, dim=-1)
            dot_product = torch.clamp(torch.abs(dot_product), 0.0, 1.0 - 1e-8)
            q_loss = torch.mean((1 - dot_product) ** 2).item()
            mean_q_diffs.append(q_loss)

           
        

    # Define the path for the results file
    results_file = os.path.join(save_path, "test_results.txt")

    # Prepare the results text
    results_text = (
        f"\nMean Absolute Differences in Testing"
        f"X-axis: {np.mean(mean_diffs_pos_x):.6f}\n"
        f"Y-axis: {np.mean(mean_diffs_pos_y):.6f}\n"
        f"Z-axis: {np.mean(mean_diffs_pos_z):.6f}\n"
        f"Overall: {np.mean(overall_mean_diffs_pos):.6f}\n\n"
        f"Theta: {np.mean(mean_diffs_theta):.6f}\n"
        f"Alpha: {np.mean(mean_diffs_axis_alpha):.6f}\n"
    )

    # Print results to console
    print(results_text)

    # Save results to file
    with open(results_file, "w") as file:
        file.write(results_text)

    print(f"Test results saved to {results_file}")
    plt.close('all')



def inference_simulation(model, application_loader, application_dataset, device, use_forces, save_path, num_sequences=100, num_denoising_steps=1, postprocessing=False):
    """
    Function to perform inference on the application dataset, reconstructing sequences sequentially.

    Args:
        model (torch.nn.Module): Trained noise predictor model.
        application_loader (DataLoader): DataLoader for the application dataset.
        application_dataset (Dataset): Application dataset (for denormalization).
        device (torch.device): Device (CPU/GPU).
        use_forces (bool): Whether forces are used as an input to the model.
        num_sequences (int): Number of sequences to process.
        num_denoising_steps (int): Number of denoising steps.
        postprocessing (bool): Whether to apply postprocessing smoothing.
    """
    
    model.eval()  # Set model to evaluation mode

    # Convert application dataset into a list for sequential access
    application_data = list(application_loader.dataset)
    
    # Ensure num_sequences does not exceed available data
    num_sequences = min(num_sequences, len(application_data))
    
    # Initialize lists for mean absolute differences
    mean_diffs_pos_x, mean_diffs_pos_y, mean_diffs_pos_z, overall_mean_diffs_pos = [], [], [], []
    mean_diffs_theta = []
    mean_diffs_axis_alpha = []
    #mean for stiffness
    mean_diffs_k_t = []
    mean_diffs_k_r = []
    mean_q_diffs = []
    theta_losses = []
    alpha_losses = []


    # Persistent storage for all data
    all_data = []
    with torch.no_grad():
        for seq_idx, (pos_0, pos, q_0, q, force, moment, dx, omega, lambda_matrix, lambda_w_matrix) in enumerate(tqdm(application_loader, desc="Inference simulation", leave=True)):
            # Fetch the sequence in order
            
            #Move data to the correct device
            clean_pos = pos_0.to(device)
            noisy_pos = pos.to(device)
            clean_q = q_0.to(device)
            noisy_q = q.to(device)
            force = force.to(device)
            moment = moment.to(device)
            

            # Start iterative denoising
            denoised_pos = noisy_pos.clone()
            denoised_q = noisy_q.clone()

            for i in range(num_denoising_steps):
                t = torch.tensor(i).long().to(denoised_pos.device)
                predicted_noise = model(denoised_pos, denoised_q, t, force, moment) if use_forces else model(denoised_pos, denoised_q,i)
                denoised_pos = denoised_pos - predicted_noise[:, :, 0:3]
                denoised_q = quaternion_multiply(denoised_q, quaternion_inverse(predicted_noise[:, :, 3:]))

            # Denormalize trajectories
            noisy_pos_np = application_dataset.denormalize(noisy_pos.detach().cpu(), "pos").numpy()
            clean_pos_np = application_dataset.denormalize(clean_pos.detach().cpu(), "pos_0").numpy()
            denoised_pos_np = application_dataset.denormalize(denoised_pos.detach().cpu(), "pos_0").numpy()

            force_np = application_dataset.denormalize(force.detach().cpu(), "force").numpy()
            moment_np = application_dataset.denormalize(moment.detach().cpu(), "moment").numpy()
            
            # Denormalize quaternions
            noisy_q_np = noisy_q.detach().cpu().numpy()
            clean_q_np = clean_q.detach().cpu().numpy()
            denoised_q_np = denoised_q.detach().cpu().numpy()

            if postprocessing:
                # Apply smoothing using a moving average filter
                window_size = 4
                denoised_pos_np = uniform_filter1d(denoised_pos_np, size=window_size, axis=1, mode='nearest')
                denoised_q_np = smooth_quaternions_slerp(torch.tensor(denoised_q_np), window_size=window_size, smoothing_factor=0.5).numpy()

                # Remove offsets
                offset_pos = np.mean(clean_pos_np[:, :1, :] - denoised_pos_np[:, :1, :], axis=1)
                denoised_pos_np += offset_pos[:, np.newaxis, :]

                q_offset = quaternion_multiply(clean_q[:, 0, :], quaternion_inverse(denoised_q[:, 0, :]))
                for t in range(denoised_q.shape[1]):
                    denoised_q[:, t, :] = quaternion_multiply(q_offset, denoised_q[:, t, :])

                denoised_q_np = denoised_q.detach().cpu().numpy()

            # Store all data in a structured format for txt file for matlab/robot
            T = clean_pos_np.shape[1]  # Sequence length
            time_array = np.arange(T) * 0.005  # Ensure time increments correctly


            for t in range(T):
                all_data.append([
                    int(seq_idx), float(time_array[t]),
                    *map(float, clean_pos_np[0, t, :]),  # Expands (x, y, z)
                    *map(float, denoised_pos_np[0, t, :]),
                    *map(float, noisy_pos_np[0, t, :]),
                    *map(float, clean_q_np[0, t, :]),  # Expands (qx, qy, qz, qw)
                    *map(float, denoised_q_np[0, t, :]),
                    *map(float, noisy_q_np[0, t, :]),
                    *map(float, force_np[0, t, :]),  # Expands (fx, fy, fz)
                    *map(float, moment_np[0, t, :]),  # Expands (mx, my, mz)

                ])

            # Compute mean absolute differences in position
            mean_diff_x = np.mean(np.abs(clean_pos_np[:, :, 0] - denoised_pos_np[:, :, 0]))
            mean_diff_y = np.mean(np.abs(clean_pos_np[:, :, 1] - denoised_pos_np[:, :, 1]))
            mean_diff_z = np.mean(np.abs(clean_pos_np[:, :, 2] - denoised_pos_np[:, :, 2]))
            overall_mean_diff_pos = np.mean(np.abs(clean_pos_np - denoised_pos_np))

            mean_diffs_pos_x.append(mean_diff_x)
            mean_diffs_pos_y.append(mean_diff_y)
            mean_diffs_pos_z.append(mean_diff_z)
            overall_mean_diffs_pos.append(overall_mean_diff_pos)

            clean_q = clean_q / (clean_q.norm(dim=-1, keepdim=True) + 1e-8)
            denoised_q = denoised_q / (denoised_q.norm(dim=-1, keepdim=True) + 1e-8)

            dot_val = torch.sum(clean_q[0, 0] * denoised_q[0, 0]).item()

            # Flip hemisphere
            dot = torch.sum(denoised_q * clean_q, dim=-1, keepdim=True)
            denoised_q_flipped = torch.where(dot < 0, -denoised_q, denoised_q)

            q_flipped = denoised_q[0, 0] if dot_val >= 0 else -denoised_q[0, 0]
            dot_flipped = torch.sum(clean_q[0, 0] * q_flipped).item()
            theta_flipped = 2 * math.acos(min(1.0, abs(dot_flipped)))

            # --- Theta calculation ---
            theta_clean = 2 * torch.acos(torch.clamp(clean_q[..., 0], -1.0, 1.0))
            theta_denoised = 2 * torch.acos(torch.clamp(denoised_q_flipped[..., 0], -1.0, 1.0))
            theta_diff = torch.abs(theta_clean - theta_denoised)
            theta_diff = torch.minimum(theta_diff, 2 * torch.pi - theta_diff)
            theta_diff_deg = torch.rad2deg(theta_diff)
            mean_theta_error = theta_diff_deg.mean().item()
            theta_loss = torch.mean(theta_diff ** 2).item()
            theta_losses.append(theta_loss)
            mean_diffs_theta.append(mean_theta_error)


            # --- Alpha ---
            clean_axis = quaternion_to_axis(clean_q.detach().cpu().numpy())
            denoised_axis = quaternion_to_axis(denoised_q.detach().cpu().numpy())
            clean_axis = torch.tensor(clean_axis, device=device, dtype=torch.float32)
            denoised_axis = torch.tensor(denoised_axis, device=device, dtype=torch.float32)
            clean_axis = clean_axis / (clean_axis.norm(dim=-1, keepdim=True) + 1e-8)
            denoised_axis = denoised_axis / (denoised_axis.norm(dim=-1, keepdim=True) + 1e-8)
            dot_product = torch.sum(clean_axis * denoised_axis, dim=-1)
            dot_product = torch.clamp(torch.abs(dot_product), -1.0, 1.0)
            alpha_error = torch.acos(dot_product)
            alpha_error_deg = torch.rad2deg(alpha_error)
            mean_alpha_error = alpha_error_deg.mean().item()
            mean_diffs_axis_alpha.append(mean_alpha_error)

            alpha_error = torch.acos(dot_product)
            alpha_loss = torch.mean(alpha_error ** 2).item()
            alpha_losses.append(alpha_loss)


            # --- Quaternion Loss (dot-based) ---
            clean_q_norm = clean_q / (clean_q.norm(dim=-1, keepdim=True) + 1e-8)
            denoised_q_norm = denoised_q / (denoised_q.norm(dim=-1, keepdim=True) + 1e-8)
            dot = torch.sum(clean_q_norm * denoised_q_norm, dim=-1, keepdim=True)
            denoised_q_norm = torch.where(dot < 0, -denoised_q_norm, denoised_q_norm)
            dot_product = torch.sum(clean_q_norm * denoised_q_norm, dim=-1)
            dot_product = torch.clamp(torch.abs(dot_product), 0.0, 1.0 - 1e-8)
            q_loss = torch.mean((1 - dot_product) ** 2).item()
            mean_q_diffs.append(q_loss)


    # Define the path for the results file
    results_file = os.path.join(save_path, "test_results.txt")

    # Prepare the results text
    results_text = (
        f"\nMean Absolute Differences in Inference Simulation:\n"
        f"X-axis: {np.mean(mean_diffs_pos_x):.6f}\n"
        f"Y-axis: {np.mean(mean_diffs_pos_y):.6f}\n"
        f"Z-axis: {np.mean(mean_diffs_pos_z):.6f}\n"
        f"Overall: {np.mean(overall_mean_diffs_pos):.6f}\n\n"
        f"Theta: {np.mean(mean_diffs_theta):.6f}\n"
        f"Alpha: {np.mean(mean_diffs_axis_alpha):.6f}\n"
            )

    print(results_text)

    # Save results to file
    with open(results_file, "w") as file:
        file.write(results_text)

    print(f"Test results saved to {results_file}")


    # Convert to DataFrame and save once at the end
    columns = [
        "Seq_Index", "Time",
        "Clean_X", "Clean_Y", "Clean_Z",
        "Denoised_X", "Denoised_Y", "Denoised_Z",
        "Noisy_X", "Noisy_Y", "Noisy_Z",
        "Clean_Q_W", "Clean_Q_X", "Clean_Q_Y", "Clean_Q_Z",
        "Denoised_Q_W", "Denoised_Q_X", "Denoised_Q_Y", "Denoised_Q_Z",
        "Noisy_Q_W", "Noisy_Q_X", "Noisy_Q_Y", "Noisy_Q_Z",
        "Force_X", "Force_Y", "Force_Z",
        "Moment_X", "Moment_Y", "Moment_Z",
    ]


    df = pd.DataFrame(all_data, columns=columns)
    output_file = os.path.join(save_path, "inference_results.txt")
    # Save in tab-separated format
    df.to_csv(output_file, sep='\t', index=False)

    print(f"Results saved to {output_file}")





def deployment(model, device,stats, pos, pos_0,q, q_0,force_model, force_stiffness, moment_model, moment_stiffness, lambda_matrix_np, dx_np,omega_np,lambda_w_matrix_np, clean_pos_before, clean_q_before, num_denoising_steps=20, K_t_prev = np.full(3, 650.0), K_r_prev = np.full(3, 100.0), iteration=0):
    #model eval
    model.eval()

    #____________data preprocessing____________
    #convert to torch tensor and match model input
    # At start of function (or outside if possible)
    to_tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    inputs = [pos, pos_0, q, q_0, force_model, force_stiffness, moment_model, moment_stiffness]
    pos, pos_0, q, q_0, force_model, force_stiffness, moment_model, moment_stiffness = [to_tensor(x).unsqueeze(0) for x in inputs]
    clean_q_before = to_tensor(clean_q_before)  
    lambda_matrix_np = lambda_matrix_np[np.newaxis, ...]
    lambda_w_matrix_np = lambda_w_matrix_np[np.newaxis, ...]

    #Normalize inputs using training stats
    min_pos = stats['min_pos_0']
    max_pos = stats['max_pos_0']
    min_force = stats['min_force']
    max_force = stats['max_force']
    min_moment = stats['min_moment']
    max_moment = stats['max_moment']
    # Normalize the input data
    pos = (pos - min_pos) / (max_pos - min_pos)
    pos_0 = (pos_0 - min_pos) / (max_pos - min_pos)
    force_model = (force_model - min_force) / (max_force - min_force)
    force_stiffness = (force_stiffness - min_force) / (max_force - min_force)
    moment_model = (moment_model - min_moment) / (max_moment - min_moment)
    moment_stiffness = (moment_stiffness - min_moment) / (max_moment - min_moment)


    # Start iterative denoising
    denoised_pos = pos.clone()
    denoised_q = q.clone()


    #Denoising with diffusion model
    t_all = torch.arange(num_denoising_steps, device=denoised_pos.device)
    with torch.no_grad():
        #Denoising
        for t in t_all:
            
            noise = model(denoised_pos, denoised_q, t, force_model, moment_model)
            denoised_pos -= noise[:, :, 0:3]
            denoised_q = quaternion_multiply(denoised_q, quaternion_inverse(noise[:, :, 3:]))

    

    #Denormalize
    denoised_pos = denoised_pos * (max_pos - min_pos) + min_pos
    pos_0 = pos_0 * (max_pos - min_pos) + min_pos
    noisy_pos = pos * (max_pos - min_pos) + min_pos
    force_model = force_model * (max_force - min_force) + min_force
    force_stiffness = force_stiffness * (max_force - min_force) + min_force
    moment_model = moment_model * (max_moment - min_moment) + min_moment
    moment_stiffness = moment_stiffness * (max_moment - min_moment) + min_moment
    # Now convert to numpy for postprocessing
    force_np_model = force_model[0].cpu().numpy()#force_np = np.squeeze(force_np, axis=0)  # (T, 3)
    force_np_stiffness = force_stiffness[0].cpu().numpy()#force_stiffness_np = np.squeeze(force_stiffness_np, axis=0)  # (T, 3)
    moment_np_model = moment_model[0].cpu().numpy()#moment_np = np.squeeze(moment_np, axis=0)  # (T, 3)
    moment_np_stiffness = moment_stiffness[0].cpu().numpy()#moment_stiffness_np = np.squeeze(moment_stiffness_np, axis=0)  # (T, 3)
    noisy_pos_np = noisy_pos[0].cpu().numpy()#noisy_pos_np = np.squeeze(noisy_pos_np, axis=0)  # (T, 3)
    denoised_pos_np = denoised_pos[0].cpu().numpy()#denoised_pos_np = np.squeeze(denoised_pos_np, axis=0)  # (T, 3)
    pos_0_np = pos_0[0].cpu().numpy()#command_pos_0_np = np.squeeze(command_pos_0_np, axis=0)  # (T, 3)
    noisy_q_np = q[0].cpu().numpy()#noisy_q_np = np.squeeze(noisy_q_np, axis=0)  # (T, 4)
    denoised_q_np = denoised_q[0].cpu().numpy()#denoised_q_np = np.squeeze(denoised_q_np, axis=0)  # (T, 4)
    dx_np = np.squeeze(dx_np)
    omega_np = np.squeeze(omega_np)


    #______postprocess data__________
    # Apply smoothing using a moving average filter
    window_size = 4
    denoised_pos = denoised_pos.transpose(1, 2)  # [1, 3, T]
    # Create a separate kernel for each channel (3 total)
    kernel = torch.ones(3, 1, window_size, device=denoised_pos.device) / window_size  # [3, 1, W]
    # Apply conv1d with groups=3 to treat each channel independently
    denoised_pos = F.conv1d(denoised_pos, kernel, padding=window_size//2, groups=3)
    # Convert back to original shape
    denoised_pos = denoised_pos.transpose(1, 2)  # back to [1, T, 3]
    denoised_q_np = smooth_quaternions_slerp(torch.tensor(denoised_q_np), window_size=window_size, smoothing_factor=0.5).numpy()

    # Remove offsets
    #position
    denoised_pos_for_offset = denoised_pos_np[np.newaxis, :, :]  # (1, T, 3)
    # compute offset
    offset_pos = np.mean(pos_0_np[0:1,:] - denoised_pos_for_offset[:, :1, :], axis=1)  # shape (1, 3)
    # squeeze offset (remove batch dimension)
    offset_pos = np.squeeze(offset_pos, axis=0)  # shape (3,)
    denoised_pos_np += offset_pos[np.newaxis, :]

    #quaternion
    q_offset = quaternion_multiply(q_0[:,0, :], quaternion_inverse(denoised_q[:, 0, :]))
    denoised_q = quaternion_multiply(q_offset.unsqueeze(1).expand(-1, denoised_q.shape[1], -1), denoised_q)
    denoised_q_np = denoised_q.detach().cpu().numpy()
    # Convert to numpy and squeeze
    denoised_q_np = np.squeeze(denoised_q_np)        # (T, 4)
    noisy_q_np = np.squeeze(noisy_q_np)              # (T, 4)
    clean_q_before = np.squeeze(clean_q_before)      # (T, 4)


    #_________compute stiffness___________
    T = clean_q_before.shape[0]  # Sequence length
    gamma = 1.0

    # Prepare data for stiffness estimation ---
    denoised_q_np_flat = denoised_q_np # (T, 4)
    omega = omega_np  # (T, 3)
    force = force_np_stiffness  # (T, 3)
    moment = moment_np_stiffness  # (T, 3)

    # Replace forces and moments with constant values, maintaining the same shape
    axis_denoised, theta_denoised = quat_to_axis_angle_stiff(denoised_q_np_flat) # transform to axis-angle representation
    theta_denoised = theta_denoised * 0.5 #for scaling of cpp code from data collection

    e_lin = denoised_pos_np - noisy_pos_np  # shape (T, 3)
    e_rot = axis_denoised * theta_denoised#[:, None]  # (T, 3)
    dt = 0.005  
    e_dot = np.gradient(e_lin, dt, axis=0)


    # Calculate the stiffness
    K_t_raw, K_r_raw, gamma = estimate_stiffness_per_window(
        axis_denoised[0:T],
        e_lin[0:T],
        e_dot[0:T],
        e_rot[0:T],
        omega[0:T],
        force[0:T],
        moment[0:T],
        gamma
        )
    
    # Smooth the stiffness values
    K_t = smooth_stiffness(K_t_raw, K_t_prev, iteration)
    K_r = smooth_stiffness(K_r_raw, K_r_prev, iteration)

    return K_r, K_t, denoised_pos_np, denoised_q_np