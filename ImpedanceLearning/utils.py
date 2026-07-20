import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def add_noise(clean_pos, noisy_pos, clean_q, noisy_q, force, moment,
              max_noiseadding_steps, beta_start, beta_end, noise_with_force=False, add_gaussian_noise=False):
    """
    Adds noise to a clean 3D trajectory using a diffusion model schedule.

    Args:
        clean_pos (torch.Tensor): Clean trajectory, shape [seq_length, 3].
        noisy_pos (torch.Tensor): Noisy trajectory, shape [seq_length, 3].
        clean_q (torch.Tensor): Clean quaternion, shape [seq_length, 4].
        noisy_q (torch.Tensor): Noisy quaternion, shape [seq_length, 4].
        force (torch.Tensor): Force values, shape [seq_length, 3].
        moment (torch.Tensor): Moment values, shape [seq_length, 3].
        max_noiseadding_steps (int): Max number of noise-adding steps.
        beta_start (float): Initial noise scale.
        beta_end (float): Final noise scale.
        noise_with_force (bool): If True, use force as noise.
        add_gaussian_noise (bool): If True, add Gaussian noise.

    Returns:
        tuple: Noisy position, noisy quaternion, noise scale.
    """
    # Compute actual noise in a single operation
    actual_noise_pos = force if noise_with_force else noisy_pos - clean_pos

    if add_gaussian_noise:
        # Generate and normalize Gaussian noise for position in one step
        gaussian_noise_pos = torch.randn_like(actual_noise_pos)
        gaussian_noise_pos /= torch.norm(gaussian_noise_pos, dim=-1, keepdim=True).clamp(min=1e-6)
        gaussian_noise_pos *= torch.norm(actual_noise_pos, dim=-1, keepdim=True)
        actual_noise_pos += gaussian_noise_pos  # In-place addition

        # Generate random axis-angle perturbation for quaternion noise
        random_axis = torch.randn_like(noisy_q[..., 1:])
        random_axis /= torch.norm(random_axis, dim=-1, keepdim=True).clamp(min=1e-6)
        random_angle = torch.randn(noisy_q.shape[:-1], device=noisy_q.device) * 0.1

        # Compute quaternion noise scaling directly
        dot_product = torch.sum(clean_q * noisy_q, dim=-1, keepdim=True).clamp(-1.0, 1.0)
        theta = 2 * torch.acos(dot_product.abs())
        scaling_factor = (theta / torch.pi).unsqueeze(-1)
        scaled_angle = random_angle * scaling_factor

        # Convert axis-angle perturbation to a quaternion in a single operation
        sin_half_angle, cos_half_angle = torch.sin(scaled_angle / 2), torch.cos(scaled_angle / 2)
        gaussian_noise_q = torch.cat([cos_half_angle, sin_half_angle * random_axis], dim=-1)

        # Apply Gaussian noise to quaternion using in-place multiplication
        noisy_q = quaternion_multiply(gaussian_noise_q, noisy_q)

    # Efficient random step selection
    noiseadding_steps = torch.randint(1, max_noiseadding_steps + 1, ())

    # Vectorized noise schedule computation
    beta_values = torch.linspace(beta_start, beta_end, noiseadding_steps, device=clean_pos.device)
    alpha_bar = torch.cumprod(1 - beta_values, dim=0)

    # Select a random timestep t efficiently
    t = torch.randint(0, noiseadding_steps, ())

    sqrt_alpha_bar_t = torch.sqrt(alpha_bar[t])

    # Compute noisy position in one operation
    noisy_pos_output = sqrt_alpha_bar_t * clean_pos + torch.sqrt(1 - alpha_bar[t]) * actual_noise_pos

    # SLERP interpolation for quaternion
    noisy_q_output = slerp(clean_q, noisy_q, sqrt_alpha_bar_t)
    # Compute noise scale directly
    noise_scale = 1 / sqrt_alpha_bar_t
    return noisy_pos_output, noisy_q_output, noise_scale, t



def quaternion_multiply(q1, q2):
    """Computes the Hamilton product of two quaternions q1 and q2, and normalizes the result."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    # Hamilton product
    q = torch.stack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ), dim=-1)

    # Normalize the output to ensure it's a unit quaternion
    q = q / torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-6)

    return q


def quaternion_inverse(q, eps=1e-12):
    """Computes the inverse of a quaternion q = (w, x, y, z).
       Handles zero quaternions safely by avoiding division by zero.
    """
    norm_sq = torch.sum(q**2, dim=-1, keepdim=True)  # Compute |q|^2
    q_conjugate = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

    # Avoid division by zero by clamping the norm
    norm_sq_safe = torch.clamp(norm_sq, min=eps)

    return q_conjugate / norm_sq_safe




def quaternion_loss(pred_q, target_q, lambda_unit=0.3):
    """
    Computes quaternion loss with geodesic distance, theta difference (angle wrap-around fixed), and alpha loss (stable).

    Args:
        pred_q (torch.Tensor): Predicted quaternion (..., 4).
        target_q (torch.Tensor): Ground truth quaternion (..., 4).
        lambda_unit (float): Weight for unit norm constraint.

    Returns:
        torch.Tensor: Combined loss value.
    """
    eps = 1e-8  # Small constant for numerical stability

    # Normalize quaternions
    target_q = F.normalize(target_q, dim=-1)
    pred_q = F.normalize(pred_q, dim=-1)

    # Ensure correct quaternion orientation
    dot_product = torch.sum(pred_q * target_q, dim=-1, keepdim=True)
    pred_q = torch.where(dot_product < 0, -pred_q, pred_q)  # Flip if necessary
    dot_product = dot_product.abs().squeeze(-1)  # Ensure positive values

    # 1. Geodesic quaternion loss
    loss_q = torch.mean((1 - dot_product) ** 2)  

    angle_diff_rad = 2 * torch.acos(dot_product.clamp(0, 1 - 1e-6))
    loss_q += torch.mean(angle_diff_rad ** 2)

    # 2. Alpha (axis difference) loss
    axis_pred = F.normalize(pred_q[..., 1:], dim=-1)
    axis_target = F.normalize(target_q[..., 1:], dim=-1)

    dot_product_axis = torch.sum(axis_pred * axis_target, dim=-1).clamp(-0.999999, 0.999999)
    alpha_error = torch.atan2(torch.sqrt(1 - dot_product_axis**2 + eps), dot_product_axis)
    loss_alpha = torch.mean(alpha_error ** 2)

    # 3. Unit norm constraint
    unit_loss = torch.mean((torch.norm(pred_q, dim=-1) - 1) ** 2)

    #4. theta loss (angle of quat)
    theta_rad = 2 * torch.acos(dot_product.clamp(0, 1 - 1e-6))  # from flipped + normalized quaternions
    loss_theta = torch.mean(theta_rad ** 2)

    # Final weighted loss - with theta loss the results getting worse?
    return 5* loss_theta + loss_q + unit_loss



def smooth_quaternions_slerp(q_series, window_size=5, smoothing_factor=0.5):
    """
    Smooths a quaternion time series using SLERP interpolation over fixed windows.

    Args:
        q_series (torch.Tensor): Quaternion time series of shape [T, 4] (T = timesteps).
        window_size (int): Number of timesteps in each smoothing window.
        smoothing_factor (float): Weight factor (0.0 = no smoothing, 1.0 = full smoothing).

    Returns:
        torch.Tensor: Smoothed quaternion time series of shape [T, 4].
    """
    T = q_series.shape[0]

    for start in range(0, T - window_size + 1, window_size):  # Process in chunks
        end = min(start + window_size, T)

        # Compute SLERP between previous and next quaternion for the whole window at once
        q_prev = q_series[start:end-2]  # All except last two
        q_next = q_series[start+2:end]  # All except first two

        q_series[start+1:end-1] = slerp(q_prev, q_next, smoothing_factor)  # Apply SLERP

    return q_series



def slerp(q0, q1, t):
    """Performs Spherical Linear Interpolation (SLERP) between two quaternions."""
    # Normalize input quaternions
    q0 = q0 / torch.norm(q0, dim=-1, keepdim=True)
    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)

    # Compute dot product and correct for long path
    dot_product = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot_product < 0.0, -q1, q1)
    dot_product = torch.abs(dot_product).clamp(-1.0, 1.0)

    # Compute angle and its sine
    theta = torch.acos(dot_product)
    sin_theta = torch.sin(theta)

    # Ensure t is broadcastable
    # Convert scalar t to tensor with proper broadcasting shape
    if isinstance(t, float):
        t = torch.tensor(t, dtype=q0.dtype, device=q0.device)
    if t.dim() == 0:
        t = t.unsqueeze(0).expand_as(dot_product)


    # Handle small angles with linear interpolation
    near_zero = sin_theta.abs() < 1e-6
    s1 = torch.where(near_zero, 1.0 - t, torch.sin((1 - t) * theta) / sin_theta)
    s2 = torch.where(near_zero, t, torch.sin(t * theta) / sin_theta)

    # Interpolate and normalize
    q_interp = s1 * q0 + s2 * q1
    return q_interp / torch.norm(q_interp, dim=-1, keepdim=True)



# Function to extract rotation axis (u) from a quaternion
def quaternion_to_axis(q):
    """
    Extracts the rotation axis from a unit quaternion.

    Args:
        q (numpy array): Quaternion array of shape (sequence_length, 4),
                         where each quaternion is (w, x, y, z).

    Returns:
        numpy array: Rotation axis (unit vector), shape (sequence_length, 3).
    """
    w = q[:, 0]  # Extract w component
    xyz = q[:, 1:]  # Extract (x, y, z) components

    # Compute sin(theta/2) with numerical stability
    sin_half_theta = np.sqrt(np.clip(1 - w**2, 0.0, None))  # Use clip for stability

    # Avoid division by zero using np.where
    u = np.where(sin_half_theta[:, np.newaxis] > 1e-6, xyz / sin_half_theta[:, np.newaxis], np.zeros_like(xyz))

    return u


def quat_to_axis_angle_stiff(q):
                """
                Converts a sequence of quaternions (N, 4) to axis (N, 3) and angle (N, 1) arrays.
                Ensures the scalar part (w) is always non-negative for consistency.
                Args:
                    q: numpy array of shape (N, 4) in (w, x, y, z) format.
                Returns:
                    axis: numpy array of shape (N, 3)
                    angle: numpy array of shape (N, 1)
                """
                q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
                # Flip quaternion if w < 0 to ensure positive hemisphere
                flip_mask = q[:, 0] < 0
                q[flip_mask] = -q[flip_mask]

                w = q[:, 0:1]  # shape (N, 1)
                xyz = q[:, 1:]  # shape (N, 3)
                angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))  # shape (N, 1)
                sin_half_angle = np.sqrt(1.0 - np.clip(w**2, 0.0, 1.0))  # shape (N, 1)

                axis = np.zeros_like(xyz)
                mask = (sin_half_angle > 1e-8).flatten()
                axis[mask] = xyz[mask] / sin_half_angle[mask]
                axis[~mask] = np.array([1.0, 0.0, 0.0])  # arbitrary axis if angle ≈ 0

                return axis, angle



def estimate_stiffness_per_window(u_0, e_lin_win, e_dot_win,
                                                 e_rot_win, omega_win,
                                                 F_win, M_win, gamma):
    """
    Estimate stiffness for a given window of data, scaling drop per axis by relative motion.
    Args:
        e_lin_win (numpy array): Linear error for the window.
        e_dot_win (numpy array): Linear velocity for the window.
        e_rot_win (numpy array): Rotational error for the window.
        omega_win (numpy array): Angular velocity for the window.
        F_win (numpy array): Force for the window.
        M_win (numpy array): Moment for the window.
        gamma (float): Damping factor. 
    Returns:
        K_t (numpy array): Translational stiffness.
        K_r (numpy array): Rotational stiffness.
        gamma (float): Damping factor. 
    """

    K_t = np.zeros(3)
    K_r = np.zeros(3)
    K_t_max = 800.0                             
    K_r_max = 150.0                             
    epsilon = 1e-6
    force_thresh = 0.2
    moment_thresh = 1.0
    aggression_factor_trans = 10
    aggression_factor_rot = 2               

    # Compute per-axis norms (magnitudes)
    trans_axis_norms = np.linalg.norm(e_lin_win, axis=0)  # [vx, vy, vz] magnitudes
    rot_axis_norms = np.linalg.norm(u_0, axis=0)    # [wx, wy, wz] magnitudes

    # Compute total sum across axes (avoid raw magnitude influence)
    sum_trans = np.sum(trans_axis_norms) + epsilon
    sum_rot = np.sum(rot_axis_norms) + epsilon

    # Compute relative importance per axis (unitless, normalized)
    rel_trans_importance = trans_axis_norms / sum_trans  # fraction per axis
    rel_rot_importance = rot_axis_norms / sum_rot        # fraction per axis

    for i in range(3):
        # Translational stiffness
        e_i = e_lin_win[:, i] - gamma * e_dot_win[:, i]
        f_i = F_win[:, i]
        if np.max(np.abs(f_i)) < force_thresh:
            k_t_i = K_t_max
        else:
            k_drop = np.abs(np.dot(f_i, e_i)) / (np.dot(e_i, e_i) + epsilon)
            # Scale drop: more motion → smaller drop
            scale_factor = 1 - rel_trans_importance[i]  # invert to drop less on active axes
            k_drop *= aggression_factor_trans * scale_factor            # WAS COMMENTED OUT!!!
            k_t_i = np.clip(K_t_max - k_drop, 0.0, K_t_max)
        K_t[i] = k_t_i

        # Rotational stiffness
        r_i = e_rot_win[:, i] - gamma * omega_win[:, i]
        m_i = M_win[:, i]
        if np.max(np.abs(m_i)) < moment_thresh:
            k_r_i = K_r_max
        else:
            k_drop_r = np.abs(np.dot(m_i, r_i)) / (np.dot(r_i, r_i) + epsilon)
            scale_factor_r = 1 - rel_rot_importance[i]
            k_drop_r *= aggression_factor_rot * scale_factor_r            # WAS COMMENTED OUT!!!
            k_r_i = np.clip(K_r_max - k_drop_r, 0.0, K_r_max)
        K_r[i] = k_r_i

    return K_t, K_r, gamma

def smooth_stiffness(K_raw, K_prev, iteration, alpha=0.15, max_step=30.0, warmup_steps=40):
    """
    Smooth the stiffness values using exponential smoothing.
    Args:
        K_raw (numpy array): Raw stiffness values.
        K_prev (numpy array): Previous stiffness values (should be initialized to K_raw[0]).
        iteration (int): Current time step.
        alpha (float): Smoothing factor.
        max_step (float): Maximum step size for smoothing.
        warmup_steps (int): Number of steps to bypass clipping to prevent initial drop.
    Returns:
        K_smooth (numpy array): Smoothed stiffness values.
    """

    if iteration < warmup_steps:
            # No clipping during warmup

            return K_raw
    else:
        K_smooth = np.zeros(3)
        for i in range(3):
            k_exp = alpha * K_raw[i] + (1 - alpha) * K_prev[i]
    
            # Apply clipping to prevent large jumps
            k_lim = np.clip(k_exp, K_prev[i] - max_step, K_prev[i] + max_step)
            K_smooth[i] = k_lim

    return K_smooth