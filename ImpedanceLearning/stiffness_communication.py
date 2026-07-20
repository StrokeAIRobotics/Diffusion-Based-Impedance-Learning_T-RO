import time
import numpy as np
from multiprocessing import shared_memory
from avp_stream import VisionProStreamer
import torch
from models import NoisePredictorTransformerWithCrossAttentionTime
from train_val_test import deployment

#FIRST SHARED MEMORY
#######################################################################
# Define shared memory parameters
SHM_NAME = "SharedMemory_1"
SHM_SIZE = 8 + 8*6

# Create shared memory
try:
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
    print("Python created shared memory.")
except FileExistsError:
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
    print("Python attached to existing shared memory.")

# Define shared memory regions
version = np.ndarray((1,), dtype=np.int64, buffer=shm.buf[:8])  # Ready flag
data = np.ndarray((1, 6), dtype=np.float64, buffer=shm.buf[8:])  # 16 + 2

# Initialization
print("Python initialized shared memory. Waiting for C++ to start...")

# Set flag to 0 to signal readiness
time.sleep(1)
version[0] = -1
print("Python is ready. Ready flag set to 0.")


#SECOND SHARED MEMORY
#######################################################################
#Define second shared memory parameters
SHM_NAME_2 = "SharedMemory_2"
SHM_SIZE_2 = 50 *8 *16 + 8 # Change that

# Create second shared memory
try:
    shm_2 = shared_memory.SharedMemory(name=SHM_NAME_2, create=True, size=SHM_SIZE_2)
    print("Python created second shared memory.")
except FileExistsError:
    shm_2 = shared_memory.SharedMemory(name=SHM_NAME_2, create=False)
    print("Python attached to existing second shared memory.")

# Define second shared memory regions
version_2 = np.ndarray((1,), dtype=np.int64, buffer=shm_2.buf[:8])  # Ready flag
data_2 = np.ndarray((16, 50), dtype=np.float64, buffer=shm_2.buf[8:])  # Change that

# Initialize second shared memory
print("Python initialized second shared memory. Waiting for C++ to start...")

# Set flag to 0 to signal readiness
time.sleep(1)
version_2[0] = -1
print("Python is ready for second shared memory. Ready flag set to 0.")


#information for stiffness adaption
seq_length = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_denoising_steps = 20

#Parameter and model initialization
#####################################################################
# Load stats
stats = torch.load("/home/newman_lab/Desktop/Model/stats.pt")
# Move each tensor in stats to the device
stats = {k: v.to(device) for k, v in stats.items()}
# Load model
model = NoisePredictorTransformerWithCrossAttentionTime(seq_length, hidden_dim = 512, use_forces=True).to(device) # You need to initialize model
model.load_state_dict(torch.load("/home/newman_lab/Desktop/Model/best_model.pth"))
stiffness_rot = np.full(3, 150.0)
stiffness_trans = np.full(3, 650.0)
num_denoising_steps = 20

#Initialization of first posiition and quaternion and parameters
########################################################################
first_pos = np.array([0.61851096, -0.04580012, -0.05646933])
first_q = np.array([0.6878653, -0.14780605, 0.62315786, -0.34156842])
clean_pos_before = np.tile(first_pos, (1, 16, 1))  
clean_q_before = np.tile(first_q, (1, 16, 1))  
# Store all average forces and translational stiffness values
all_avg_forces = []
all_stiffness_trans = []
i=0


try:
    while True:

        #read data from shared memory
        if version_2[0] == 0 or version_2[0] == -1:

            # Extract structured values from shared memory (16, 37)
            pos = data_2[:, 0:3]                           # Position (x, y, z)
            pos_0 = data_2[:,3:6]                          # Position (x, y, z) - command
            q = data_2[:, 6:10]                             # Quaternion (x, y, z, w)
            q0 = data_2[: ,10:14]                            # Quaternion (x, y, z, w) - command
            force = data_2[:, 14:17]                       # Force (Fx, Fy, Fz)
            force_model = data_2[:, 17:20]                        # Force (Fx, Fy, Fz)
            # Replace forces and moments with constant values, maintaining the same shape
            moment = data_2[:, 20:23]                      # Moment (Mx, My, Mz)
            moment_model = data_2[:, 23:26]                   # Moment (Mx, My, Mz)
            lambda_matrix = data_2[:, 26:35].reshape(-1, 3, 3)  # Lambda_v (3x3 matrix)
            dx = data_2[:, 35:38]                          # Translational velocity
            omega = data_2[:, 38:41]                  # Angular velocity
            lambda_w_matrix = data_2[:, 41:50].reshape(-1, 3, 3)  # Lambda_w (3x3 matrix)

            start_time = time.time()
            #calculate stiffness here
            stiffness_rot, stiffness_trans, clean_pos_before_model, clean_q_before_model = deployment(
                model=model,
                device=device,
                stats=stats,
                pos=pos,
                pos_0=pos_0,
                q=q,
                q_0=q0,
                force_model=force_model,
                force_stiffness=force,
                moment_model=moment_model,
                moment_stiffness=moment,
                lambda_matrix_np=lambda_matrix,
                dx_np=dx,
                omega_np=omega,
                lambda_w_matrix_np=lambda_w_matrix,
                clean_pos_before=clean_pos_before,
                clean_q_before=clean_q_before,
                num_denoising_steps=num_denoising_steps,
                K_t_prev = stiffness_trans, 
                K_r_prev = stiffness_rot,
                iteration = i
            )

            end_time = time.time()

            if i<35:
                stiffness_rot = np.full(3, 650.0)
                stiffness_trans = np.full(3, 100.0)
            if i ==35:
                print("updated stiffness from now on")
            i+=1

            # Reshape clean_pos_before and clean_q_before to match the expected shape
            # Only update clean_q_before if clean_q_before_model is not all zeros (or near-zero)
            if not np.allclose(clean_q_before_model, 0.0, atol=1e-8):
                clean_q_before = clean_q_before_model.reshape(1, 16, 4)
                clean_pos_before = clean_pos_before_model.reshape(1, 16, 3)

            # Store data for logging
            all_avg_forces.append(np.mean(moment, axis=0))
            all_stiffness_trans.append(stiffness_rot.copy())

            version_2[0] = 1 
     
        if version[0] == 0 or version[0] == -1:
            # Prepare data for shared memory
            data_storage = np.concatenate([stiffness_trans, stiffness_rot]).astype(np.float64)
            data[0, :] = data_storage  # Write the values into the shared memory
            version[0] = 1  # Set Ready flag to 1
            
except KeyboardInterrupt:
    print("Python: Stopping.")


finally:
    shm.close()
    shm.unlink()
    shm_2.close()
    shm_2.unlink()
    print("Shared memory cleaned up.")
