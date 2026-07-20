import time
import numpy as np
from multiprocessing import shared_memory
from avp_stream import VisionProStreamer


avp_ip = "your_AVP_ip_adress"  # Replace with your Apple Vision Pro IP address
s = VisionProStreamer(ip=avp_ip, record=True)
finger_closed = True

# Define shared memory parameters
SHM_NAME = "SharedMemory_AVP_new"
SHM_SIZE = 8 + 1* 16 * 8  + 8 # 8 bytes for Ready flag + 6 * 16 doubles (4x4 matrix) + 8 for booleans

# Create shared memory
try:
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
    print("Python created shared memory.")
except FileExistsError:
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
    print("Python attached to existing shared memory.")

# Define shared memory regions
version = np.ndarray((1,), dtype=np.int64, buffer=shm.buf[:8])  # Ready flag
data = np.ndarray((1, 17), dtype=np.float64, buffer=shm.buf[8:])  # 16 + 2

# Initialization
version[0] = -1  # Indicate that Python is initializing
print("Python initialized shared memory. Waiting for C++ to start...")

# Set flag to 0 to signal readiness
time.sleep(1)
version[0] = 0
print("Python is ready. Ready flag set to 0.")

try:
    while True:
        r = s.latest  # Get the latest data
        # check if finder should be closed
        if r['right_pinch_distance'] < 0.01:
            finger_closed = False 
        elif r['right_pinch_distance'] > 0.12:
            finger_closed = True

        # Prepare data for shared memory
        data_storage = np.concatenate([
            np.array(r['right_wrist'], dtype=np.float64).flatten(),
            np.array([float(finger_closed)], dtype=np.float64),
        ])

        # Write data to shared memory
        if version[0] == 0:
            data[0, :] = data_storage
            version[0] = 1  # Set Ready flag to 1

except KeyboardInterrupt:
    print("Python: Stopping.")

finally:
    shm.close()
    shm.unlink()
    print("Shared memory cleaned up.")
