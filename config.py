import os

# =====================================================================
DATASET_DIR_PATH = r"D:\2026\dataset_collections\Dataset-Zerotrust\hmog_dataset\public_dataset" 
# =====================================================================

BASE_DIR = os.getcwd()
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# NÂNG CẤP LÊN 50 USERS
NUM_USERS = 100          
WINDOW_SIZE = 128       
FEATURES = 6            
FL_ROUNDS = 5           # Chạy 5 vòng gộp Server
LOCAL_EPOCHS = 5        # Tăng lên 5 Epochs để AI học sâu hơn
LEARNING_RATE = 0.001
BATCH_SIZE = 256

def setup_env():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)