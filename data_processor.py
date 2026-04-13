import os
import zipfile
import pandas as pd
import numpy as np
import re
import gc
from sklearn.model_selection import train_test_split
import config

def sliding_window_fast(data, window_size, overlap=0.5):
     
    step = int(window_size * (1 - overlap))
    if step <= 0: step = 1
    shape = ((data.shape[0] - window_size) // step + 1, window_size, data.shape[1])
    strides = (data.strides[0] * step, data.strides[0], data.strides[1])
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

def prepare_data():
    print(f"\n[STEP 1] NẠP DATASET HMOG TỪ THƯ MỤC: {config.DATASET_DIR_PATH}")
    if os.path.exists(os.path.join(config.PROCESSED_DIR, 'y_val_0.npy')):
        print("[INFO] Dữ liệu đã chia 70/15/15 tồn tại. Bỏ qua trích xuất.")
        return

    if not os.path.isdir(config.DATASET_DIR_PATH):
        raise FileNotFoundError(f"[ERROR] Không tìm thấy thư mục: {config.DATASET_DIR_PATH}")

    zip_files = sorted([f for f in os.listdir(config.DATASET_DIR_PATH) if f.endswith('.zip')])[:config.NUM_USERS]
    users_data = {}
    print(f"[INFO] Bắt đầu trích xuất {len(zip_files)} Users (Bật chế độ SIÊU TỐI ƯU CHỐNG TRÀN RAM)...")

    # Giới hạn số cửa sổ cần thiết: Tối đa 5000 mẫu/user
    MAX_SAMPLES = 5000
     
    REQUIRED_ROWS = int(MAX_SAMPLES * (config.WINDOW_SIZE * 0.5) + config.WINDOW_SIZE)

    for u_idx, zip_name in enumerate(zip_files):
        zip_path = os.path.join(config.DATASET_DIR_PATH, zip_name)
        user_features = []
        total_rows = 0
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                acc_files = [f for f in z.namelist() if 'accelerometer.csv' in f.lower() and '__MACOSX' not in f]
                gyr_files = [f for f in z.namelist() if 'gyroscope.csv' in f.lower() and '__MACOSX' not in f]
                sessions = set([re.search(r'session_(\d+)', f.lower()).group(1) for f in acc_files if re.search(r'session_(\d+)', f.lower())])
                    
                for session_id in sessions:
                    if total_rows > REQUIRED_ROWS:
                        break # [EARLY STOPPING] Cắt tỉa sớm, ngưng đọc file nếu đã đủ dữ liệu
                        
                    s_acc = [f for f in acc_files if f'session_{session_id}' in f.lower()]
                    s_gyr = [f for f in gyr_files if f'session_{session_id}' in f.lower()]
                    if not s_acc or not s_gyr: continue
                    try:
                        df_acc = pd.read_csv(z.open(s_acc[0]))
                        df_gyr = pd.read_csv(z.open(s_gyr[0]))
                        
                        # Đọc thẳng vào float32 để giảm tải I/O bộ nhớ
                        acc_vals = df_acc.iloc[:, -3:].values.astype(np.float32)
                        gyr_vals = df_gyr.iloc[:, -3:].values.astype(np.float32)
                        
                        min_len = min(len(acc_vals), len(gyr_vals))
                        if min_len < config.WINDOW_SIZE: continue
                        user_features.append(np.hstack((acc_vals[:min_len], gyr_vals[:min_len])))
                        total_rows += min_len
                    except: continue
        except: continue
            
        if user_features:
            merged = np.vstack(user_features)
            del user_features
            
            # [FIX LỖI CHÍ MẠNG ARRAYMEMORYERROR CỦA SKLEARN]
            # Tự động Standardize bằng numpy để duy trì float32, chống upcast lên float64
            mean = np.mean(merged, axis=0, dtype=np.float32)
            std = np.std(merged, axis=0, dtype=np.float32)
            merged = (merged - mean) / (std + 1e-8)
            
            # Khởi tạo Cửa sổ trượt
            windows = sliding_window_fast(merged, config.WINDOW_SIZE)
            
            # Rút trích đúng 5000 mẫu đẹp nhất và copy ra mảng chuẩn để giải phóng mảng gốc
            num_windows = len(windows)
            if num_windows > MAX_SAMPLES:
                idx = np.random.choice(num_windows, MAX_SAMPLES, replace=False)
                sampled_windows = windows[idx].copy()
            else:
                sampled_windows = windows.copy()
                
            users_data[u_idx] = sampled_windows
            print(f"  -> User {zip_name.split('.')[0]}: Xử lý xong {len(sampled_windows)} mẫu (Tốc độ Cao).")
            del merged, windows
            gc.collect()

    print("\n[INFO] Đang xây dựng tập Train (70%) - Val (15%) - Test (15%) và Trộn Kẻ gian...")
    valid_users = list(users_data.keys())
    for u_idx in valid_users:
        genuine = users_data[u_idx]
        if len(genuine) == 0: continue
        
        # SMART IMPOSTER SAMPLING: Chống phình RAM bằng cách lấy đều mỗi người một ít
        other_users = [i for i in valid_users if i != u_idx]
        if not other_users: continue
        
        samples_per_other = max(1, len(genuine) // len(other_users)) + 1
        imposter_list = []
        for i in other_users:
            other_data = users_data[i]
            if len(other_data) > 0:
                idx = np.random.choice(len(other_data), min(len(other_data), samples_per_other), replace=False)
                imposter_list.append(other_data[idx])
                
        imposter = np.vstack(imposter_list)
        np.random.shuffle(imposter)
        imposter = imposter[:len(genuine)] # Cân bằng tuyệt đối nhãn 1:1
        
        min_samples = min(len(genuine), len(imposter))
        if min_samples == 0: continue
        
        X = np.vstack((genuine[:min_samples], imposter[:min_samples]))
        y = np.vstack((np.ones((min_samples, 1)), np.zeros((min_samples, 1)))).astype(np.float32)
        
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
        
        np.save(os.path.join(config.PROCESSED_DIR, f'X_train_{u_idx}.npy'), X_train)
        np.save(os.path.join(config.PROCESSED_DIR, f'y_train_{u_idx}.npy'), y_train)
        np.save(os.path.join(config.PROCESSED_DIR, f'X_val_{u_idx}.npy'), X_val)
        np.save(os.path.join(config.PROCESSED_DIR, f'y_val_{u_idx}.npy'), y_val)
        np.save(os.path.join(config.PROCESSED_DIR, f'X_test_{u_idx}.npy'), X_test)
        np.save(os.path.join(config.PROCESSED_DIR, f'y_test_{u_idx}.npy'), y_test)
        
    print("[OK] ĐÃ XỬ LÝ XONG DỮ LIỆU THẬT 100 USERS THEO TỶ LỆ 70/15/15!")