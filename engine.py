import os
# [QUAN TRỌNG] Tắt tính năng biên dịch nội bộ của PyTorch để tránh lỗi MemoryError trên Windows
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCH_DYNAMO_DISABLE'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import gc
import traceback
import config
from models import BaseModel, ProposedModel
from evaluator import export_all

def load_data(user_idx, split='train'):
    X = np.load(os.path.join(config.PROCESSED_DIR, f'X_{split}_{user_idx}.npy'))
    y = np.load(os.path.join(config.PROCESSED_DIR, f'y_{split}_{user_idx}.npy'))
    return torch.tensor(X), torch.tensor(y)

def load_labels_only(user_idx, split='test'):
     
    y = np.load(os.path.join(config.PROCESSED_DIR, f'y_{split}_{user_idx}.npy'))
    return y.flatten()

def train_model(model, X, y, epochs, is_proposed=False, device='cpu'):
    if len(X) <= 1: return model
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCELoss()
    
    dl_drop = True if len(X) > config.BATCH_SIZE else False
    loader = DataLoader(TensorDataset(X, y), batch_size=config.BATCH_SIZE, shuffle=True, drop_last=dl_drop)
    
    for _ in range(epochs):
        for bx, by in loader:
            if len(bx) <= 1: continue
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            if is_proposed:
                noise = torch.std(bx, dim=[1,2], unbiased=False, keepdim=True)
                outputs = model(bx, noise)
            else: 
                outputs = model(bx)
            loss = criterion(outputs, by)
            loss.backward()
            optimizer.step()
    return model

def eval_model(model, X_test, is_proposed=False, device='cpu'):
    if len(X_test) == 0: return np.array([])
    model.eval()
    loader = DataLoader(TensorDataset(X_test), batch_size=512, shuffle=False)
    preds = []
    with torch.no_grad():
        for bx in loader:
            bx = bx[0].to(device)
            if is_proposed:
                noise = torch.std(bx, dim=[1,2], unbiased=False, keepdim=True)
                outputs = model(bx, noise)
            else: 
                outputs = model(bx)
            preds.append(outputs.cpu().numpy())
    if not preds: return np.array([])
    return np.concatenate(preds).flatten()

def run_baselines():
    try:
        print(f"\n[STEP 2] TIẾN HÀNH TRAINING VỚI TẬP TRAIN (70%) - VAL (15%) - TEST (15%) TRÊN {config.NUM_USERS} USERS...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        valid_users = [u for u in range(config.NUM_USERS) if os.path.exists(os.path.join(config.PROCESSED_DIR, f'y_test_{u}.npy'))]
        if not valid_users:
            print("[LỖI] Không tìm thấy dữ liệu. Hãy chạy lại Step 1.")
            return
            
        results = {}
        
        # [GIẢI QUYẾT MEMORY ERROR] Chỉ nạp nhãn, không nạp mảng X
        y_test_all = [load_labels_only(u, 'test') for u in valid_users]
        y_true_all = np.concatenate(y_test_all)
        del y_test_all
        gc.collect()
        
        # -------------------------------------------------------------
        # 1. Centralized Model
        # -------------------------------------------------------------
        print("\n -> 1. Đang train Centralized CNN-LSTM...")
        cent_model = BaseModel(config.FEATURES).to(device)
        optimizer_cent = optim.Adam(cent_model.parameters(), lr=config.LEARNING_RATE)
        criterion_cent = nn.BCELoss()
        
        for ep in range(config.LOCAL_EPOCHS):
            users_perm = np.random.permutation(valid_users)
            for idx, u in enumerate(users_perm):
                if idx % 20 == 0: 
                    print(f"    - Epoch {ep+1}/{config.LOCAL_EPOCHS} | Tiến độ: {idx}/{len(valid_users)} Users")
                
                X_tr, y_tr = load_data(u, 'train')
                if len(X_tr) <= 1: 
                    del X_tr, y_tr
                    continue 
                
                dl_drop = True if len(X_tr) > config.BATCH_SIZE else False
                loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=config.BATCH_SIZE, shuffle=True, drop_last=dl_drop)
                cent_model.train()
                for bx, by in loader:
                    if len(bx) <= 1: continue
                    bx, by = bx.to(device), by.to(device)
                    optimizer_cent.zero_grad()
                    loss = criterion_cent(cent_model(bx), by)
                    loss.backward()
                    optimizer_cent.step()
                del X_tr, y_tr, loader 
            gc.collect() 
                
        preds_cent = []
        for u in valid_users:
            X_te, _ = load_data(u, 'test')
            preds_cent.append(eval_model(cent_model, X_te, False, device))
            del X_te, _
        results["1. Centralized Model"] = {"y_true": y_true_all, "y_scores": np.concatenate(preds_cent)}
        del cent_model, optimizer_cent
        gc.collect()
        
        # -------------------------------------------------------------
        # 2. Local AI Model
        # -------------------------------------------------------------
        print("\n -> 2. Đang train Local AI (Không chia sẻ)...")
        y_scores_local = []
        for idx, u in enumerate(valid_users):
            if idx % 20 == 0: 
                print(f"    - Tiến độ: Train độc lập xong {idx}/{len(valid_users)} Users")
            loc_model = BaseModel(config.FEATURES).to(device)
            X_tr, y_tr = load_data(u, 'train')
            loc_model = train_model(loc_model, X_tr, y_tr, config.LOCAL_EPOCHS, False, device)
            del X_tr, y_tr
            
            X_te, _ = load_data(u, 'test')
            y_scores_local.append(eval_model(loc_model, X_te, False, device))
            del X_te, _, loc_model
        results["2. Local AI Model"] = {"y_true": y_true_all, "y_scores": np.concatenate(y_scores_local)}
        gc.collect()
        
        # -------------------------------------------------------------
        # 3. Standard Personalized FedAvg
        # -------------------------------------------------------------
        print("\n -> 3. Đang train Personalized FedAvg...")
        fed_global = BaseModel(config.FEATURES).to(device)
        fed_clients = {u: BaseModel(config.FEATURES).to(device) for u in valid_users}
        
        for rnd in range(config.FL_ROUNDS):
            print(f"    - FL Round {rnd+1}/{config.FL_ROUNDS} đang chạy...")
            weights = []
            for u in valid_users:
                m = fed_clients[u]
                m.cnn.load_state_dict(fed_global.cnn.state_dict())
                m.lstm.load_state_dict(fed_global.lstm.state_dict())
                
                X_tr, y_tr = load_data(u, 'train')
                m = train_model(m, X_tr, y_tr, config.LOCAL_EPOCHS, False, device)
                del X_tr, y_tr
                
                weights.append({'cnn': m.cnn.state_dict(), 'lstm': m.lstm.state_dict()})
                
            if weights:
                g_cnn = fed_global.cnn.state_dict()
                g_lstm = fed_global.lstm.state_dict()
                for k in g_cnn.keys(): 
                    g_cnn[k] = sum(w['cnn'][k] for w in weights) / len(weights)
                for k in g_lstm.keys(): 
                    g_lstm[k] = sum(w['lstm'][k] for w in weights) / len(weights)
                fed_global.cnn.load_state_dict(g_cnn)
                fed_global.lstm.load_state_dict(g_lstm)
            gc.collect()
                
        preds_fed = []
        for u in valid_users:
            X_te, _ = load_data(u, 'test')
            preds_fed.append(eval_model(fed_clients[u], X_te, False, device))
            del X_te, _
        results["3. Standard FedAvg"] = {"y_true": y_true_all, "y_scores": np.concatenate(preds_fed)}
        del fed_global, fed_clients
        gc.collect()
        
        # -------------------------------------------------------------
        # 4. Proposed Adaptive Neuro-Fuzzy FL
        # -------------------------------------------------------------
        print("\n -> 4. Đang train Proposed Fuzzy FL...")
        prop_global = ProposedModel(config.FEATURES).to(device)
        prop_clients = {u: ProposedModel(config.FEATURES).to(device) for u in valid_users}
        
        for rnd in range(config.FL_ROUNDS):
            print(f"    - Trust-Aware FL Round {rnd+1}/{config.FL_ROUNDS} đang chạy...")
            weights, trusts = [], []
            for u in valid_users:
                m = prop_clients[u]
                m.base.cnn.load_state_dict(prop_global.base.cnn.state_dict())
                m.base.lstm.load_state_dict(prop_global.base.lstm.state_dict())
                
                X_tr, y_tr = load_data(u, 'train')
                m = train_model(m, X_tr, y_tr, config.LOCAL_EPOCHS, True, device)
                del X_tr, y_tr
                
                X_val, y_val = load_data(u, 'val')
                m.eval()
                with torch.no_grad():
                    if len(X_val) > 0:
                        X_device, y_device = X_val.to(device), y_val.to(device)
                        idx_gen = (y_device == 1).squeeze()
                        if idx_gen.dim() == 0: idx_gen = idx_gen.unsqueeze(0)
                        
                        if idx_gen.sum() > 0:
                            X_gen = X_device[idx_gen]
                            noise = torch.std(X_gen, dim=[1,2], unbiased=False, keepdim=True)
                            trust_score = m(X_gen, noise).mean().item()
                        else: trust_score = 0.5
                    else: trust_score = 0.5
                    trusts.append(np.exp(trust_score * 5.0))
                del X_val, y_val
                
                weights.append({'cnn': m.base.cnn.state_dict(), 'lstm': m.base.lstm.state_dict()})
                
            if weights:
                total_t = sum(trusts) + 1e-9
                g_cnn = prop_global.base.cnn.state_dict()
                g_lstm = prop_global.base.lstm.state_dict()
                for k in g_cnn.keys(): 
                    g_cnn[k] = sum(weights[i]['cnn'][k] * (trusts[i]/total_t) for i in range(len(weights)))
                for k in g_lstm.keys(): 
                    g_lstm[k] = sum(weights[i]['lstm'][k] * (trusts[i]/total_t) for i in range(len(weights)))
                prop_global.base.cnn.load_state_dict(g_cnn)
                prop_global.base.lstm.load_state_dict(g_lstm)
            gc.collect()

        preds_prop = []
        for u in valid_users:
            X_te, _ = load_data(u, 'test')
            preds_prop.append(eval_model(prop_clients[u], X_te, True, device))
            del X_te, _
        results["4. Proposed Fuzzy FL"] = {"y_true": y_true_all, "y_scores": np.concatenate(preds_prop)}
        del prop_global, prop_clients
        gc.collect()
        
        export_all(results)
        print("\n[OK] CHƯƠNG TRÌNH ĐÃ HOÀN TẤT. KẾT QUẢ NẰM TRONG THƯ MỤC RESULTS.")
        
    except Exception as e:
        print("\n" + "="*65)
        print("LỖI HỆ THỐNG:")
        print("="*65)
        traceback.print_exc()
        print("="*65)

if __name__ == "__main__":
    run_baselines()