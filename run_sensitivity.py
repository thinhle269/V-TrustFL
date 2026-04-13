import os
# Tắt tính năng biên dịch để chống MemoryError trên Windows
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCH_DYNAMO_DISABLE'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import config
from models import ProposedModel
from evaluator import calc_metrics

def load_data(user_idx, split='train'):
    X = np.load(os.path.join(config.PROCESSED_DIR, f'X_{split}_{user_idx}.npy'))
    y = np.load(os.path.join(config.PROCESSED_DIR, f'y_{split}_{user_idx}.npy'))
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)
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
            else: outputs = model(bx)
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
            else: outputs = model(bx)
            preds.append(outputs.cpu().numpy())
    if not preds: return np.array([])
    return np.concatenate(preds).flatten()

def run_lambda_analysis():
    print("="*70)
    print(" BẮT ĐẦU PHÂN TÍCH ĐỘ NHẠY (SENSITIVITY ANALYSIS) CHO THAM SỐ LAMBDA")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    valid_users = [u for u in range(config.NUM_USERS) if os.path.exists(os.path.join(config.PROCESSED_DIR, f'y_test_{u}.npy'))]
    
    y_test_all = [load_labels_only(u, 'test') for u in valid_users]
    y_true_all = np.concatenate(y_test_all)
    del y_test_all; gc.collect()

    lambda_values = [1.0, 3.0, 5.0, 7.0, 9.0]
    records = []

    for l_val in lambda_values:
        print(f"\n[🔬] ĐANG HUẤN LUYỆN VỚI HỆ SỐ PHẠT MŨ LAMBDA = {l_val}")
        prop_global = ProposedModel(config.FEATURES).to(device)
        prop_clients = {u: ProposedModel(config.FEATURES).to(device) for u in valid_users}
        
        for rnd in range(config.FL_ROUNDS):
            weights, trusts = [], []
            for idx, u in enumerate(valid_users):
                if idx % 20 == 0: print(f"    - Round {rnd+1}/{config.FL_ROUNDS} | Tiến độ: {idx}/{len(valid_users)} Users")
                
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
                    
                    # [QUAN TRỌNG: THAY ĐỔI LAMBDA TẠI ĐÂY]
                    trusts.append(np.exp(trust_score * l_val))
                del X_val, y_val
                
                weights.append({'cnn': m.base.cnn.state_dict(), 'lstm': m.base.lstm.state_dict()})
                
            if weights:
                total_t = sum(trusts) + 1e-9
                g_cnn = prop_global.base.cnn.state_dict()
                g_lstm = prop_global.base.lstm.state_dict()
                for k in g_cnn.keys(): g_cnn[k] = sum(weights[i]['cnn'][k] * (trusts[i]/total_t) for i in range(len(weights)))
                for k in g_lstm.keys(): g_lstm[k] = sum(weights[i]['lstm'][k] * (trusts[i]/total_t) for i in range(len(weights)))
                prop_global.base.cnn.load_state_dict(g_cnn); prop_global.base.lstm.load_state_dict(g_lstm)
            gc.collect()

        preds_prop = []
        for u in valid_users:
            X_te, _ = load_data(u, 'test')
            preds_prop.append(eval_model(prop_clients[u], X_te, True, device))
            del X_te, _
        
        y_scores_all = np.concatenate(preds_prop)
        acc, far, frr, eer, _, _, _, _ = calc_metrics(y_true_all, y_scores_all)
        
        print(f"  -> Kết quả tại Lambda={l_val}: EER={eer*100:.2f}%, FAR={far*100:.2f}%, FRR={frr*100:.2f}%")
        records.append({"Lambda (λ)": l_val, "Accuracy (%)": round(acc*100, 2), "FAR (%)": round(far*100, 2), "FRR (%)": round(frr*100, 2), "EER (%)": round(eer*100, 2)})
        del prop_global, prop_clients, y_scores_all
        gc.collect()

    df = pd.DataFrame(records)
    df.to_excel(os.path.join(config.RESULTS_DIR, "Table_Lambda_Sensitivity.xlsx"), index=False)
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 6))
    plt.plot(df["Lambda (λ)"], df["EER (%)"], marker='s', markersize=8, linewidth=3, color='navy', label='EER (%)')
    plt.plot(df["Lambda (λ)"], df["FAR (%)"], marker='o', linestyle='--', linewidth=2, color='red', label='FAR (%)')
    plt.plot(df["Lambda (λ)"], df["FRR (%)"], marker='^', linestyle='-.', linewidth=2, color='green', label='FRR (%)')
    
    plt.axvline(x=5.0, color='gray', linestyle=':', linewidth=2, label='Optimal λ Selection')
    plt.title('Sensitivity Analysis of Trust Penalty Factor (λ)', fontweight='bold', fontsize=14)
    plt.xlabel('Exponential Penalty Factor (λ)', fontweight='bold', fontsize=12)
    plt.ylabel('Error Rate (%)', fontweight='bold', fontsize=12)
    plt.xticks(lambda_values)
    plt.legend(loc='upper center')
    plt.savefig(os.path.join(config.RESULTS_DIR, "Fig6_Lambda_Sensitivity.png"), dpi=300, bbox_inches='tight')
    print("\n[🎉] XUẤT SẮC! Đã lưu Bảng và Biểu đồ Sensitivity Analysis vào thư mục results.")

if __name__ == "__main__":
    run_lambda_analysis()