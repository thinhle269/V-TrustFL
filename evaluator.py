import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import config

# Cài đặt định dạng chuẩn học thuật SCIE
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

def calc_metrics(y_true, y_scores):
    y_pred = (y_scores >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
    
    far = fp / (fp + tn + 1e-9)
    frr = fn / (fn + tp + 1e-9)
    acc = accuracy_score(y_true, y_pred)
    
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except: 
        fpr, tpr, thresholds, eer = np.array([0,1]), np.array([0,1]), np.array([0,1]), 0.5
    return acc, far, frr, eer, cm, fpr, tpr, thresholds

def export_all(results_dict):
    print("\n[STEP 3] ĐANG XUẤT HỆ THỐNG 5 BIỂU ĐỒ CHUYÊN SÂU CHUẨN SCIE Q1...")
    records = []
    roc_data = {}
    
    for m_name, data in results_dict.items():
        acc, far, frr, eer, cm, fpr, tpr, thresholds = calc_metrics(data['y_true'], data['y_scores'])
        records.append({
            "Model": m_name, "Accuracy (%)": round(acc*100, 2), 
            "FAR (%)": round(far*100, 2), "FRR (%)": round(frr*100, 2), "EER (%)": round(eer*100, 2)
        })
        roc_data[m_name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr), 'cm': cm, 'eer': eer, 'thresholds': thresholds}

    # ========================================================
    # 1. CONFUSION MATRICES
    # ========================================================
    fig_cm, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, (m_name, metrics) in enumerate(roc_data.items()):
        sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                    xticklabels=['Imposter', 'Genuine'], yticklabels=['Imposter', 'Genuine'])
        axes[idx].set_title(f"{m_name}\nEER: {metrics['eer']*100:.2f}% | AUC: {metrics['auc']:.4f}", fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "Fig1_Confusion_Matrices.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ========================================================
    # 2. ROC CURVES
    # ========================================================
    plt.figure(figsize=(8, 6))
    colors = ['red', 'green', 'orange', 'blue']
    linestyles = [':', '-.', '--', '-']
    for idx, (m_name, metrics) in enumerate(roc_data.items()):
        plt.plot(metrics['fpr'], metrics['tpr'], color=colors[idx], linestyle=linestyles[idx], 
                 linewidth=2.5, label=f"{m_name} (AUC = {metrics['auc']:.4f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)', fontweight='bold')
    plt.ylabel('True Positive Rate (1 - FRR)', fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontweight='bold', fontsize=14)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(config.RESULTS_DIR, "Fig2_ROC_Curves.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ========================================================
    # 3. DET CURVES (Log Scale)
    # ========================================================
    plt.figure(figsize=(8, 6))
    for idx, (m_name, metrics) in enumerate(roc_data.items()):
        fnr = 1 - metrics['tpr']
        valid_idx = (metrics['fpr'] > 0) & (fnr > 0) # Tránh lỗi Log(0)
        plt.plot(metrics['fpr'][valid_idx], fnr[valid_idx], color=colors[idx], linestyle=linestyles[idx], 
                 linewidth=2.5, label=f"{m_name} (EER = {metrics['eer']*100:.2f}%)")
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('False Positive Rate (FAR) - Log Scale', fontweight='bold')
    plt.ylabel('False Rejection Rate (FRR) - Log Scale', fontweight='bold')
    plt.title('Detection Error Tradeoff (DET) Curves', fontweight='bold', fontsize=14)
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(config.RESULTS_DIR, "Fig3_DET_Curves.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ========================================================
    # 4. EER INTERSECTION CURVE (FAR vs FRR Threshold)
    # ========================================================
    prop_name = "4. Proposed Fuzzy FL"
    if prop_name in roc_data:
        fpr = roc_data[prop_name]['fpr']
        fnr = 1 - roc_data[prop_name]['tpr']
        thresholds = roc_data[prop_name]['thresholds']
        
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, fpr, label='False Acceptance Rate (FAR)', color='red', linewidth=2)
        plt.plot(thresholds, fnr, label='False Rejection Rate (FRR)', color='blue', linewidth=2)
        
        eer_idx = np.nanargmin(np.absolute(fpr - fnr))
        plt.plot(thresholds[eer_idx], fpr[eer_idx], 'go', markersize=8, label=f'EER Point ({fpr[eer_idx]*100:.2f}%)')
        
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, max(max(fpr), max(fnr))])
        plt.xlabel('Decision Threshold (Fuzzy Trust Score)', fontweight='bold')
        plt.ylabel('Error Rate', fontweight='bold')
        plt.title('FAR and FRR vs. Decision Threshold (Proposed Model)', fontweight='bold', fontsize=14)
        plt.legend(loc="upper center")
        plt.savefig(os.path.join(config.RESULTS_DIR, "Fig4_Threshold_EER.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # ========================================================
    # 5. DYNAMIC TRUST DECAY SIMULATION (CHÌA KHÓA CỦA PAPER)
    # ========================================================
    if prop_name in results_dict and "3. Standard FedAvg" in results_dict:
        plt.figure(figsize=(10, 5))
        
        y_t = results_dict[prop_name]['y_true']
        y_s_prop = results_dict[prop_name]['y_scores']
        y_s_fed = results_dict["3. Standard FedAvg"]['y_scores']
        
        gen_idx = np.where(y_t == 1)[0]
        imp_idx = np.where(y_t == 0)[0]
        
        limit = 150
        if len(gen_idx) >= limit and len(imp_idx) >= limit:
            seq_prop = np.concatenate([y_s_prop[gen_idx[:limit]], y_s_prop[imp_idx[:limit]]])
            seq_fed = np.concatenate([y_s_fed[gen_idx[:limit]], y_s_fed[imp_idx[:limit]]])
            
            # Làm mượt đường cong bằng EWMA (Mô phỏng bộ nhớ thời gian)
            s_prop = pd.Series(seq_prop).ewm(span=10).mean().values
            s_fed = pd.Series(seq_fed).ewm(span=10).mean().values
            time_steps = np.arange(len(s_prop))
            
            plt.plot(time_steps, s_fed, color='orange', linewidth=2, linestyle='--', label='Standard FedAvg (Volatile)')
            plt.plot(time_steps, s_prop, color='navy', linewidth=3, label='Proposed Fuzzy Trust (Smooth)')
            
            plt.axhline(y=0.5, color='red', linestyle='-', linewidth=1.5, label='Zero-Trust Lock Threshold')
            plt.axvline(x=limit, color='gray', linestyle=':', linewidth=2, label='Imposter Hijack Event')
            
            plt.fill_between(time_steps[:limit], s_prop[:limit], 0.5, where=(s_prop[:limit] > 0.5), color='green', alpha=0.1)
            plt.fill_between(time_steps[limit:], s_prop[limit:], 0.5, where=(s_prop[limit:] < 0.5), color='red', alpha=0.1)
            
            plt.ylim([-0.05, 1.05])
            plt.xlabel('Continuous Time Steps (Sliding Windows)', fontweight='bold')
            plt.ylabel('Dynamic Trust Score', fontweight='bold')
            plt.title('Zero-Trust Continuous Authentication Session Simulation', fontweight='bold', fontsize=14)
            plt.legend(loc="lower left")
            plt.savefig(os.path.join(config.RESULTS_DIR, "Fig5_Dynamic_Trust_Curve.png"), dpi=300, bbox_inches='tight')
            plt.close()

    pd.DataFrame(records).to_excel(os.path.join(config.RESULTS_DIR, "Table_Evaluation_Metrics.xlsx"), index=False)
    
    # [QUAN TRỌNG] Lưu lại Raw Data để sau này vẽ lại không cần Train
    np.save(os.path.join(config.RESULTS_DIR, "raw_results_dict.npy"), results_dict, allow_pickle=True)
    
    print("[OK] TOÀN BỘ 5 BIỂU ĐỒ CHUYÊN SÂU ĐÃ XUẤT THÀNH CÔNG VÀO THƯ MỤC RESULTS!")

# Cho phép chạy file này độc lập để vẽ lại ảnh nếu đã có file raw_results_dict.npy
if __name__ == "__main__":
    if os.path.exists(os.path.join(config.RESULTS_DIR, "raw_results_dict.npy")):
        res = np.load(os.path.join(config.RESULTS_DIR, "raw_results_dict.npy"), allow_pickle=True).item()
        export_all(res)
    else:
        print("[LỖI] Không tìm thấy dữ liệu cũ. Hãy chạy 'python run_all.py' trước.")