import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Cài đặt font chuẩn học thuật IEEE
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

def extract_real_session_data(results_file, limit=150):
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"[LỖI] Không tìm thấy file dữ liệu thực nghiệm: {results_file}. Hãy chạy evaluator.py trước.")
    
    print(f"[1] Đang nạp dữ liệu thực nghiệm (Real Predictions) từ: {results_file}")
    res = np.load(results_file, allow_pickle=True).item()
    
    # Lấy tên key chính xác từ dictionary
    model_fed = "3. Standard FedAvg"
    model_prop = "4. Proposed Fuzzy FL"
    
    if model_fed not in res or model_prop not in res:
        raise KeyError("[LỖI] Dữ liệu không chứa keys của mô hình FedAvg hoặc Proposed. Vui lòng kiểm tra lại.")

    y_true = res[model_prop]['y_true']
    scores_fed = res[model_fed]['y_scores']
    scores_vtrust = res[model_prop]['y_scores']
    
    # Tìm index của dữ liệu Hợp lệ (1) và Kẻ cướp máy (0)
    idx_gen = np.where(y_true == 1)[0]
    idx_imp = np.where(y_true == 0)[0]
    
    if len(idx_gen) < limit or len(idx_imp) < limit:
        print(f"[Cảnh báo] Số lượng mẫu Test không đủ {limit}. Đang dùng số lượng tối đa có thể.")
        limit = min(len(idx_gen), len(idx_imp))
        
    print(f"[2] Trích xuất chuỗi thời gian thực: {limit} samples Genuine + {limit} samples Imposter")
    
    # Ghép chuỗi để tạo kịch bản Cướp máy: Nửa đầu là Chủ máy, Nửa sau là Kẻ gian
    # Dùng dữ liệu thật để đảm bảo tính tái lập (Reproducibility) 100%
    seq_fed = np.concatenate([scores_fed[idx_gen[:limit]], scores_fed[idx_imp[:limit]]])
    seq_vtrust = np.concatenate([scores_vtrust[idx_gen[:limit]], scores_vtrust[idx_imp[:limit]]])
    
    return seq_fed, seq_vtrust, limit

def run_zero_trust_simulation():
    results_file = os.path.join("results", "raw_results_dict.npy")
    
    try:
        seq_fed, seq_vtrust, hijack_step = extract_real_session_data(results_file, limit=150)
    except Exception as e:
        print(e)
        return
        
    total_steps = len(seq_fed)
    time_steps = np.arange(total_steps)
    threshold = 0.5
    
    # Áp dụng EWMA để mô phỏng bộ nhớ thời gian (Time-decay) của hệ thống
    # AI xác thực liên tục luôn dùng Moving Average để tránh bị khóa bởi 1 frame nhiễu
    span_val = 8
    fedavg_smoothed = pd.Series(seq_fed).ewm(span=span_val).mean().values
    vtrust_smoothed = pd.Series(seq_vtrust).ewm(span=span_val).mean().values
    
    print("[3] Đang tính toán Metrics định lượng...")
    # Tính Variance trong pha Genuine (Sự ổn định, càng nhỏ càng tốt)
    var_fed = np.var(fedavg_smoothed[:hijack_step])
    var_vtrust = np.var(vtrust_smoothed[:hijack_step])
    
    # Số lần khóa oan (False Lockouts) trong lúc chủ máy đang dùng
    false_locks_fed = int(np.sum(fedavg_smoothed[:hijack_step] < threshold))
    false_locks_vtrust = int(np.sum(vtrust_smoothed[:hijack_step] < threshold))
    
    # Độ trễ phát hiện (Detection Delay) - Nhịp từ lúc cướp máy đến lúc bị khóa
    try:
        delay_fed = int(np.argmax(fedavg_smoothed[hijack_step:] < threshold))
        if delay_fed == 0 and fedavg_smoothed[hijack_step] >= threshold: delay_fed = total_steps - hijack_step
    except: delay_fed = total_steps - hijack_step
        
    try:
        delay_vtrust = int(np.argmax(vtrust_smoothed[hijack_step:] < threshold))
        if delay_vtrust == 0 and vtrust_smoothed[hijack_step] >= threshold: delay_vtrust = total_steps - hijack_step
    except: delay_vtrust = total_steps - hijack_step
    
    metrics_df = pd.DataFrame({
        "Metric": ["Genuine Phase Variance (Stability)", "False Lockouts (Instances)", "Lock-out Delay (Timesteps)"],
        "Standard FedAvg (Empirical)": [round(var_fed, 4), false_locks_fed, delay_fed],
        "V-TrustFL Proposed (Empirical)": [round(var_vtrust, 4), false_locks_vtrust, delay_vtrust]
    })
    
    os.makedirs("results", exist_ok=True)
    table_path = os.path.join("results", "Table_ZeroTrust_Metrics.csv")
    metrics_df.to_csv(table_path, index=False)
    
    print("[4] Đang vẽ Biểu đồ Zero-Trust Continuous Authentication...")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    ax.plot(time_steps, fedavg_smoothed, color='#f59e0b', linewidth=2, linestyle='--', label='Standard FedAvg (Volatile Trust)')
    ax.plot(time_steps, vtrust_smoothed, color='#1e3a8a', linewidth=3, label='V-TrustFL (Smooth Fuzzy Trust)')
    
    ax.axhline(y=threshold, color='#dc2626', linestyle='-', linewidth=1.5, label=r'Zero-Trust Lock Threshold ($\tau = 0.5$)')
    ax.axvline(x=hijack_step, color='gray', linestyle=':', linewidth=2, label=f'Imposter Hijack Event ($t={hijack_step}$)')
    
    # Tô màu nền vùng an toàn và vùng nguy hiểm
    ax.fill_between(time_steps[:hijack_step], vtrust_smoothed[:hijack_step], threshold, 
                    where=(vtrust_smoothed[:hijack_step] >= threshold), color='#dcfce7', alpha=0.5)
    ax.fill_between(time_steps[hijack_step:], vtrust_smoothed[hijack_step:], threshold, 
                    where=(vtrust_smoothed[hijack_step:] < threshold), color='#fee2e2', alpha=0.5)
    
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([0, total_steps])
    ax.set_xlabel('Continuous Time Steps (Sliding Windows)', fontweight='bold')
    ax.set_ylabel('Empirical Trust Score ($p_t$)', fontweight='bold')
    ax.set_title('Zero-Trust Continuous Authentication Session (Empirical HMOG Data)', fontweight='bold')
    
    ax.legend(loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    fig_path = os.path.join("results", "Fig_ZeroTrust_Simulation_Empirical.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Đã xuất thành công Bảng và Biểu đồ dựa trên DỮ LIỆU THỰC!")

if __name__ == "__main__":
    run_zero_trust_simulation()