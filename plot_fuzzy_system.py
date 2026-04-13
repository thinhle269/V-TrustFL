import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Cài đặt font chữ chuẩn học thuật
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

def gaussian_mf(x, m, s, eps=1e-4):
    return np.exp(-0.5 * ((x - m) / (np.abs(s) + eps))**2)

def generate_fuzzy_figure():
    print("[1] Đang tính toán không gian toán học của Hệ Mờ 25-Luật...")
    
    # 1. Khởi tạo tham số (Mô phỏng trạng thái mạng sau khi huấn luyện hội tụ)
    # 5 Mức cho AI Confidence (c_base)
    m_c = np.linspace(0.1, 0.9, 5)
    s_c = np.array([0.15] * 5)
    
    # 5 Mức cho Sensor Noise (v_noise - chuẩn hóa)
    m_n = np.linspace(0.1, 0.9, 5)
    s_n = np.array([0.15] * 5)
    
    # Ma trận 25 luật mờ (W_rule): Mô phỏng Logic 
    # (Confidence cao & Noise thấp -> Trust cao. Noise cao -> Trust bị kéo giảm mượt mà)
    w_rule = np.zeros((5, 5))
    for i in range(5):     # i là index của Confidence
        for j in range(5): # j là index của Noise
            trust_val = m_c[i] * (1.0 - 0.5 * m_n[j]) # Hệ số suy hao do nhiễu
            w_rule[i, j] = max(0.01, min(0.99, trust_val))

    # 2. Tạo không gian lưới 2D
    c_vals = np.linspace(0, 1, 100)
    n_vals = np.linspace(0, 1, 100)
    C, N = np.meshgrid(c_vals, n_vals)
    T_fuzzy = np.zeros_like(C)

    # 3. Tính toán Forward Pass của Hệ mờ
    for r in range(100):
        for c in range(100):
            c_val = C[r, c]
            n_val = N[r, c]
            
            # Fuzzification
            mu_c = np.array([gaussian_mf(c_val, m_c[i], s_c[i]) for i in range(5)])
            mu_n = np.array([gaussian_mf(n_val, m_n[j], s_n[j]) for j in range(5)])
            
            # Firing Strength & Defuzzification (Center of Gravity)
            F = np.outer(mu_c, mu_n)
            num = np.sum(F * w_rule)
            den = np.sum(F) + 1e-4
            T_fuzzy[r, c] = num / den

    print("[2] Đang kết xuất hình ảnh chuẩn 300 DPI...")
    fig = plt.figure(figsize=(14, 5.5))
    
    # --- Trục 1: Hàm liên thuộc (Membership Functions) ---
    ax1 = fig.add_subplot(1, 2, 1)
    colors = ['#d7191c', '#fdae61', '#ffffbf', '#abdda4', '#2b83ba']
    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    for i in range(5):
        ax1.plot(c_vals, gaussian_mf(c_vals, m_c[i], s_c[i]), 
                 color=colors[i], lw=2.5, label=labels[i])
    ax1.set_title('(a) Gaussian Membership Functions', fontweight='bold')
    ax1.set_xlabel('Input Variable Domain (Normalized)')
    ax1.set_ylabel('Degree of Membership ($\mu$)')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=5)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.05])

    # --- Trục 2: Mặt phẳng mờ 3D (Fuzzy Surface) ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax2.plot_surface(C, N, T_fuzzy, cmap='coolwarm', edgecolor='none', alpha=0.9)
    ax2.set_title('(b) Non-linear Neuro-Fuzzy Control Surface', fontweight='bold')
    ax2.set_xlabel('AI Confidence ($c_{base}$)', labelpad=10)
    ax2.set_ylabel('Sensor Variance ($v_{noise}$)', labelpad=10)
    ax2.set_zlabel('Fuzzy Output ($T_{fuzzy}$)', labelpad=10)
    ax2.view_init(elev=25, azim=-125) # Xoay góc nhìn khoe độ mượt
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10, pad=0.1)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "Fig_Fuzzy_Surface.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Đã xuất ảnh {out_path} thành công!")

if __name__ == "__main__":
    generate_fuzzy_figure()