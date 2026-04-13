import matplotlib.pyplot as plt

# Dữ liệu từ bảng
papers = ['Sitová', 'Abuhamad', 'Yang', 'Meng', 'Yin', 'V-TrustFL']
eer_values = [7.16, 5.20, 5.84, 8.72, 8.50, 8.99]

# Định nghĩa màu sắc theo nhóm
# Nhóm 1 (Centralized): Đỏ/Xám -> Chọn Đỏ nhạt (Lightcoral)
# Nhóm 2 (Standard Federated): Cam -> Chọn Orange
# Nhóm 3 (Zero-Trust Federated): Xanh đậm -> Chọn Darkblue
colors = ['#f08080', '#f08080', '#f08080', '#ffa500', '#ffa500', '#00008b']

# Khởi tạo biểu đồ
plt.figure(figsize=(10, 6))
bars = plt.bar(papers, eer_values, color=colors, width=0.4)

# Thêm tiêu đề và nhãn trục
plt.title('Comparison of EER (%) across different Architectures', fontsize=14, fontweight='bold')
plt.xlabel('V-TrustFL vs SOTA', fontsize=12)
plt.ylabel('Equal Error Rate (EER %)', fontsize=12)
plt.ylim(0, 11)  # Giới hạn trục Y để biểu đồ thoáng hơn

# Hiển thị giá trị cụ thể trên đầu mỗi cột
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval}%', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Thêm chú thích (Legend)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#f08080', lw=4, label='Centralized (None Privacy)'),
    Line2D([0], [0], color='#ffa500', lw=4, label='Standard Federated'),
    Line2D([0], [0], color='#00008b', lw=4, label='Zero-Trust Federated (Proposed)')
]
plt.legend(handles=legend_elements, loc='upper left')

# Hiển thị lưới mờ cho trục Y
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Xuất biểu đồ
plt.tight_layout()
plt.show()