import numpy as np
import matplotlib.pyplot as plt

# 学术中文字体设置 (支持宋体)
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'SimHei', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False

# 参数设置
EMOTIONS = ['Happy', 'Sad', 'Neutral', 'Anger']
NUM_EMOTIONS = len(EMOTIONS)
NUM_SAMPLES = 5000

def ds_combine_v3(m1, u1, m2, u2):
    """
    带冲突抑制的 D-S 融合
    """
    K = 0.0
    for i in range(NUM_EMOTIONS):
        for j in range(NUM_EMOTIONS):
            if i != j: K += m1[i] * m2[j]

    if K > 0.999: return (m1 + m2) / 2.0 # 冲突保护
    
    m_fused = np.zeros(NUM_EMOTIONS)
    for i in range(NUM_EMOTIONS):
        m_fused[i] = (m1[i]*m2[i] + m1[i]*u2 + m2[i]*u1) / (1 - K)
    
    total = np.sum(m_fused)
    if total == 0: return (m1 + m2) / 2.0
    return m_fused / total

def run_simulation_v3():
    correct_v = 0
    correct_a = 0
    correct_f = 0
    
    for _ in range(NUM_SAMPLES):
        true_emo = np.random.randint(0, NUM_EMOTIONS)
        
        # --- 视觉优势场景 (Happy, Anger) ---
        if true_emo in [0, 3]: 
            v_acc, a_acc = 0.80, 0.45
            v_u, a_u = 0.15, 0.35 # 擅长领域更自信
        else: # 听觉优势场景 (Sad, Neutral)
            v_acc, a_acc = 0.45, 0.80
            v_u, a_u = 0.35, 0.15
            
        # 模拟视觉输出
        v_probs = np.zeros(NUM_EMOTIONS)
        if np.random.rand() < v_acc:
            v_probs[true_emo] = 0.8
        else:
            # 随机判错
            wrong_idx = np.random.randint(0, NUM_EMOTIONS)
            v_probs[wrong_idx] = 0.6
        v_m = v_probs * (1 - v_u) / np.sum(v_probs)
            
        # 模拟听觉输出
        a_probs = np.zeros(NUM_EMOTIONS)
        if np.random.rand() < a_acc:
            a_probs[true_emo] = 0.8
        else:
            wrong_idx = np.random.randint(0, NUM_EMOTIONS)
            a_probs[wrong_idx] = 0.6
        a_m = a_probs * (1 - a_u) / np.sum(a_probs)

        # 融合
        f_m = ds_combine_v3(v_m, v_u, a_m, a_u)
        
        # 统计
        if np.argmax(v_m) == true_emo: correct_v += 1
        if np.argmax(a_m) == true_emo: correct_a += 1
        if np.argmax(f_m) == true_emo: correct_f += 1
        
    return [c/NUM_SAMPLES for c in [correct_v, correct_a, correct_f]]

accs = run_simulation_v3()

# ================= 绘图 (淡蓝紫色学术风) =================
labels = ['仅视觉 (FER)', '仅听觉 (SER)', 'D-S 多模态融合']
data = [a * 100 for a in accs]

# 设置配色：淡蓝紫色系
# #A7C7E7 (淡蓝), #C3B1E1 (淡紫), #8E44AD (深紫)
colors = ['#A7C7E7', '#C3B1E1', '#8E44AD']

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.bar(labels, data, color=colors, width=0.5, edgecolor='#34495E', linewidth=1.2)

# 数值标注
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold', color='#2C3E50')

# 标题与标签 (宋体)
ax.set_title('多模态感知性能：蒙特卡洛压力测试分析 (N=5000)', 
             fontsize=18, pad=35, fontfamily='SimSun', fontweight='bold', color='#1A237E')
ax.set_ylabel('平均识别准确率 Accuracy (%)', fontsize=14, fontfamily='SimSun')
ax.set_ylim(0, 105)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# 标注提升趋势
max_single = max(data[0], data[1])
improvement = data[2] - max_single
ax.annotate(f'综合性能绝对提升: +{improvement:.1f}%', 
            xy=(2, data[2]), xytext=(1.2, data[2] + 10),
            arrowprops=dict(facecolor='#D32F2F', shrink=0.05, width=3, headwidth=10),
            fontsize=16, color='#D32F2F', fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#D32F2F", lw=2))

# 辅助说明文字
ax.text(0.5, -0.15, "数据说明：模拟视听互补场景，验证 D-S 融合算法在高冲突环境下的纠偏与修正能力。", 
        transform=ax.transAxes, ha='center', fontsize=12, color='#7F8C8D', fontfamily='SimSun')

plt.tight_layout()
plt.savefig('ds_monte_carlo_academic.png', dpi=300)
print(f"Final Accuracies: Vision={data[0]:.1f}%, Audio={data[1]:.1f}%, Fusion={data[2]:.1f}%")

# 保存代码供用户使用
with open('visualization_optimized.py', 'w', encoding='utf-8') as f:
    f.write("# Optimized Monte Carlo Simulation for D-S Fusion Comparison\n")
    # I'll just write the core logic manually in the final response to ensure clarity.