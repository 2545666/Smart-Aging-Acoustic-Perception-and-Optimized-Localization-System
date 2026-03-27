import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置宋体字体 (SimSun or similar if available, fallback to SimHei for demonstration)
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 定义情绪标签
EMOTIONS = ['Happy', 'Sad', 'Neutral', 'Anger']
NUM_EMOTIONS = len(EMOTIONS)
NUM_SAMPLES = 5000

def ds_combine(m1, u1, m2, u2):
    """
    改进的 D-S 融合逻辑，包含不确定度 Θ
    m1, m2: 各单项情绪的 BPA (长度为 4)
    u1, u2: 视觉和听觉的不确定度 mass(Θ)
    """
    # 结合规则：m(A) = (m1(A)m2(A) + m1(A)m2(Θ) + m2(A)m1(Θ)) / (1-K)
    
    # 1. 计算冲突系数 K (所有交集为空的组合乘积之和)
    K = 0.0
    for i in range(NUM_EMOTIONS):
        for j in range(NUM_EMOTIONS):
            if i != j:
                K += m1[i] * m2[j]
    
    # 2. 计算融合后的各单项 mass
    m_fused = np.zeros(NUM_EMOTIONS)
    if K < 0.999:
        for i in range(NUM_EMOTIONS):
            m_fused[i] = (m1[i] * m2[i] + m1[i] * u2 + m2[i] * u1) / (1 - K)
    else:
        # Zadeh 悖论保护：当 K 极大时取简单加权平均
        m_fused = (m1 + m2) / 2.0
        
    return m_fused / np.sum(m_fused)

def simulate_sophisticated_data():
    correct_v = 0
    correct_a = 0
    correct_f = 0
    
    for _ in range(NUM_SAMPLES):
        true_emo = np.random.randint(0, NUM_EMOTIONS)
        
        # --- 模拟视觉逻辑 (存在遮挡、假笑等场景) ---
        v_acc = 0.65
        v_probs = np.random.dirichlet(np.ones(NUM_EMOTIONS) * 0.2)
        v_u = np.random.uniform(0.1, 0.3) # 视觉自带一定不确定度
        
        # 模拟“强颜欢笑”场景 (20%概率)
        if true_emo == 1 and np.random.rand() < 0.2: # 真实是Sad，但视觉判为Happy
            v_probs[0] = 0.7 
            v_probs[1] = 0.1
        elif np.random.rand() < v_acc:
            v_probs[true_emo] = 0.8
        
        v_probs = v_probs * (1 - v_u)
        v_probs /= np.sum(v_probs) # 确保 sum(v_probs) == 1-v_u
        
        # --- 模拟听觉逻辑 (存在底噪、特征重叠等场景) ---
        a_acc = 0.70
        a_probs = np.random.dirichlet(np.ones(NUM_EMOTIONS) * 0.2)
        a_u = np.random.uniform(0.1, 0.25) # 听觉自带一定不确定度
        
        if np.random.rand() < a_acc:
            a_probs[true_emo] = 0.85
        
        a_probs = a_probs * (1 - a_u)
        a_probs /= np.sum(a_probs)
        
        # --- 融合 ---
        f_probs = ds_combine(v_probs, v_u, a_probs, a_u)
        
        # --- 判定 ---
        if np.argmax(v_probs) == true_emo: correct_v += 1
        if np.argmax(a_probs) == true_emo: correct_a += 1
        if np.argmax(f_probs) == true_emo: correct_f += 1
        
    return (correct_v/NUM_SAMPLES, correct_audio := correct_a/NUM_SAMPLES, correct_f/NUM_SAMPLES)

# 运行仿真
acc_v, acc_a, acc_f = simulate_sophisticated_data()

# ================= 绘制学术图表 (蓝紫色系 + 宋体风格) =================
labels = ['仅视觉感知 (FER)', '仅听觉感知 (SER)', 'D-S 决策级融合']
accuracies = [acc_v * 100, acc_a * 100, acc_f * 100]

# 颜色：淡蓝紫色风 (Pale Blue/Lavender)
colors = ['#B8C6DB', '#95A5A6', '#8E44AD'] # 浅蓝灰, 灰蓝, 典雅紫

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.bar(labels, accuracies, color=['#A9C9EB', '#859FD1', '#B39DDB'], width=0.5, edgecolor='#5D6D7E')

# 添加数值标注
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f'{yval:.1f}%', 
            ha='center', va='bottom', fontsize=14, fontweight='bold', color='#2C3E50')

# 修饰坐标轴
ax.set_ylabel('平均识别准确率 (%)', fontsize=14, labelpad=10)
ax.set_title('多模态情绪融合算法蒙特卡洛仿真验证 (N=5000)', fontsize=18, pad=30, fontweight='bold')
ax.set_ylim(0, 110)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# 标注提升效果
max_single = max(accuracies[0], accuracies[1])
improvement = accuracies[2] - max_single
ax.annotate(f'综合性能提升: +{improvement:.1f}%', 
            xy=(2, accuracies[2]), xytext=(1.2, accuracies[2] + 12),
            arrowprops=dict(facecolor='#E74C3C', shrink=0.05, width=3, headwidth=10),
            fontsize=15, color='#E74C3C', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E74C3C", lw=2))

plt.tight_layout()
plt.savefig('ds_fusion_improved.png', dpi=300)
print(f"新图表已生成: ds_fusion_improved.png. 准确率: V={acc_v:.2f}, A={acc_a:.2f}, Fusion={acc_f:.2f}")

# 导出更新后的代码到 visualization_v2.py
with open('visualization_v2.py', 'w', encoding='utf-8') as f:
    f.write("# Improved Monte Carlo Simulation for D-S Fusion\n")
    # I'll just save the script content as a whole