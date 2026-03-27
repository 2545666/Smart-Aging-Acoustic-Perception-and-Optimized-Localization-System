import numpy as np
import matplotlib.pyplot as plt

# ================= 1. 环境配置 (学术风 + 宋体) =================
# 注意：如果您的系统没有 SimSun 字体，matplotlib 会自动回退到默认字体
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'SimHei', 'serif'] 
plt.rcParams['axes.unicode_minus'] = False

# ================= 2. 参数定义 =================
EMOTIONS = ['Happy', 'Sad', 'Neutral', 'Anger']
NUM_EMOTIONS = len(EMOTIONS)
NUM_SAMPLES = 5000 # 5000次压力测试

def ds_combine(m1, u1, m2, u2):
    """
    核心 D-S 融合算法 (Dempster's Rule of Combination)
    m1, m2: 视觉和听觉的基本概率分配 (BPA)
    u1, u2: 视觉和听觉的不确定度 Θ (Uncertainty)
    """
    # 计算冲突系数 K
    K = 0.0
    for i in range(NUM_EMOTIONS):
        for j in range(NUM_EMOTIONS):
            if i != j: K += m1[i] * m2[j]

    # Zadeh 悖论保护：如果冲突接近 1.0，则退化为简单平均
    if K > 0.999: return (m1 + m2) / 2.0
    
    # 计算正交和公式：m(A) = (m1(A)m2(A) + m1(A)u2 + m2(A)u1) / (1-K)
    m_fused = np.zeros(NUM_EMOTIONS)
    for i in range(NUM_EMOTIONS):
        m_fused[i] = (m1[i]*m2[i] + m1[i]*u2 + m2[i]*u1) / (1 - K)
    
    return m_fused / np.sum(m_fused)

def run_simulation():
    correct_v, correct_a, correct_f = 0, 0, 0
    
    for _ in range(NUM_SAMPLES):
        true_emo = np.random.randint(0, NUM_EMOTIONS)
        
        # --- 模拟互补逻辑：视觉擅长 Happy/Anger，听觉擅长 Sad/Neutral ---
        if true_emo in [0, 3]: # 视觉强势领域
            v_acc, a_acc = 0.78, 0.45
            v_u, a_u = 0.15, 0.35 # 擅长时更自信 (不确定度低)
        else: # 听觉强势领域
            v_acc, a_acc = 0.45, 0.78
            v_u, a_u = 0.35, 0.15
            
        # 模拟视觉输出 (BPA生成)
        v_probs = np.zeros(NUM_EMOTIONS)
        if np.random.rand() < v_acc:
            v_probs[true_emo] = 0.8
        else:
            v_probs[np.random.randint(0, NUM_EMOTIONS)] = 0.7
        v_m = v_probs * (1 - v_u) / np.sum(v_probs)
            
        # 模拟听觉输出 (BPA生成)
        a_probs = np.zeros(NUM_EMOTIONS)
        if np.random.rand() < a_acc:
            a_probs[true_emo] = 0.8
        else:
            a_probs[np.random.randint(0, NUM_EMOTIONS)] = 0.7
        a_m = a_probs * (1 - a_u) / np.sum(a_probs)

        # 执行 D-S 融合
        f_m = ds_combine(v_m, v_u, a_m, a_u)
        
        # 统计结果 (Argmax判定)
        if np.argmax(v_m) == true_emo: correct_v += 1
        if np.argmax(a_m) == true_emo: correct_a += 1
        if np.argmax(f_m) == true_emo: correct_f += 1
        
    return [c/NUM_SAMPLES for c in [correct_v, correct_a, correct_f]]

# ================= 3. 运行并绘图 =================
accs = run_simulation()
labels = ['仅视觉感知 (FER)', '仅听觉感知 (SER)', 'D-S 决策级融合']
data = [a * 100 for a in accs]

# 配色方案：淡蓝紫色系 (Academic Blue-Purple)
# #A7C7E7 (淡蓝), #C3B1E1 (淡紫), #8E44AD (深紫)
colors = ['#A7C7E7', '#C3B1E1', '#8E44AD']

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.bar(labels, data, color=colors, width=0.5, edgecolor='#34495E', linewidth=1.2)

# 添加数值标注
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold', color='#2C3E50')

# 修饰坐标轴与标题
ax.set_title('“智岁灵犀”多模态感知性能：蒙特卡洛压力测试分析 (N=5000)', 
             fontsize=18, pad=35, fontfamily='SimSun', fontweight='bold', color='#1A237E')
ax.set_ylabel('平均识别准确率 Accuracy (%)', fontsize=14, fontfamily='SimSun')
ax.set_ylim(0, 105)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# 绘制提升趋势显著标注
max_single = max(data[0], data[1])
improvement = data[2] - max_single
ax.annotate(f'综合性能绝对提升: +{improvement:.1f}%', 
            xy=(2, data[2]), xytext=(1.2, data[2] + 10),
            arrowprops=dict(facecolor='#D32F2F', shrink=0.05, width=3, headwidth=10),
            fontsize=16, color='#D32F2F', fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#D32F2F", lw=2))

# 添加底部说明
ax.text(0.5, -0.15, "数据说明：模拟视听互补场景，验证 D-S 融合算法在高冲突环境下的纠偏与修正能力。", 
        transform=ax.transAxes, ha='center', fontsize=12, color='#7F8C8D', fontfamily='SimSun')

plt.tight_layout()
plt.savefig('ds_monte_carlo_academic.png', dpi=300)
print(f"✅ 仿真完成！准确率：视觉={data[0]:.1f}%, 听觉={data[1]:.1f}%, 融合={data[2]:.1f}%")