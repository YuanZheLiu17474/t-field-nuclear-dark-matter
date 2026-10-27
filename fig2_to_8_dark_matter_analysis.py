import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

print("="*70)
print("综合场 t 理论：高级扩展分析 (旋转曲线 & BEC形成红移)")
print("="*70)

# ============================================================
# 模块 1：星系旋转曲线 (Rotation Curve) - 基于 Fig 4 的 BEC 核心
# ============================================================
print("\n[模块 1] 正在计算：BEC 暗物质晕的旋转曲线...")

G = 4.3e-6  # kpc (km/s)^2 / Msun
rho_0 = 1e7 # 中心密度
R_TF = 15.0 # BEC 云边界

r = np.linspace(0.1, 50, 500) # kpc
# 我们用简化的BEC密度分布 (抛物线型)
rho_BEC = np.where(r <= R_TF, rho_0 * (1 - (r/R_TF)**2), 0)

# 计算质量 M(r) = 4*pi * integral(rho * r^2 dr)
M_enclosed = np.zeros_like(r)
dr = r[1] - r[0]
for i in range(1, len(r)):
    M_enclosed[i] = M_enclosed[i-1] + 4 * np.pi * r[i]**2 * rho_BEC[i] * dr

# 计算旋转速度 v(r) = sqrt(G*M(r)/r)
v_BEC = np.sqrt(np.where(r > 0, G * M_enclosed / r, 0))

# 对比标准的暗物质晕 (假设总质量相同，用近似 NFW 的旋转曲线)
M_total = M_enclosed[-1]
v_NFW = np.sqrt(G * M_total / r) * 0.7 # 仅作形状对比

plt.figure(figsize=(10, 6))
plt.plot(r, v_BEC, 'b-', lw=3, label='t-Field BEC Halo (Flat Core)')
plt.plot(r, v_NFW, 'r--', lw=2.5, label='Standard CDM Halo (Approx.)')
plt.xlabel(r'Radius \(r\) (kpc)', fontsize=14)
plt.ylabel(r'Rotation Velocity \(v(r)\) (km/s)', fontsize=14)
plt.title(r'Figure 7: Galactic Rotation Curves (BEC vs CDM)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.xlim(0, 30)
plt.show()

# ============================================================
# 模块 2：暗光子 BEC 形成的红移时刻 (Redshift of Condensation)
# ============================================================
print("\n[模块 2] 正在计算：早期宇宙 BEC 形成的红移时刻...")

# 红移 z 从 0 到 10000
z = np.logspace(0, 4, 500)

# 宇宙微波背景温度 (K)
T_gamma = 2.725 * (1 + z)

# 假设暗光子 BEC 的临界温度 (现象学参数，取决于暗光子数和质量)
# 这里为了展示，我们假设 Tc ~ 1000 K
T_c = 1000.0 

# 找到交点：当宇宙温度 T_gamma 降到 T_c 时，BEC 开始形成
cross_index = np.argmin(np.abs(T_gamma - T_c))
z_cross = z[cross_index]

plt.figure(figsize=(10, 6))
plt.loglog(z, T_gamma, 'k-', lw=2, label=r'Cosmic Temperature \(T(z) = T_0(1+z)\)')
plt.axhline(T_c, color='b', linestyle='--', lw=2.5, label=r'Critical Temperature for BEC \(T_c\)')

# 标注交点
plt.scatter([z_cross], [T_c], color='red', s=150, zorder=5)
plt.annotate(r'BEC Formation: \(z \approx {0:.0f}\)'.format(z_cross), xy=(z_cross, T_c), xytext=(z_cross*0.1, T_c*3),
             arrowprops=dict(facecolor='red', shrink=0.05), fontsize=14, color='red')

plt.xlabel(r'Redshift \(z\)', fontsize=14)
plt.ylabel(r'Temperature (K)', fontsize=14)
plt.title(r'Figure 8: Redshift of Dark Photon BEC Formation', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", alpha=0.3)
plt.xlim(1, 1e4)
plt.ylim(1, 1e5)
plt.show()

print("\n" + "="*70)
print("生成完毕！关闭窗口即可。")
print("="*70)