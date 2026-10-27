import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] # 如果报错可改为英文
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. 填入你之前的 58 个核数据 (格式: [名称, A, Z, 真实质量u])
nuclei = [
    ("D",2,1,2.0141017781),("T",3,1,3.0160492813),("He3",3,2,3.0160293220),
    ("He4",4,2,4.0026032541),("Li6",6,3,6.0151228874),("Li7",7,3,7.0160034366),
    ("Be9",9,4,9.0121830658),("B10",10,5,10.0129369985),("B11",11,5,11.0093051665),
    ("C12",12,6,12.0000000000),("C13",13,6,13.0033548351),("N14",14,7,14.0030740044),
    ("N15",15,7,15.0001088989),("O16",16,8,15.9949146196),("O17",17,8,16.9991317565),
    ("O18",18,8,17.9991596129),("F19",19,9,18.9984031627),("Ne20",20,10,19.9924401762),
    ("Ne22",22,10,21.9913851140),("Na23",23,11,22.9897692820),("Mg24",24,12,23.9850416971),
    ("Mg26",26,12,25.9825929670),("Al27",27,13,26.9815385300),("Si28",28,14,27.9769265347),
    ("Si30",30,14,29.9737701710),("P31",31,15,30.9737619984),("S32",32,16,31.9720711744),
    ("S34",34,16,33.9678669000),("Cl35",35,17,34.9688526820),("Ar36",36,18,35.9675451060),
    ("Ar40",40,18,39.9623831237),("K39",39,19,38.9637064864),("Ca40",40,20,39.9625908630),
    ("Ca48",48,20,47.9525229000),("Ti48",48,22,47.9479419800),("Cr52",52,24,51.9405062300),
    ("Mn55",55,25,54.9380439100),("Fe54",54,26,53.9396089900),("Fe56",56,26,55.9349363300),
    ("Fe57",57,26,56.9353928400),("Co59",59,27,58.9331942900),("Ni58",58,28,57.9353424100),
    ("Ni60",60,28,59.9307858800),("Ni62",62,28,61.9283448800),("Cu63",63,29,62.9295977200),
    ("Zn64",64,30,63.9291420100),("Zn66",66,30,65.9260338100),("Ga69",69,31,68.9255735000),
    ("Ge74",74,32,73.9211777600),("As75",75,33,74.9215945700),("Se80",80,34,79.9165218000),
    ("Br79",79,35,78.9183376000),("Kr84",84,36,83.9114977300),("Rb85",85,37,84.9117897379),
    ("Sr88",88,38,87.9056125000),("Y89",89,39,88.9058403000),("Zr90",90,40,89.9046977000),
    ("Mo98",98,42,97.9054069000)
]

M_p = 938.27208816
M_n = 939.56542052
u_to_MeV = 931.49410242
m_e = 0.51099895
t_unit = M_p
# 拟合参数
popt = [0.015272, 0.015793, 0.000580, 0.021634, 0.002963]
a, b, c, d, de = popt

def liquid_drop_dt(A, Z):
    N = A - Z
    pairing = de if (Z%2==0 and N%2==0) else (-de if (Z%2!=0 and N%2!=0) else 0.0)
    return a*A - b*A**(2/3) - c*Z*(Z-1)/A**(1/3) - d*(N-Z)**2/A + pairing

# 计算残差
A_list, Z_list, N_list, errors, names = [], [], [], [], []
for name, A, Z, mu in nuclei:
    N = A - Z
    M_true = mu * u_to_MeV - Z * m_e * 1e-3 * u_to_MeV
    dt = liquid_drop_dt(A, Z)
    M_pred = (Z*(M_p/t_unit) + N*(M_n/t_unit) - dt) * t_unit
    err = M_pred - M_true
    A_list.append(A)
    Z_list.append(Z)
    N_list.append(N)
    errors.append(err)
    names.append(name)

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(A_list, errors, c=Z_list, cmap='viridis', s=80, edgecolor='k', alpha=0.8)
plt.colorbar(sc, label='Proton Number (Z)')

# 标注幻数（壳层效应假说）
magic_numbers = [2, 8, 20, 28, 50, 82]
for m in magic_numbers:
    ax.axvline(m, color='red', linestyle='--', alpha=0.3)
    if m <= max(A_list):
        ax.text(m, max(errors)+1, f'Z/N={m}', color='red', rotation=90, va='top', ha='right', fontsize=8)

ax.axhline(0, color='black', linewidth=1)
ax.set_xlabel('Mass Number A', fontsize=14)
ax.set_ylabel('Residual (Predicted - True) [MeV]', fontsize=14)
ax.set_title('Residuals of 58 Nuclear Masses (Color: Z)', fontsize=16)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()