import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib parameters to adhere to TechMes checklist:
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.labelweight': 'bold',
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
    'axes.linewidth': 1.5,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.6
})

with open('Resultat/4w_sauttemp.lvm', 'r') as f:
    lines = f.readlines()

t_raw = np.array([float(l.split()[0].replace(',', '.')) for l in lines])
T_raw = np.array([float(l.split()[1].replace(',', '.')) for l in lines])

# Exclure les 2 premiers points avant le demarrage du saut (points a t=0.0s et t=0.415s)
t = t_raw[2:] - t_raw[2]  # On reinitialise l'origine des temps t=0 au debut du saut
T = T_raw[2:]

T_0 = T[0]  # 24.11 °C au debut du saut
T_inf = np.mean(T[-20:])  # 66.71 °C

# Fit exponentiel: T(t) = A + B * exp(-t / tau)
tau_grid = np.linspace(2.0, 10.0, 10001)
best_res = 1e9
best_tau = None
best_A = None
best_B = None

for tau in tau_grid:
    x = np.exp(-t / tau)
    M = np.column_stack([np.ones_like(x), x])
    coeffs, residuals, rank, s = np.linalg.lstsq(M, T, rcond=None)
    res = np.sum((T - (coeffs[0] + coeffs[1] * x))**2)
    if res < best_res:
        best_res = res
        best_tau = tau
        best_A = coeffs[0]
        best_B = coeffs[1]

# Courbe du fit
t_dense = np.linspace(0, t[-1], 500)
T_fit_curve = best_A + best_B * np.exp(-t_dense / best_tau)

# Methode 1: Tangente a l'origine (a t = 0)
slope_tan = (best_A - T_0) / best_tau
t_inter = best_tau
t_tan = np.linspace(0, t_inter + 1.5, 100)
T_tan = T_0 + slope_tan * t_tan

# Methode 2: 63% amplitude
T_63 = T_0 + 0.632 * (best_A - T_0)
t_63 = np.interp(T_63, T, t)

# Methode 4: 95% reponse
t_95 = 2.996 * best_tau
T_95 = T_0 + 0.95 * (best_A - T_0)

fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

# Points experimentaux (sans les 2 premiers points)
ax.plot(t, T, 'o', color='#1f77b4', markersize=4.5, label='Mesures du saut (Pt100 4W)', alpha=0.85)

# Fit curve
ax.plot(t_dense, T_fit_curve, '-', color='#d62728', linewidth=2.5, label=f'Fit exponentiel (tau = {best_tau:.2f} s)')

# Tangente
ax.plot(t_tan, T_tan, '--', color='#2ca02c', linewidth=2.2, label=f'Tangente a l origine (tau = {best_tau:.2f} s)')

# Asymptote
ax.axhline(best_A, color='#7f7f7f', linestyle=':', linewidth=1.8, label=f'Asymptote T_inf = {best_A:.2f} °C')
ax.axhline(T_0, color='#bcbd22', linestyle=':', linewidth=1.5, label=f'T_initiale = {T_0:.2f} °C')

# Intersection tangente
ax.plot(t_inter, best_A, 's', color='#2ca02c', markersize=8, zorder=5)
ax.vlines(t_inter, ymin=20, ymax=best_A, colors='#2ca02c', linestyles='dashdot', linewidth=1.2)

# Point 63%
ax.plot(t_63, T_63, '^', color='#ff7f0e', markersize=8, zorder=5, label=f'Methode 63% (T = {T_63:.1f} °C, t = {t_63:.2f} s)')
ax.hlines(T_63, xmin=0, xmax=t_63, colors='#ff7f0e', linestyles='dotted', linewidth=1.2)
ax.vlines(t_63, ymin=20, ymax=T_63, colors='#ff7f0e', linestyles='dotted', linewidth=1.2)

# Point 95%
ax.plot(t_95, T_95, 'D', color='#9467bd', markersize=7, zorder=5, label=f'Temps a 95% (t_95 = {t_95:.2f} s)')
ax.hlines(T_95, xmin=0, xmax=t_95, colors='#9467bd', linestyles='dotted', linewidth=1.2)
ax.vlines(t_95, ymin=20, ymax=T_95, colors='#9467bd', linestyles='dotted', linewidth=1.2)

ax.set_xlabel('Temps t [s]')
ax.set_ylabel('Temperature T [°C]')
ax.set_title('Reponse indicielle de la Pt100 (echelon dans l eau - sans points initiaux)')
ax.set_xlim(-1, 50)
ax.set_ylim(20, 72)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='lower right', framealpha=0.95)

plt.tight_layout()
plt.savefig('assets/graphs/reponse_indicielle_pt100.png', dpi=300)
plt.savefig('assets/graphs/reponse_indicielle_pt100.pdf')
print('OK new graph')
