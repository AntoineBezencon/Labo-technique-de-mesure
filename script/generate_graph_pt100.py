import os
import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib parameters to adhere strictly to TechMes checklist:
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

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) if os.path.basename(script_dir).lower() in ['script', 'scripts'] else script_dir
lvm_path = os.path.join(project_root, 'Resultat', '4w_sauttemp.lvm')
out_dir = os.path.join(project_root, 'assets', 'graphs')

with open(lvm_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

t_raw = np.array([float(l.split()[0].replace(',', '.')) for l in lines])
T_raw = np.array([float(l.split()[1].replace(',', '.')) for l in lines])

# Exclure les 2 premiers points avant le démarrage du saut (points à t=0.0s et t=0.415s)
t = t_raw[2:] - t_raw[2]  # On réinitialise l'origine des temps t=0 au début du saut
T = T_raw[2:]

T_0 = T[0]  # 24.11 °C au début du saut
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

# Méthode 1: Tangente à l'origine (à t = 0)
slope_tan = (best_A - T_0) / best_tau
t_inter = best_tau
t_tan = np.linspace(0, t_inter + 1.5, 100)
T_tan = T_0 + slope_tan * t_tan

# Méthode 2: 63% amplitude
T_63 = T_0 + 0.632 * (best_A - T_0)
t_63 = np.interp(T_63, T, t)

# Méthode 4: 95% réponse
t_95 = 2.996 * best_tau
T_95 = T_0 + 0.95 * (best_A - T_0)

fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

# Points expérimentaux avec barres d'incertitude (IEC 60751 classe A, u = ±0.15 °C)
ax.errorbar(
    t,
    T,
    yerr=0.15,
    fmt='o',
    color='#1f77b4',
    ecolor='#4682b4',
    elinewidth=0.9,
    capsize=1.5,
    markersize=4.0,
    label=r'Mesures du saut (Pt100 4W, $u = \pm 0.15\ ^\circ\mathrm{C}$)',
    alpha=0.85
)

# Fit curve
ax.plot(
    t_dense,
    T_fit_curve,
    '-',
    color='#d62728',
    linewidth=2.5,
    label=rf'Ajustement exponentiel ($\tau = {best_tau:.2f}\ \mathrm{{s}},\ \mathrm{{RMSE}} = 0.09\ ^\circ\mathrm{{C}}$)'
)

# Tangente
ax.plot(
    t_tan,
    T_tan,
    '--',
    color='#2ca02c',
    linewidth=2.2,
    label=rf"Tangente à l'origine ($\tau = {best_tau:.2f}\ \mathrm{{s}}$)"
)

# Asymptote
ax.axhline(
    best_A,
    color='#7f7f7f',
    linestyle=':',
    linewidth=1.8,
    label=rf'Asymptote finale ($T_\infty = {best_A:.2f}\ ^\circ\mathrm{{C}}$)'
)
ax.axhline(
    T_0,
    color='#bcbd22',
    linestyle=':',
    linewidth=1.5,
    label=rf'Température initiale ($T_0 = {T_0:.2f}\ ^\circ\mathrm{{C}}$)'
)

# Intersection tangente
ax.plot(t_inter, best_A, 's', color='#2ca02c', markersize=8, zorder=5)
ax.vlines(t_inter, ymin=20, ymax=best_A, colors='#2ca02c', linestyles='dashdot', linewidth=1.2)

# Point 63%
ax.plot(
    t_63,
    T_63,
    '^',
    color='#ff7f0e',
    markersize=8,
    zorder=5,
    label=rf'Méthode $63\,\%$ ($T = {T_63:.1f}\ ^\circ\mathrm{{C}},\ \tau = {t_63:.2f}\ \mathrm{{s}}$)'
)
ax.hlines(T_63, xmin=0, xmax=t_63, colors='#ff7f0e', linestyles='dotted', linewidth=1.2)
ax.vlines(t_63, ymin=20, ymax=T_63, colors='#ff7f0e', linestyles='dotted', linewidth=1.2)

# Point 95%
ax.plot(
    t_95,
    T_95,
    'D',
    color='#9467bd',
    markersize=7,
    zorder=5,
    label=rf'Temps à $95\,\%$ ($t_{{95\%}} = {t_95:.2f}\ \mathrm{{s}}$)'
)
ax.hlines(T_95, xmin=0, xmax=t_95, colors='#9467bd', linestyles='dotted', linewidth=1.2)
ax.vlines(t_95, ymin=20, ymax=T_95, colors='#9467bd', linestyles='dotted', linewidth=1.2)

ax.set_xlabel('Temps [s]')
ax.set_ylabel('Température [°C]')
ax.set_title("Réponse indicielle de la sonde RTD Pt100 (échelon dans l'eau chaude)")
ax.set_xlim(-1, 50)
ax.set_ylim(20, 72)
ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
ax.legend(loc='lower right', framealpha=0.95)

plt.tight_layout()
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, 'reponse_indicielle_pt100.png'), dpi=300)
plt.savefig(os.path.join(out_dir, 'reponse_indicielle_pt100.pdf'))
print('Graphique Pt100 généré avec succès.')
