import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set matplotlib parameters to adhere strictly to TechMes checklist:
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.labelweight': 'bold',
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'lines.linewidth': 2.0,
    'axes.linewidth': 1.5,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.6
})


def get_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # if script is in script/, repo root is parent
    if os.path.basename(script_dir).lower() in ['script', 'scripts']:
        return os.path.dirname(script_dir)
    return script_dir


def charger_lvm(chemin_fichier):
    """Charge un fichier .lvm de LabVIEW en gérant le séparateur tabulation

    et la virgule comme séparateur décimal.
    """
    df = pd.read_csv(chemin_fichier, sep=r'\s+', decimal=',', header=None)
    df = df.iloc[:, :2].dropna()
    return df.iloc[:, 0].values.astype(float), df.iloc[:, 1].values.astype(float)


def comparer_mesures_lvm(chemin_standard, chemin_oversampling, out_dir=None):
    # 1. Chargement des deux fichiers .lvm
    t_std, T_std = charger_lvm(chemin_standard)
    t_over, T_over = charger_lvm(chemin_oversampling)

    # 2. Calculs statistiques
    N_std = len(T_std)
    moyenne_std = np.mean(T_std)
    std_std = np.std(T_std, ddof=1)
    u_moy_std = std_std / np.sqrt(N_std)

    N_over = len(T_over)
    moyenne_over = np.mean(T_over)
    std_over = np.std(T_over, ddof=1)
    u_moy_over = std_over / np.sqrt(N_over)

    # 3. Affichage console
    print("==================================================")
    print("        RAPPORT STATISTIQUE NTC (LABO 3)")
    print("==================================================")
    print(f"--- 1. Série Standard (Section 1.4) ---")
    print(f"  * Nombre d'échantillons (N) : {N_std}")
    print(f"  * Température moyenne       : {moyenne_std:.3f} °C")
    print(f"  * Écart-type empirique (s)  : {std_std:.3f} °C")
    print(f"  * Incertitude sur moyenne   : {u_moy_std:.4f} °C")
    print(f"\n--- 2. Série Suréchantillonnée (Section 1.5) ---")
    print(f"  * Nombre d'échantillons (N) : {N_over}")
    print(f"  * Température moyenne       : {moyenne_over:.3f} °C")
    print(f"  * Écart-type empirique (s)  : {std_over:.3f} °C")
    print(f"  * Incertitude sur moyenne   : {u_moy_over:.4f} °C")
    print("==================================================")

    # 4. Tracé graphique conforme à la check-list TechMes
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    ax.plot(
        t_std,
        T_std,
        label=f'Acquisition standard (cadence 20 ms, N = {N_std})',
        color='#ff7f0e',
        alpha=0.8,
        linewidth=2.0
    )
    ax.plot(
        t_over,
        T_over,
        label=f'Suréchantillonnage moyenné (paquet de 100 pts, N = {N_over})',
        color='#1f77b4',
        linewidth=2.2
    )

    ax.set_xlabel('Temps [s]')
    ax.set_ylabel('Température [°C]')
    ax.set_title("Comparaison temporelle de la température de l'air chaud (CTN SEMI833-ET)")
    ax.set_xlim(0, max(t_std[-1], t_over[-1]))
    ax.set_ylim(24, 46)
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    ax.legend(loc='lower right', framealpha=0.95)

    plt.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        pdf_path = os.path.join(out_dir, 'comparaison_ntc_oversampling.pdf')
        png_path = os.path.join(out_dir, 'comparaison_ntc_oversampling.png')
        plt.savefig(pdf_path)
        plt.savefig(png_path, dpi=300)
        print(f"Graphiques sauvegardés dans :\n  - {pdf_path}\n  - {png_path}")

    plt.close()


if __name__ == '__main__':
    root = get_project_root()
    std_file = os.path.join(root, 'Resultat', 'Normal.lvm')
    over_file = os.path.join(root, 'Resultat', 'Oversampling.lvm')
    graph_dir = os.path.join(root, 'assets', 'graphs')

    comparer_mesures_lvm(std_file, over_file, graph_dir)
