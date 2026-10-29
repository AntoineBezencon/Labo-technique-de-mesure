import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def charger_lvm(chemin_fichier):
  """Charge un fichier .lvm de LabVIEW en ignorant l'en-tête et en utilisant

  la tabulation comme séparateur.
  """
  # skiprows=23 permet généralement de sauter l'en-tête LabVIEW standard.
  # Si cela pose un problème, ajustez ce nombre en ouvrant votre fichier avec un bloc-notes.
  df = pd.read_csv(chemin_fichier, sep="\t", skiprows=23, header=None)

  # Nettoyage : on garde les deux premières colonnes (Temps et Température) et on supprime les NaN éventuels
  df = df.iloc[:, :2].dropna()
  return df.iloc[:, 0].values, df.iloc[:, 1].values


def comparer_mesures_lvm(chemin_standard, chemin_oversampling):
  # 1. Chargement des deux fichiers .lvm
  t_std, T_std = charger_lvm(chemin_standard)
  t_over, T_over = charger_lvm(chemin_oversampling)

  # 2. Calculs statistiques avec NumPy
  moyenne_std = np.mean(T_std)
  std_std = np.std(T_std, ddof=1)  # Écart-type empirique

  moyenne_over = np.mean(T_over)
  std_over = np.std(T_over, ddof=1)

  # 3. Affichage des résultats dans la console
  print("==========================================")
  print("        RAPPORT DE COMPARAISON")
  print("==========================================")
  print(f"--- 1. Série Standard (Section 1.4) ---")
  print(f"  * Température moyenne : {moyenne_std:.3f} °C")
  print(f"  * Écart-type (bruit)  : {std_std:.3f} °C")
  print(f"\n--- 2. Série Suréchantillonnée (Section 1.5) ---")
  print(f"  * Température moyenne : {moyenne_over:.3f} °C")
  print(f"  * Écart-type (bruit)  : {std_over:.3f} °C")
  print("==========================================")

  # 4. Tracé graphique comparatif
  plt.figure(figsize=(10, 5))
  plt.plot(t_std, T_std, label="Mesure Standard", alpha=0.6, color="orange")
  plt.plot(
      t_over,
      T_over,
      label="Suréchantillonnage (Moyenne 100 pts)",
      linewidth=1.5,
      color="blue",
  )

  plt.xlabel("Temps (s)")
  plt.ylabel("Température (°C)")
  plt.title("Comparaison des mesures - Sèche-cheveux (Labo 3)")
  plt.legend()
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.tight_layout()
  plt.show()


# --- Exemple d'utilisation ---
# Remplacez par les noms de vos fichiers .lvm réels :
# comparer_mesures_lvm('Normal.lvm', 'Oversampling.lvm')
comparer_mesures_lvm('/Resultat/Normal.lvm', '/Resultat/Oversampling.lvm')
