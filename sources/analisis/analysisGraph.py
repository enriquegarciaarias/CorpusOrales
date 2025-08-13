from sources.common.common import processControl, logger, log_
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def visualizeProsodicComparisons(stats_csv_path):
    """
    Genera visualizaciones para resaltar conclusiones de análisis comparativo
    sobre prosodia por lemma, sexo y edad.

    Parámetros:
    -----------
    stats_csv_path : str
        Ruta al CSV generado por processProsodicStats.
    output_dir : str
        Carpeta donde guardar las visualizaciones.
    """
    # Leer datos

    edad_labels = {
        "1": "18–34",
        "2": "35–64",
        "3": "≥65"
    }

    df = pd.read_csv(stats_csv_path)

    # Convertir edad a etiquetas
    df["edad_label"] = df["edad"].astype(str).map(edad_labels)

    # Gráfico 1: Boxplot de mean_pitch por sexo y edad, separado por lemma
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=df,
        x="edad_label",
        y="mean_mean_pitch",
        hue="sexo",
        palette={"H": "blue", "M": "red"}
    )
    plt.title("Distribución de mean_pitch por sexo y edad")
    plt.xlabel("Grupo de edad")
    plt.ylabel("mean_pitch (Hz)")

    # Capturar leyenda original y reasignar nombres manteniendo colores
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, ["Hombres", "Mujeres"], title="Sexo")

    plt.tight_layout()
    outputPath = os.path.join(processControl.env['outputDir'], "boxplot_mean_pitch.png")
    plt.savefig(outputPath, dpi=300)
    plt.close()

    # Gráfico 2: Barras de duración media por lemma, sexo y edad
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="edad_label",
        y="mean_duration",
        hue="sexo",
        palette={"H": "blue", "M": "red"},
        errorbar=None  # reemplaza ci=None
    )
    plt.title("Duración media por sexo y edad")
    plt.xlabel("Grupo de edad")
    plt.ylabel("Duración media (s)")
    plt.legend(title="Sexo", labels=["Hombres", "Mujeres"])
    plt.tight_layout()
    outputPath = os.path.join(processControl.env['outputDir'], "barplot_duration.png")
    plt.savefig(outputPath, dpi=300)
    plt.close()

    print(f"Gráficos guardados en {outputPath}")

