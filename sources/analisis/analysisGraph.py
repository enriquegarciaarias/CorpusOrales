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

    df = pd.read_csv(stats_csv_path, sep=';', decimal=',')

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

    log_("info", logger, f"Gráficos guardados en {outputPath}")


def analyze_correlations(stats_csv_path):
    """
    Analiza correlaciones entre variables numéricas globalmente y por categoría.
    """
    df = pd.read_csv(stats_csv_path, sep=';', decimal=',')

    # Variables numéricas para el análisis
    numeric_cols = [
        'mean_mean_pitch', 'median_mean_pitch', 'std_mean_pitch',
        'mean_hnr', 'median_hnr', 'std_hnr',
        'mean_mean_energy', 'median_mean_energy', 'std_mean_energy',
        'mean_duration', 'median_duration', 'std_duration',
        'mean_speech_rate', 'median_speech_rate', 'std_speech_rate'
    ]

    # -------------------
    # 1. Correlación global
    # -------------------
    corr_global = df[numeric_cols].corr(method='pearson')

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_global, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
    plt.title("Matriz de correlación global")
    plt.tight_layout()
    outputPath = os.path.join(processControl.env['outputDir'], f"correlacion_global.png")
    plt.savefig(outputPath, dpi=300)
    plt.close()

    # -------------------
    # 2. Correlaciones por lemma
    # -------------------
    for lemma in df['lemma'].unique():
        df_lemma = df[df['lemma'] == lemma]
        corr_lemma = df_lemma[numeric_cols].corr(method='pearson')

        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_lemma, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
        plt.title(f"Matriz de correlación - Lemma: {lemma}")
        plt.tight_layout()
        outputPath = os.path.join(processControl.env['outputDir'], f"correlacion_{lemma}.png")
        plt.savefig(outputPath, dpi=300)
        plt.close()

    # -------------------
    # 3. Correlaciones por sexo
    # -------------------
    for sexo in df['sexo'].unique():
        df_sexo = df[df['sexo'] == sexo]
        corr_sexo = df_sexo[numeric_cols].corr(method='pearson')

        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_sexo, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
        plt.title(f"Matriz de correlación - Sexo: {'Hombres' if sexo=='H' else 'Mujeres'}")
        plt.tight_layout()
        outputPath = os.path.join(processControl.env['outputDir'], f"correlacion_sexo_{sexo}.png")
        plt.savefig(outputPath, dpi=300)
        plt.close()

    # -------------------
    # 4. Correlaciones por grupo de edad
    # -------------------
    for edad in df['edad'].unique():
        df_edad = df[df['edad'] == edad]
        corr_edad = df_edad[numeric_cols].corr(method='pearson')

        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_edad, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
        plt.title(f"Matriz de correlación - Grupo edad: {edad}")
        plt.tight_layout()
        outputPath = os.path.join(processControl.env['outputDir'], f"correlacion_edad_{edad}.png")
        plt.savefig(outputPath, dpi=300)
        plt.close()

    log_("info", logger, f"Correlaciones guardadas en {outputPath}")


def analyze_selected_features_correlation(stats_csv_path):
    """
    Analiza correlaciones de un subconjunto de variables prosódicas
    separadas por sexo y genera visualizaciones.

    Parámetros:
    -----------
    stats_csv_path : str
        Ruta al CSV con las estadísticas (generado por processProsodicStats).
    output_dir : str
        Carpeta donde se guardarán los gráficos.
    """
    # Variables seleccionadas
    selected_features = ["mean_mean_pitch", "mean_hnr", "mean_duration", "mean_speech_rate"]

    # Leer datos
    df = pd.read_csv(stats_csv_path, sep=";", decimal=",")

    # Asegurar que las columnas existen
    missing_cols = [col for col in selected_features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas en el CSV: {missing_cols}")

    # Filtrar por sexo
    for sexo, label in [("H", "Hombres"), ("M", "Mujeres")]:
        df_sexo = df[df["sexo"] == sexo][selected_features]

        # Calcular matriz de correlación
        corr_matrix = df_sexo.corr()

        # Graficar
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            vmin=-1, vmax=1,
            fmt=".2f",
            annot_kws={"size": 10}
        )
        plt.title(f"Matriz de correlación - {label}")
        plt.tight_layout()

        # Guardar gráfico
        outputPath = os.path.join(processControl.env['outputDir'], f"correlacion_{sexo}_4features.png")

        plt.savefig(outputPath, dpi=300)
        plt.close()

        log_("info", logger, f"Gráfico guardado en: {outputPath}")
