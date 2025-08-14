from sources.common.common import processControl, logger, log_
from sources.common.utils import leeJson
from sources.analisis.analysisAcusticoProsodico import processProsodicStats
from sources.analisis.analysisGraph import visualizeProsodicComparisons, analyze_correlations, analyze_selected_features_correlation

import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def frecuenciaBase(resultsPath):
    results = leeJson(resultsPath)
    header_data = []

    for entry in results:
        query = entry.get("query", {})
        results_data = entry.get("results", {})
        header_data.append({
            "lemma": query.get("lemma", "unknown"),
            "sexo": query.get("sexo", "unknown"),
            "edad": query.get("edad", "unknown"),
            "Ejemplos": results_data.get("Ejemplos", 0),
            "Entrevistas": results_data.get("Entrevistas", 0)
        })

    df = pd.DataFrame(header_data)

    df_grouped = df.groupby(["lemma", "sexo", "edad"], as_index=False)[["Ejemplos", "Entrevistas"]].sum()

    print("\nTabla combinada de frecuencias:")
    print(df_grouped)
    return df_grouped

def preparar_datos_para_indices(df_grouped, lemmas_objetivo=None):
    """
    Prepara la estructura lista de dicts para calcular_indices_genero a partir del df_grouped.
    lemmas_objetivo: lista con lemmas a filtrar (ejemplo: ["bar", "iglesia"])

    Devuelve lista con dicts {'sexo': 'H'/'M', 'elementos': [lemma-edad, lemma-edad, ...]}
    donde cada elemento se repite tantas veces como la suma de Ejemplos + Entrevistas.
    """
    lista_resultados = []

    # Filtrar por lemmas si se indica
    if lemmas_objetivo is not None:
        df_filtered = df_grouped[df_grouped['lemma'].isin(lemmas_objetivo)]
    else:
        df_filtered = df_grouped

    for _, row in df_filtered.iterrows():
        elementos = []
        # Concatenar lemma y edad → p.ej. "bar-1"
        elemento_lexico = f"{row['lemma']}-{row['edad']}"
        # Repetir el elemento la cantidad de veces de "Ejemplos" + "Entrevistas"
        total = int(row['Ejemplos']) + int(row['Entrevistas'])
        elementos.extend([elemento_lexico] * total)

        if elementos:
            lista_resultados.append({
                "sexo": row['sexo'],
                "elementos": elementos
            })
    return lista_resultados

def OLDpreparar_datos_para_indices(df_grouped, lemmas_objetivo=None):
    """
    Prepara la estructura lista de dicts para calcular_indices_genero a partir del df_grouped.
    lemmas_objetivo: lista con lemmas a filtrar (ejemplo: ["bar", "iglesia"])

    Devuelve lista con dicts {'sexo': 'H'/'M', 'elementos': [lemma, lemma, ...]}
    donde cada elemento se repite tantas veces como la suma de Ejemplos + Entrevistas.
    """
    lista_resultados = []

    # Filtrar por lemmas si se indica
    if lemmas_objetivo is not None:
        df_filtered = df_grouped[df_grouped['lemma'].isin(lemmas_objetivo)]
    else:
        df_filtered = df_grouped

    for _, row in df_filtered.iterrows():
        elementos = []
        # Repetir el lemma la cantidad de veces de "Ejemplos" + "Entrevistas"
        total = int(row['Ejemplos']) + int(row['Entrevistas'])
        elementos.extend([row['lemma']] * total)

        if elementos:
            lista_resultados.append({
                "sexo": row['sexo'],
                "elementos": elementos
            })
    return lista_resultados


def calcular_indices_genero(resultados_previos):
    hombres = Counter()
    mujeres = Counter()
    total_hombres = 0
    total_mujeres = 0

    for r in resultados_previos:
        if r["sexo"] == "H":
            hombres.update(r["elementos"])
            total_hombres += len(r["elementos"])
        elif r["sexo"] == "M":
            mujeres.update(r["elementos"])
            total_mujeres += len(r["elementos"])

    elementos = set(hombres.keys()) | set(mujeres.keys())

    datos = []
    for elem in elementos:
        freq_H = hombres[elem]
        freq_M = mujeres[elem]
        prop_H = freq_H / total_hombres if total_hombres > 0 else 0
        prop_M = freq_M / total_mujeres if total_mujeres > 0 else 0
        if prop_M > 0:
            indice = prop_H / prop_M
        else:
            indice = float('inf') if prop_H > 0 else 1.0

        datos.append({
            "elemento": elem,
            "freq_H": freq_H,
            "freq_M": freq_M,
            "prop_H": prop_H,
            "prop_M": prop_M,
            "indice_sesgo": indice
        })

    return pd.DataFrame(datos).sort_values(by="indice_sesgo", ascending=False)


def visualizar_indices_genero(df_indices, top_n=20, titulo="Índice de sesgo por género"):
    df_plot = df_indices.copy()
    df_plot["sesgo_abs"] = df_plot["indice_sesgo"].apply(lambda x: abs(x - 1))
    df_plot = df_plot.sort_values(by="sesgo_abs", ascending=False).head(top_n)

    # Crear columna para hue
    df_plot["predominio"] = df_plot["indice_sesgo"].apply(lambda x: "Más Hombres" if x > 1 else "Más Mujeres")

    # Ordenar alfabéticamente el eje Y
    df_plot = df_plot.sort_values(by="elemento", ascending=True)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df_plot,
        x="indice_sesgo",
        y="elemento",
        hue="predominio",
        palette="coolwarm"
    )

    # Línea vertical de referencia
    plt.axvline(1, color="gray", linestyle="--")

    # Añadir etiquetas de valor en las barras
    for p in ax.patches:
        valor = p.get_width()
        ax.annotate(
            f"{valor:.2f}",
            (p.get_width(), p.get_y() + p.get_height() / 2),
            ha="left", va="center",
            fontsize=8,
            xytext=(3, 0),  # Desplazamiento para que no tape la barra
            textcoords="offset points"
        )

    plt.title(titulo)
    plt.xlabel("Índice de sesgo (prop_H / prop_M)")
    plt.ylabel("Elemento léxico")
    plt.legend(title="Predominio")
    plt.tight_layout()

    output_path = os.path.join(processControl.env['outputDir'], "graficoSesgoGenero.png")
    plt.savefig(output_path)
    log_("info", logger, f"Gráfico guardado en {output_path}")



def processAnalisis(resultsPath, lemmas_objetivo=None):
    # Ejemplo: lemmas_objetivo = ["bar", "iglesia"]
    df_grouped = frecuenciaBase(resultsPath)
    resultados_preparados = preparar_datos_para_indices(df_grouped, lemmas_objetivo)
    df_indices = calcular_indices_genero(resultados_preparados)
    visualizar_indices_genero(df_indices, top_n=15)
    filePath = processProsodicStats(resultsPath)
    #visualizeProsodicComparisons(filePath)
    analyze_correlations(filePath)
    analyze_selected_features_correlation(filePath)
    return True


