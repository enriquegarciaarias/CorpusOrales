from sources.common.common import processControl, logger, log_
from sources.common.utils import leeJson

import json
import pandas as pd
import os
from collections import defaultdict

def frecuenciaBase(resultsPath):
    results = leeJson(resultsPath)
    all_results = []
    for entry in results:
        query = entry.get("query", {})
        lemma = query.get("lemma", "unknown")
        sexo = query.get("sexo", "unknown")
        if "analysis" in entry:
            for analysis in entry["analysis"]:
                result = analysis["result"]
                prosodia = result.get("prosodia", {})
                result.update({
                    "hnr": prosodia.get("hnr"),
                    "jitter_local": prosodia.get("jitter_local"),
                    "shimmer_local": prosodia.get("shimmer_local"),
                    "lemma": lemma,
                    "sexo": sexo
                })
                all_results.append(result)

    # Convertir a DataFrame
    df = pd.DataFrame(all_results)

    # Seleccionar columnas numéricas de interés
    numeric_columns = ['mean_pitch', 'mean_energy', 'speech_rate']

    # Eliminar filas con NaN solo para las columnas numéricas seleccionadas
    df_numeric = df[numeric_columns].dropna()

    # Agregar las columnas categóricas relevantes al DataFrame filtrado
    df_analysis = df[['lemma', 'sexo']].loc[df_numeric.index].reset_index(drop=True)
    df_analysis = pd.concat([df_analysis, df_numeric], axis=1)

    # Calcular estadísticas descriptivas por lema y sexo
    descriptive_stats = df_analysis.groupby(['lemma', 'sexo'])[numeric_columns].agg(['mean', 'median', 'std']).round(2)

    # Realizar pruebas t para diferencias significativas
    def perform_ttest(df, metric, lemma):
        group_h = df[(df['lemma'] == lemma) & (df['sexo'] == 'H')][metric].dropna()
        group_m = df[(df['lemma'] == lemma) & (df['sexo'] == 'M')][metric].dropna()
        if len(group_h) > 1 and len(group_m) > 1:  # Asegurar suficientes datos
            t_stat, p_value = stats.ttest_ind(group_h, group_m, equal_var=False)  # Welch's t-test
            return t_stat, p_value
        return None, None

    # Analizar diferencias por lema
    results_ttest = {}
    for lemma in ['bar', 'iglesia']:
        results_ttest[lemma] = {}
        for metric in numeric_columns:
            t_stat, p_value = perform_ttest(df_analysis, metric, lemma)
            results_ttest[lemma][metric] = {'t_stat': t_stat, 'p_value': p_value}

    # Mostrar resultados
    print("Estadísticas Descriptivas por Lema y Sexo:")
    print(descriptive_stats)
    print("\nPruebas t para Diferencias Significativas (p < 0.05 indica diferencia significativa):")
    for lemma, metrics in results_ttest.items():
        print(f"\nLema: {lemma}")
        for metric, stats in metrics.items():
            t_stat, p_value = stats['t_stat'], stats['p_value']
            if t_stat is not None and p_value is not None:
                significance = "Significativa" if p_value < 0.05 else "No significativa"
                print(f"  {metric}: t-statistic = {t_stat:.2f}, p-value = {p_value:.4f} ({significance})")
            else:
                print(f"  {metric}: No hay suficientes datos para comparación")

    # Opcional: Guardar resultados en CSV
    output_csv = "/home/vant/Documentos/proyectos/HumanidadesDigitales/CorpusOrales/process/output/acoustic_prosodic_analysis.csv"
    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write("Lemma,Sexo,Metric,Mean,Median,Std\n")
        for (lemma, sexo), group in df_analysis.groupby(['lemma', 'sexo']):
            for metric in numeric_columns:
                mean = group[metric].mean()
                median = group[metric].median()
                std = group[metric].std()
                f.write(f"{lemma},{sexo},{metric},{mean:.2f},{median:.2f},{std:.2f}\n")
        f.write("\nLemma,Metric,t_statistic,p_value,Significance\n")
        for lemma, metrics in results_ttest.items():
            for metric, stats in metrics.items():
                t_stat, p_value = stats['t_stat'], stats['p_value']
                if t_stat is not None and p_value is not None:
                    significance = "Yes" if p_value < 0.05 else "No"
                    f.write(f"{lemma},{metric},{t_stat:.2f},{p_value:.4f},{significance}\n")

    print(f"Resultados guardados en {output_csv}")