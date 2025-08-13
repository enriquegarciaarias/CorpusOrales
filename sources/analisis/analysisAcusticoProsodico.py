from sources.common.common import processControl, logger, log_
from sources.common.utils import leeJson

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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

    outputPath = os.path.join(processControl.env['outputDir'], "acusticoProsodicoBasico.csv")
    with open(outputPath, 'w', encoding='utf-8') as f:
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

    log_("info", logger, f"Resultados guardados en {outputPath}")



def basicos(path):
    # Cargar JSON o usar sample
    data = leeJson(path)
    if not data:
        return
    # Extraer datos válidos
    rows = []
    for entry in data:
        query = entry.get('query', {})
        sexo = query.get('sexo')
        edad = query.get('edad')
        for analysis in entry.get('analysis', []):
            result = analysis.get('result', {})
            mean_pitch = result.get('mean_pitch')
            hnr = result.get('prosodia', {}).get('hnr')
            jitter_local = result.get('prosodia', {}).get('jitter_local')
            # Ignorar si cualquier valor es NaN o None
            if all(x is not None and not np.isnan(x) if isinstance(x, (float, int)) else x is not None for x in
                   [mean_pitch, hnr, jitter_local]):
                rows.append({
                    'sexo': sexo,
                    'edad': edad,
                    'mean_pitch': mean_pitch,
                    'hnr': hnr,
                    'jitter_local': jitter_local
                })

    # Crear DataFrame
    df = pd.DataFrame(rows)

    # Agrupar por edad y sexo, calcular medias
    grouped = df.groupby(['edad', 'sexo']).mean().reset_index()

    # Preparar datos para gráfico
    edades = sorted(grouped['edad'].unique())
    sexos = grouped['sexo'].unique()
    metrics = ['mean_pitch', 'hnr', 'jitter_local']

    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 8))

    bar_width = 0.35
    positions = np.arange(len(edades))

    for i, sexo in enumerate(sexos):
        subset = grouped[grouped['sexo'] == sexo]
        if not subset.empty:
            bottom = np.zeros(len(edades))
            for metric in metrics:
                values = []
                for edad in edades:
                    value = subset[subset['edad'] == edad][metric].values
                    values.append(value[0] if len(value) > 0 else 0)
                ax.bar(positions + i * bar_width, values, bar_width, bottom=bottom,
                       label=f'{metric} - {sexo}' if i == 0 else None)
                bottom += values

    ax.set_xlabel('Edad (Grupos: 1, 2, 3)')
    ax.set_ylabel('Valor Promedio')
    ax.set_title('Métricas Prosódicas por Edad y Sexo (Barras Apiladas por Métrica, Side-by-Side por Sexo)')
    ax.set_xticks(positions + bar_width / 2)
    ax.set_xticklabels(edades)
    ax.legend()

    # Guardar gráfico
    plt.savefig('/home/vant/Documentos/proyectos/HumanidadesDigitales/CorpusOrales/grafico_prosodia.png')
    print("Gráfico guardado en 'grafico_prosodia.png'")

    # Generar y guardar tabla
    table = grouped.pivot(index='edad', columns='sexo', values=metrics)
    table.to_csv('/home/vant/Documentos/proyectos/HumanidadesDigitales/CorpusOrales/resultados_tabla.csv')
    print("Tabla guardada en 'resultados_tabla.csv'")
    print("\nTabla de Resultados:")
    print(table)

    # Opcional: Mostrar tabla en consola con formato
    print("\nTabla Formateada:")
    print(table.to_string())


def processProsodicStats(resultsPath):
    # Cargar datos del JSON

    data = leeJson(resultsPath)
    if not data:
        return None
    # Aplanar estructura de datos en una lista de registros
    registros = []
    for item in data:
        # Extrae variables principales
        query = item.get('query', {})
        lemma = query.get("lemma")
        sexo = query.get("sexo")
        edad = query.get("edad")

        # Extrae resultados de audio (puede variar según la estructura de tu JSON)
        analisis = item.get("analysis", {})
        for analysis in analisis:
            values = analysis.get("result", {})

            registros.append({
                "lemma": lemma,
                "sexo": sexo,
                "edad": edad,
                "mean_pitch": values.get("mean_pitch"),
                "hnr": values.get("prosodia", {}).get("hnr"),
                "mean_energy": values.get("mean_energy"),
                "duration": values.get("duration"),
                "speech_rate": values.get("speech_rate")
            })

    # Crear DataFrame
    df = pd.DataFrame(registros)

    # Eliminar filas con valores nulos en variables clave
    df = df.dropna(subset=["mean_pitch", "hnr", "mean_energy", "duration", "speech_rate"])

    # Agrupar por lemma, sexo y edad
    stats_df = df.groupby(["lemma", "sexo", "edad"]).agg(
        mean_mean_pitch=('mean_pitch', 'mean'),
        median_mean_pitch=('mean_pitch', 'median'),
        std_mean_pitch=('mean_pitch', lambda x: x.std(ddof=0)),

        mean_hnr=('hnr', 'mean'),
        median_hnr=('hnr', 'median'),
        std_hnr=('hnr', lambda x: x.std(ddof=0)),

        mean_mean_energy=('mean_energy', 'mean'),
        median_mean_energy=('mean_energy', 'median'),
        std_mean_energy=('mean_energy', lambda x: x.std(ddof=0)),

        mean_duration=('duration', 'mean'),
        median_duration=('duration', 'median'),
        std_duration=('duration', lambda x: x.std(ddof=0)),

        mean_speech_rate=('speech_rate', 'mean'),
        median_speech_rate=('speech_rate', 'median'),
        std_speech_rate=('speech_rate', lambda x: x.std(ddof=0))
    ).reset_index()

    outputPath = os.path.join(processControl.env['outputDir'], "acusticoProsodico.csv")
    # Guardar resultados
    stats_df.to_csv(outputPath, index=False, encoding='utf-8')
    log_("info", logger, f"Estadísticas guardadas en: {outputPath}")
    return outputPath