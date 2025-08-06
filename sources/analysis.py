from sources.common.common import processControl, logger, log_
from sources.common.utils import leeJson

import json
import pandas as pd
import os
from collections import defaultdict

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

    # Convertir a DataFrame
    df = pd.DataFrame(header_data)

    # Seleccionar columnas para análisis
    categorical_columns = ['lemma', 'sexo', 'edad']
    numeric_columns = ['Ejemplos', 'Entrevistas']

    # Generar tablas de frecuencia para columnas categóricas
    frequency_tables_categorical = {}
    for column in categorical_columns:
        freq_table = df[column].value_counts()
        frequency_tables_categorical[column] = freq_table

    # Generar sumas acumuladas para columnas numéricas por categoría
    sum_tables_numeric = {}
    for column in numeric_columns:
        sum_table = df.groupby(categorical_columns)[column].sum()
        sum_tables_numeric[column] = sum_table

    # Mostrar las tablas de frecuencia (conteo)
    print("Tablas de Frecuencia Categóricas (Conteo de Apariciones):")
    for column, table in frequency_tables_categorical.items():
        print(f"\nFrecuencia para {column}:")
        print(table)
        print("-" * 50)

    # Mostrar las tablas de sumas acumuladas
    print("\nTablas de Sumas Acumuladas por Categoría:")
    for column, table in sum_tables_numeric.items():
        print(f"\nSuma acumulada para {column}:")
        print(table)
        print("-" * 50)

    # Guardar las tablas en un archivo CSV
    output_csv = os.path.join(processControl.env['outputDir'], "headerFrequencyTables.csv")

    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write("Type,Column,Category/Value,Frequency/Sum\n")
        for column, table in frequency_tables_categorical.items():
            for value, freq in table.items():
                f.write(f"Categorical,{column},{value},{freq}\n")
        for column, table in sum_tables_numeric.items():
            for categories, value in table.items():
                if isinstance(categories, tuple):
                    cat_str = '-'.join(str(c) for c in categories if c != "unknown")
                else:
                    cat_str = str(categories)
                f.write(f"Numeric,{column},{cat_str},{value}\n")

    print(f"Tablas de frecuencia guardadas en {output_csv}")

def tablasFrecuencia(resultsPath):
    frecuenciaBase(resultsPath)
    return True

    results = leeJson(resultsPath)

    # Extraer todos los resultados de análisis
    all_results = []
    for entry in results:
        query = entry.get("query", {})
        lemma = query.get("lemma", "unknown")
        sexo = query.get("sexo", "unknown")
        edad = query.get("edad", "unknown")
        if "analysis" in entry:
            for analysis in entry["analysis"]:
                result = analysis["result"]
                # Desanidar el diccionario prosodia
                prosodia = result.get("prosodia", {})
                result.update({
                    "hnr": prosodia.get("hnr"),
                    "jitter_local": prosodia.get("jitter_local"),
                    "shimmer_local": prosodia.get("shimmer_local")
                })
                # Derivar categoría desde audioFile como validación (opcional)
                audio_file = analysis["audioFile"]
                file_parts = os.path.basename(os.path.dirname(audio_file)).split('-')
                derived_sexo = file_parts[1] if len(file_parts) >= 2 else "unknown"
                derived_edad = file_parts[2] if len(file_parts) >= 3 else "unknown"
                derived_lemma = "bar" if "bar" in audio_file else "iglesia" if "iglesia" in audio_file else "unknown"
                # Crear combinación de género
                genre_combination = f"{lemma}-{sexo}-{edad}"
                # Usar query como fuente principal, validar con derivación
                result.update({
                    "lemma": lemma,
                    "sexo": sexo,
                    "edad": edad,
                    "derived_sexo": derived_sexo,
                    "derived_edad": derived_edad,
                    "derived_lemma": derived_lemma,
                    "genre_combination": genre_combination
                })
                all_results.append(result)

    # Convertir a DataFrame
    df = pd.DataFrame(all_results)

    # Seleccionar columnas para análisis
    numeric_columns = ['mean_pitch', 'hnr', 'mean_energy', 'duration', 'speech_rate', 'jitter_local', 'shimmer_local']
    categorical_columns = ['lemma', 'sexo', 'edad', 'derived_lemma', 'derived_sexo', 'derived_edad',
                           'genre_combination']

    # Rellenar NaN en columnas numéricas con 0 para análisis de frecuencia
    df_numeric = df[numeric_columns].fillna(0)

    # Generar tablas de frecuencia para columnas numéricas
    frequency_tables_numeric = {}
    for column in numeric_columns:
        freq_table = pd.cut(df_numeric[column], bins=10).value_counts().sort_index()
        frequency_tables_numeric[column] = freq_table

    # Generar tablas de frecuencia para columnas categóricas
    frequency_tables_categorical = {}
    for column in categorical_columns:
        freq_table = df[column].value_counts()
        frequency_tables_categorical[column] = freq_table

    # Calcular estadísticas descriptivas por categoría
    descriptive_stats = df.groupby(['lemma', 'sexo', 'edad'])[numeric_columns].agg(['mean', 'median', 'std']).round(2)

    # Mostrar las tablas de frecuencia
    print("Tablas de Frecuencia Numéricas:")
    for column, table in frequency_tables_numeric.items():
        print(f"\nFrecuencia para {column}:")
        print(table)
        print("-" * 50)

    print("\nTablas de Frecuencia Categóricas:")
    for column, table in frequency_tables_categorical.items():
        print(f"\nFrecuencia para {column}:")
        print(table)
        print("-" * 50)

    # Mostrar estadísticas descriptivas
    print("\nEstadísticas Descriptivas por Categoría (lemma, sexo, edad):")
    print(descriptive_stats)

    # Opcional: Guardar las tablas en un archivo CSV para análisis posterior
    output_csv = os.path.join(processControl.env['outputDir'], "frequencyTables.csv")

    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write("Type,Column,Range/Value,Frequency\n")
        for column, table in frequency_tables_numeric.items():
            for range_, freq in table.items():
                f.write(f"Numeric,{column},{range_},{freq}\n")
        for column, table in frequency_tables_categorical.items():
            for value, freq in table.items():
                f.write(f"Categorical,{column},{value},{freq}\n")

    log_("info", logger, f"Tablas de frecuencia guardadas en {output_csv}")