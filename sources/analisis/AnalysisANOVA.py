from sources.common.common import processControl, logger, log_

import os
import pandas as pd
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

def peachSpeechAnova(stats_csv_path):
    # === 1. Leer datos y preparar ===
    df = pd.read_csv(stats_csv_path, sep=";", decimal=",")
    df.columns = [c.strip() for c in df.columns]  # limpiar nombres de columnas

    selected_features = ["mean_mean_pitch", "mean_speech_rate"]

    # Comprobar que están las columnas
    missing_cols = [col for col in selected_features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas: {missing_cols}")

    # Inicializar listas para acumular resultados
    anova_records = []
    tukey_records = []

    AnovaShow(df, selected_features, anova_records, tukey_records)

    # Guardar CSVs
    anova_df = pd.DataFrame(anova_records)
    tukey_df = pd.DataFrame(tukey_records)

    anova_out = os.path.join(processControl.env['outputDir'], "anova_results.csv")
    tukey_out = os.path.join(processControl.env['outputDir'], "tukey_results.csv")

    anova_df.to_csv(anova_out, index=False, encoding="utf-8")
    tukey_df.to_csv(tukey_out, index=False, encoding="utf-8")

    log_("info", logger, f"[OK] Resultados ANOVA guardados en: {anova_out}")
    print(f"[OK] Resultados Tukey guardados en: {tukey_out}")


def AnovaShow(df, selected_features, anova_records, tukey_records):

    for feature in selected_features:
        for group_col in ["sexo", "edad"]:
            aov = pg.anova(data=df, dv=feature, between=group_col, detailed=True)
            print(f"\n=== ANOVA {feature} por {group_col} ===")
            print(aov)

            # Guardar en lista
            for _, row in aov.iterrows():
                anova_records.append({
                    "feature": feature,
                    "group": group_col,
                    "Source": row["Source"],
                    "SS": row["SS"],
                    "DF": row["DF"],
                    "MS": row["MS"],
                    "F": row["F"],
                    "p-unc": row["p-unc"],
                    "np2": row["np2"]
                })

            if aov["p-unc"].iloc[0] < 0.05:
                tukey = pairwise_tukeyhsd(endog=df[feature], groups=df[group_col], alpha=0.05)
                print("\nPost-hoc Tukey:")
                print(tukey.summary())

                # Guardar Tukey en lista
                tukey_df_tmp = pd.DataFrame(
                    tukey.summary().data[1:],  # sin header
                    columns=tukey.summary().data[0]  # con header
                )
                tukey_df_tmp["feature"] = feature
                tukey_df_tmp["group"] = group_col
                tukey_records.extend(tukey_df_tmp.to_dict("records"))

                plot_with_significance(df, feature, group_col, tukey, f"{feature} por {group_col} (Tukey HSD)")
            else:
                plotStandard(df, group_col, feature)


def plotStandard(df, group_col, feature):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x=group_col, y=feature)
    plt.title(f"{feature} por {group_col} (no sig.)")
    plt.tight_layout()
    outputPath = os.path.join(processControl.env['outputDir'], f"{feature}_{group_col}_plotStandard.png")
    plt.savefig(outputPath, dpi=300)
    plt.close()


def plot_with_significance(df, feature, group_col, tukey_result, title):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x=group_col, y=feature)
    plt.title(title)

    max_y = df[feature].max()
    offset = (df[feature].max() - df[feature].min()) * 0.1
    ypos = max_y + offset

    for i, (g1, g2, p_val) in enumerate(zip(
            tukey_result._multicomp.pairindices[0],
            tukey_result._multicomp.pairindices[1],
            tukey_result.pvalues
    )):
        if p_val < 0.05:
            x1, x2 = g1, g2
            plt.plot([x1, x1, x2, x2], [ypos, ypos + offset, ypos + offset, ypos], lw=1.5, c='black')
            plt.text((x1 + x2) / 2, ypos + offset, f"p={p_val:.3f}", ha='center', va='bottom')
            ypos += offset * 1.5

    plt.tight_layout()
    outputPath = os.path.join(processControl.env['outputDir'], f"{feature}_{group_col}_plotSignificance.png")
    plt.savefig(outputPath, dpi=300)
    plt.close()

