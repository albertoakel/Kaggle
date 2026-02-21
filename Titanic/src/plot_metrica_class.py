#plot_metrica_class.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.stats import ttest_rel
from sklearn.metrics import roc_curve, roc_auc_score,confusion_matrix, accuracy_score


color_palette21 = [
    "#004C4C", "#006666", "#008080", "#199191", "#29A3A3",
    "#40B5B5", "#55C7C7", "#66D9D9", "#80ECEC", "#99FFFF",
    "#FFD580", "#FFC460", "#FFB240", "#FFA020", "#FF8E00",
    "#FF7C00", "#FF6400", "#FF4C00", "#FF3300", "#FF1A00", "#FF0000"]

def gerar_relatorio_estatistico(models_list, X_train, y_train, X_test, y_test):
    """
    Gera relatório estatístico completo de performance, estabilidade e significância
    entre múltiplos modelos binários.

    models_list: lista de tuplas
        (nome, pipeline, cv_roc_scores, cv_acc_scores, y_probs_test, best_threshold)
    """

    # ===============================
    # Funções auxiliares
    # ===============================
    def check_sig(p):
        return "SIM" if p < 0.05 else "NÃO"

    # ===============================
    # Tabela comparativa de métricas
    # ===============================
    results_data = []

    for name, model, s_roc, s_acc, probs, thresh in models_list:
        test_roc = roc_auc_score(y_test, probs)
        test_acc_std = accuracy_score(y_test, probs > 0.5)
        test_acc_opt = accuracy_score(y_test, probs > thresh)

        results_data.append({
            'Modelo': name,
            'CV ROC-AUC': f"{s_roc.mean():.4f} ± {s_roc.std():.4f}",
            'CV ACC': f"{s_acc.mean():.4f} ± {s_acc.std():.4f}",
            'Test ROC-AUC': f"{test_roc:.4f}",
            'Test ACC (0.5)': f"{test_acc_std:.4f}",
            'Best Thresh': f"{thresh:.3f}",
            'Test ACC (Opt)': f"{test_acc_opt:.4f}"
        })

    df_results = pd.DataFrame(results_data)

    print(f"{'='*95}")
    print(f"{'RELATÓRIO DE DESEMPENHO E ESTABILIDADE ESTATÍSTICA':^95}")
    print(f"{'='*95}")
    print(df_results.to_string(index=False, justify='center', col_space=15))

    # ===============================
    # Testes Estatísticos Pareados
    # ===============================
    print(f"\n{'='*95}")
    print(f"{'ANÁLISE DE SIGNIFICÂNCIA ESTATÍSTICA (T-TEST PAREADO)':^95}")
    print(f"{'='*95}")

    for i in range(1, len(models_list)):
        for j in range(i + 1, len(models_list)):
            t, p = ttest_rel(models_list[i][2], models_list[j][2])
            print(f"{models_list[i][0]} vs {models_list[j][0]}: "
                  f"p-value = {p:.4f} | Diferença Significativa? {check_sig(p)}")

    # ===============================
    # Identificação do vencedor
    # ===============================
    df_results['ROC_Numeric'] = df_results['Test ROC-AUC'].astype(float)
    best_idx = df_results['ROC_Numeric'].idxmax()
    vencedor = df_results.iloc[best_idx]

    winner=[models_list[best_idx][0], models_list[best_idx][1]]
    baseline = df_results.iloc[0]
    ganho_roc = vencedor['ROC_Numeric'] - float(baseline['Test ROC-AUC'])

    # Significância vencedor vs baseline (ACC CV)
    t_stat, p_val = ttest_rel(models_list[best_idx][3], models_list[0][3])
    sig_text = (
        f"estatisticamente significativa ({p_val:.4f} < 0.05)"
        if p_val < 0.05 else
        f"não significativa ({p_val:.4f} > 0.05)"
    )


    print(f"\n{'='*95}")
    print(f"{'CONCLUSÃO TÉCNICA AUTOMÁTICA':^95}")
    print(f"{'='*95}")

    print(f"1. VENCEDOR: {vencedor['Modelo']}")
    print(f"   - Ganho real sobre o Baseline: {ganho_roc:+.4f} em Test ROC-AUC.")

    print(f"\n2. ESTABILIDADE E SIGNIFICÂNCIA:")
    print(f"   - A melhoria em relação ao Baseline é {sig_text}.")
    print(f"   - Threshold otimizado: {vencedor['Best Thresh']}")
    print(f"   - ACC padrão: {vencedor['Test ACC (0.5)']}")
    print(f"   - ACC otimizada: {vencedor['Test ACC (Opt)']}")

    # ===============================
    # Overfitting / Generalização
    # ===============================
    cv_roc_mean = float(vencedor['CV ROC-AUC'].split(' ± ')[0])
    diff_cv_test = abs(cv_roc_mean - vencedor['ROC_Numeric'])
    status_fit = "ALTA" if diff_cv_test < 0.03 else "MODERADA"

    print(f"\n3. CONFIANÇA DO MODELO:")
    print(f"   - Aderência CV vs Teste: {status_fit} (Δ = {diff_cv_test:.4f})")

    thresh = float(vencedor['Best Thresh'])
    if thresh < 0.45:
        print("   - Estratégia: modelo AGRESSIVO (threshold baixo).")
    elif thresh > 0.55:
        print("   - Estratégia: modelo CONSERVADOR (threshold alto).")
    else:
        print("   - Estratégia: equilíbrio próximo a 0.5.")

    print("\n#Processo finalizado em:", time.strftime("%H:%M:%S"))

    return df_results.sort_values(by='Test ROC-AUC', ascending=False),winner



def gerar_relatorio_estatistico2(models_list, X_train, y_train, X_test, y_test, criterio="roc_auc"):
    """
    criterio: "roc_auc" ou "acc"
    """

    from scipy.stats import ttest_rel
    import pandas as pd
    import time

    def check_sig(p):
        return "SIM" if p < 0.05 else "NÃO"

    results_data = []

    for name, model, s_roc, s_acc, probs, thresh in models_list:

        test_roc = roc_auc_score(y_test, probs)
        test_acc_std = accuracy_score(y_test, probs > 0.5)
        test_acc_opt = accuracy_score(y_test, probs > thresh)

        results_data.append({
            'Modelo': name,
            'CV ROC Mean': s_roc.mean(),
            'CV ROC Std': s_roc.std(),
            'CV ACC Mean': s_acc.mean(),
            'CV ACC Std': s_acc.std(),
            'Test ROC-AUC': test_roc,
            'Test ACC (0.5)': test_acc_std,
            'Best Thresh': thresh,
            'Test ACC (Opt)': test_acc_opt
        })

    df_results = pd.DataFrame(results_data)

    print(f"{'='*95}")
    print(f"{'RELATÓRIO DE DESEMPENHO E ESTABILIDADE ESTATÍSTICA':^95}")
    print(f"{'='*95}")
    print(df_results.round(4).to_string(index=False))

    print(f"\n{'='*95}")
    print(f"{'ANÁLISE DE SIGNIFICÂNCIA ESTATÍSTICA (T-TEST PAREADO)':^95}")
    print(f"{'='*95}")

    # Seleção do vetor correto para teste estatístico
    for i in range(1, len(models_list)):
        for j in range(i + 1, len(models_list)):
            if criterio == "roc_auc":
                t, p = ttest_rel(models_list[i][2], models_list[j][2])
            else:
                t, p = ttest_rel(models_list[i][3], models_list[j][3])

            print(f"{models_list[i][0]} vs {models_list[j][0]}: "
                  f"p-value = {p:.4f} | Diferença Significativa? {check_sig(p)}")

    # ===============================
    # Definição do vencedor
    # ===============================
    if criterio == "roc_auc":
        metric_col = "Test ROC-AUC"
        cv_mean_col = "CV ROC Mean"
    else:
        metric_col = "Test ACC (Opt)"
        cv_mean_col = "CV ACC Mean"

    best_idx = df_results[metric_col].idxmax()
    vencedor = df_results.iloc[best_idx]
    winner = [models_list[best_idx][0], models_list[best_idx][1]]

    baseline = df_results.iloc[0]
    ganho = vencedor[metric_col] - baseline[metric_col]

    # Significância vencedor vs baseline
    if criterio == "roc_auc":
        t_stat, p_val = ttest_rel(models_list[best_idx][2], models_list[0][2])
    else:
        t_stat, p_val = ttest_rel(models_list[best_idx][3], models_list[0][3])

    sig_text = (
        f"estatisticamente significativa ({p_val:.4f} < 0.05)"
        if p_val < 0.05 else
        f"não significativa ({p_val:.4f} > 0.05)"
    )

    print(f"\n{'='*95}")
    print(f"{'CONCLUSÃO TÉCNICA AUTOMÁTICA':^95}")
    print(f"{'='*95}")

    print(f"1. VENCEDOR: {vencedor['Modelo']}")
    print(f"   - Ganho real sobre o Baseline: {ganho:+.4f} em {metric_col}.")

    print(f"\n2. ESTABILIDADE E SIGNIFICÂNCIA:")
    print(f"   - A melhoria é {sig_text}.")
    print(f"   - Threshold otimizado: {vencedor['Best Thresh']:.3f}")
    print(f"   - ACC padrão: {vencedor['Test ACC (0.5)']:.4f}")
    print(f"   - ACC otimizada: {vencedor['Test ACC (Opt)']:.4f}")

    # ===============================
    # Overfitting
    # ===============================
    cv_mean = vencedor[cv_mean_col]
    diff_cv_test = abs(cv_mean - vencedor[metric_col])
    status_fit = "ALTA" if diff_cv_test < 0.03 else "MODERADA"

    print(f"\n3. CONFIANÇA DO MODELO:")
    print(f"   - Aderência CV vs Teste: {status_fit} (Δ = {diff_cv_test:.4f})")

    thresh = vencedor['Best Thresh']
    if thresh < 0.45:
        print("   - Estratégia: modelo AGRESSIVO (threshold baixo).")
    elif thresh > 0.55:
        print("   - Estratégia: modelo CONSERVADOR (threshold alto).")
    else:
        print("   - Estratégia: equilíbrio próximo a 0.5.")

    print("\n#Processo finalizado em:", time.strftime("%H:%M:%S"))

    return df_results.sort_values(by=metric_col, ascending=False), winner



def model_evaluation_grid(
        models_list,
        X_test,
        y_test,
        best_model_pipeline,
        best_model_name,
        model_step='model',
        preprocess_step='preprocess'
):
    """
    Painel científico unificado para comparação de modelos:
    - Curvas ROC (destaque principal)
    - Estabilidade (Boxplot CV Accuracy)
    - Accuracy × Threshold
    - Matrizes de confusão (grid 2x2 padronizado)
    - Feature importance (somente melhor modelo)
    """

    # ======================================================
    # 1. ROC CURVES — DESTAQUE
    # ======================================================
    fig = plt.figure(figsize=(18, 22))
    gs = fig.add_gridspec(
        4, 2,
        height_ratios=[1.4, 1.0, 1.6, 1.4],
        hspace=0.35, wspace=0.25
    )

    ax_roc = fig.add_subplot(gs[0, :])

    cores = [color_palette21[0], color_palette21[4], color_palette21[12], color_palette21[18]]

    for idx, (name, _, _, _, y_prob, _) in enumerate(models_list):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax_roc.plot(fpr, tpr, color=cores[idx], linewidth=2, label=f'{name} (AUC={auc:.3f})')

    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    ax_roc.set_title('Comparativo Global — Curvas ROC', fontsize=20, weight='bold')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(fontsize=11)
    ax_roc.grid(alpha=0.3)

    # ======================================================
    # 2. ESTABILIDADE — BOXPLOT CV
    # ======================================================
    ax_box = fig.add_subplot(gs[1, 0])

    cv_scores = [m[3] for m in models_list]
    names = [m[0] for m in models_list]

    ax_box.boxplot(cv_scores, labels=names)
    ax_box.set_title('Estabilidade Estatística (10-Fold CV Accuracy)', fontsize=14, weight='bold')
    ax_box.set_ylabel('Accuracy')
    ax_box.grid(axis='y', linestyle='--', alpha=0.6)

    # ======================================================
    # 3. ACCURACY × THRESHOLD
    # ======================================================
    ax_thr = fig.add_subplot(gs[1, 1])

    thresholds = np.linspace(0.05, 0.95, 60)

    for idx, (name, _, _, _, probs, _) in enumerate(models_list):
        accs = [accuracy_score(y_test, probs >= t) for t in thresholds]
        ax_thr.plot(thresholds, accs, color=cores[idx], linewidth=2, label=name)

    ax_thr.set_title('Accuracy × Threshold (Curvas de Decisão)', fontsize=14, weight='bold')
    ax_thr.set_xlabel('Threshold')
    ax_thr.set_ylabel('Accuracy')
    ax_thr.legend(fontsize=10)
    ax_thr.grid(alpha=0.3)

    plt.show()

    # ======================================================
    # 4. MATRIZES DE CONFUSÃO — GRID 2x2
    # ======================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    fig.suptitle(
        'Análise Comparativa — Matrizes de Confusão (Normalizadas)',
        fontsize=18, y=1.02
    )

    for i, (name, model, *_) in enumerate(models_list):
        if i >= 4:
            break

        y_pred = model.predict(X_test)
        cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
        acc = accuracy_score(y_test, y_pred)

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.1%',
            ax=axes[i],
            cmap='GnBu',
            cbar=(i in [1, 3]),
            vmin=0, vmax=1,
            annot_kws={"size": 12, "weight": "bold"},
            linewidths=0.5,
            linecolor='gray'
        )

        axes[i].set_title(f'{name}\nAcc: {acc:.2%}', fontsize=14)
        axes[i].set_xticklabels(['Não Sobrev.', 'Sobrev.'])
        axes[i].set_yticklabels(['Não Sobrev.', 'Sobrev.'], rotation=0)
        axes[i].set_xlabel('Previsto')
        axes[i].set_ylabel('Real')

    plt.tight_layout()
    plt.show()

    # ======================================================
    # 5. FEATURE IMPORTANCE — MELHOR MODELO
    # ======================================================
    # importances = (
    #         best_model_pipeline
    #         .named_steps[model_step]
    #         .feature_importances_ * 100
    # )

    raw_importances = (
        best_model_pipeline
        .named_steps[model_step]
        .feature_importances_
    )

    importances = 100 * raw_importances / raw_importances.sum()

    try:
        feature_names = (
            best_model_pipeline
            .named_steps[preprocess_step]
            .get_feature_names_out()
        )
    except:
        X_tmp = best_model_pipeline.named_steps[preprocess_step].transform(X_test.iloc[:1])
        feature_names = (
            X_tmp.columns if hasattr(X_tmp, 'columns')
            else [f'Feature {i}' for i in range(len(importances))]
        )

    df_imp = (
        pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        .sort_values('Importance', ascending=False)
    )

    plt.figure(figsize=(11, 8))
    ax = sns.barplot(
        data=df_imp,
        x='Importance',
        y='Feature',
        palette=sns.color_palette('GnBu_r', n_colors=len(df_imp))
    )

    for p in ax.patches:
        width = p.get_width()
        ax.annotate(
            f'{width:.1f}%',
            (width + 0.3, p.get_y() + p.get_height() / 2),
            va='center', fontsize=10
        )

    plt.title(
        f'Feature Importance Relativa (%) — {best_model_name}',
        fontsize=16, pad=20
    )
    plt.xlabel('Contribuição para Redução de Impureza (Gini) [%]')
    plt.ylabel('Atributos')
    plt.xlim(0, df_imp['Importance'].max() * 1.15)
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
    print("\n#Processo finalizado em:", time.strftime("%H:%M:%S"))

    return df_imp
