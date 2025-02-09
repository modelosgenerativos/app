# --- Instalação de Bibliotecas ---
!pip install pingouin

# --- Importações ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
from google.colab import drive
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.express as px  # Para gráficos interativos (opcional)
# import sklearn.model_selection as ms  # Comentado por enquanto
# import sklearn.linear_model as lm  # Comentado por enquanto
# import sklearn.cluster as cluster  # Comentado por enquanto
# import sklearn.metrics as metrics  # Comentado por enquanto
# import nltk  # Comentado por enquanto
# import spacy  # Comentado por enquanto
# import language_tool_python  # Descomentar se for usar
# import gensim  # Comentado por enquanto
from IPython.display import display, Markdown
import pingouin as pg
# import xgboost as xgb  # Comentado por enquanto
# import shap  # Comentado por enquanto
# import streamlit as st  # Comentado por enquanto

# --- Configurações e Constantes ---
DEFAULT_PALETTE = "viridis"
FIGSIZE = (12, 8)

sns.set(style="darkgrid")
plt.rcParams.update({
    'figure.facecolor': 'black',
    'axes.facecolor': 'black',
    'axes.edgecolor': 'white',
    'axes.labelcolor': 'cyan',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'text.color': 'white',
    'grid.color': 'gray',
    'grid.linestyle': '--',
    'legend.facecolor': 'black',
    'legend.edgecolor': 'white',
    'figure.titlesize': 20,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

# --- Funções Auxiliares ---
def mount_google_drive():
    """Monta o Google Drive."""
    drive.mount('/content/drive')

def get_drive_path(relative_path):
    """Retorna o caminho completo no Drive."""
    return os.path.join('/content/drive/MyDrive', relative_path)

def ensure_directory_exists_on_drive(relative_path):
    """Cria diretório no Drive, se não existir."""
    drive_path = get_drive_path(relative_path)
    if not os.path.exists(drive_path):
        os.makedirs(drive_path, exist_ok=True)
        print(f"Diretório criado: {drive_path}")
    return drive_path

def save_fig(fig, filename, drive_folder_path):
    """Salva a figura no Drive."""
    try:
        filepath = os.path.join(drive_folder_path, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo: {filepath}")
    except Exception as e:
        print(f"Erro ao salvar '{filename}': {e}")
    finally:
        plt.close(fig)

def create_figure():
    """Cria uma figura com tamanho padrão."""
    return plt.figure(figsize=FIGSIZE)

# --- Funções de Visualização (Com as Novas Funções) ---

def plot_categorical_count(df, x_col, title, filename, drive_folder_path, hue=None):
    """Plota contagem de categorias."""
    try:
        if x_col not in df.columns:
            raise ValueError(f"Coluna '{x_col}' não encontrada.")
        fig = create_figure()
        sns.countplot(x=x_col, data=df, palette=DEFAULT_PALETTE, hue=hue)
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' '))
        plt.xticks(rotation=45, ha='right')
        if hue:
            plt.legend(title=hue.replace('_', ' '), loc='upper right')
        save_fig(fig, filename, drive_folder_path)
    except Exception as e:
        print(f"Erro em plot_categorical_count: {e}")

def plot_numeric_distribution(df, x_col, title, filename, drive_folder_path, hue=None):
    """Plota distribuição de variável numérica (histograma + KDE)."""
    try:
        if x_col not in df.columns:
            raise ValueError(f"Coluna '{x_col}' não encontrada.")
        fig = create_figure()
        sns.histplot(data=df, x=x_col, kde=True, hue=hue, palette=DEFAULT_PALETTE)
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' '))
        if hue:
            plt.legend(title=hue.replace('_', ' '), loc='upper right')
        save_fig(fig, filename, drive_folder_path)
    except Exception as e:
        print(f"Erro em plot_numeric_distribution: {e}")

def plot_boxplot(df, x_col, y_col, title, filename, drive_folder_path, hue=None):
    """Plota boxplots."""
    try:
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Colunas '{x_col}' ou '{y_col}' não encontradas.")
        fig = create_figure()
        sns.boxplot(x=x_col, y=y_col, data=df, palette=DEFAULT_PALETTE, hue=hue)
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' '))
        plt.ylabel(y_col.replace('_', ' '))
        if hue:
            plt.legend(title=hue.replace('_', ' '), loc='upper right')
        save_fig(fig, filename, drive_folder_path)
    except Exception as e:
        print(f"Erro em plot_boxplot: {e}")

def plot_correlation_heatmap(df, title, filename, drive_folder_path):
    """Plota heatmap de correlação."""
    try:
        fig = create_figure()
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap=DEFAULT_PALETTE, fmt=".2f", linewidths=.5)
        plt.title(title)
        save_fig(fig, filename, drive_folder_path)
    except Exception as e:
        print(f"Erro em plot_correlation_heatmap: {e}")

def plot_scatter(df, x_col, y_col, title, filename, drive_folder_path, hue=None, trendline=False):
    """Plota gráfico de dispersão com opção de linha de tendência."""
    try:
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Colunas '{x_col}' ou '{y_col}' não encontradas.")
        fig = create_figure()
        if trendline:
            sns.regplot(x=x_col, y=y_col, data=df, scatter_kws={'color': 'cyan'}, line_kws={'color': 'red'})
        else:
            sns.scatterplot(x=x_col, y=y_col, data=df, hue=hue, palette=DEFAULT_PALETTE)
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' '))
        plt.ylabel(y_col.replace('_', ' '))
        if hue:
            plt.legend(title=hue.replace('_', ' '), loc='upper right')
        save_fig(fig, filename, drive_folder_path)
    except Exception as e:
        print(f"Erro em plot_scatter: {e}")

def plot_violin(df, x_col, y_col, title, filename, drive_folder_path, hue=None):
    """Plota violin plots."""
    try:
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Colunas '{x_col}' ou '{y_col}' não encontradas.")
        fig = create_figure()
        sns.violinplot(x=x_col, y=y_col, data=df, palette=DEFAULT_PALETTE, hue=hue, split=(hue is not None))
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' '))
        plt.ylabel(y_col.replace('_', ' '))
        if hue:
            plt.legend(title=hue.replace('_', ' '), loc='upper right')
        save_fig(fig, filename, drive_folder_path)
    except Exception as e:
        print(f"Erro em plot_violin: {e}")

# --- Novas Funções de Visualização ---

def plot_grouped_barplot(df, x_col, y_col, hue_col, title, filename, drive_folder_path):
    """Plota barras agrupadas (média de y_col por x_col e hue_col)."""
    try:
        if not all(col in df.columns for col in [x_col, y_col, hue_col]):
            raise ValueError("Colunas necessárias não encontradas.")
        fig = create_figure()
        sns.barplot(x=x_col, y=y_col, hue=hue_col, data=df, palette=DEFAULT_PALETTE, errorbar='sd')
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' '))
        plt.ylabel(y_col.replace('_', ' '))
        plt.xticks(rotation=45, ha='right')
        plt.legend(title=hue_col.replace('_', ' '))
        save_fig(fig, filename, drive_folder_path)
    except Exception as e:
        print(f"Erro em plot_grouped_barplot: {e}")

def plot_scatter_colored(df, x_col, y_col, color_col, title, filename, drive_folder_path):
    """Dispersão com cores representando uma terceira variável."""
    try:
        if not all(col in df.columns for col in [x_col, y_col, color_col]):
            raise ValueError("Colunas necessárias não encontradas.")
        fig = create_figure()
        sns.scatterplot(x=x_col, y=y_col, data=df, hue=color_col, palette=DEFAULT_PALETTE, size=color_col)
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' '))
        plt.ylabel(y_col.replace('_', ' '))
        plt.legend(title=color_col.replace('_', ' '), loc='upper right')
        save_fig(fig, filename, drive_folder_path)
    except Exception as e:
        print(f"Erro em plot_scatter_colored: {e}")

def plot_stacked_histogram(df, x_col, hue_col, title, filename, drive_folder_path):
    """Histograma empilhado."""
    try:
        if not all(col in df.columns for col in [x_col, hue_col]):
            raise ValueError("Colunas necessárias não encontradas.")
        fig = create_figure()
        for group in df[hue_col].unique():
            sns.histplot(data=df[df[hue_col] == group], x=x_col, label=str(group), alpha=0.7, kde=False)
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' '))
        plt.ylabel("Contagem")
        plt.legend(title=hue_col.replace('_', ' '))
        save_fig(fig, filename, drive_folder_path)
    except Exception as e:
        print(f"Erro em plot_stacked_histogram: {e}")

# --- Funções de Análise Estatística ---
def perform_t_test(df, group_col, value_col, equal_var=False):
    """Realiza teste t de Student."""
    try:
        if group_col not in df.columns or value_col not in df.columns:
            raise ValueError(f"Colunas '{group_col}' ou '{value_col}' não encontradas.")

        groups = df[group_col].unique()
        if len(groups) != 2:
            raise ValueError("Teste t requer exatamente dois grupos.")

        group1 = df[df[group_col] == groups[0]][value_col]
        group2 = df[df[group_col] == groups[1]][value_col]
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
        print(f"Teste t para {value_col} entre {groups[0]} e {groups[1]}:")
        print(f"  t = {t_stat:.3f}, p = {p_value:.3f}")
        return t_stat, p_value
    except Exception as e:
        print(f"Erro em perform_t_test: {e}")
        return None, None

def perform_anova(df, group_col, value_col):
    """Realiza ANOVA de um fator."""
    try:
        if group_col not in df.columns or value_col not in df.columns:
            raise ValueError(f"Colunas '{group_col}' ou '{value_col}' não encontradas.")

        groups = df[group_col].unique()
        if len(groups) < 2:
            raise ValueError("ANOVA requer pelo menos dois grupos.")

        formula = f"{value_col} ~ C({group_col})"
        model = smf.ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(f"ANOVA para {value_col} entre grupos {groups}:")
        print(anova_table)
        return anova_table
    except Exception as e:
        print(f"Erro em perform_anova: {e}")
        return None

# --- Função Principal (main) ---
if __name__ == "__main__":
    try:
        mount_google_drive()
        graficos_drive_path = ensure_directory_exists_on_drive('graficos_modelos_generativos_final_v2')

        # --- Carregar Dados (Tabela Markdown) ---
        markdown_table = """
|estudante_id|grupo|habilidade_previa|tempo_tarefa|qualidade_escrita|resolucao_problemas|pensamento_critico|num_prompts|satisfacao|
|---|---|---|---|---|---|---|---|---|
|1|Com_MG|65|35|7|6|5|8|4|
|2|Sem_MG|72|48|6|5|4|0|3|
|3|Com_MG|58|28|8|7|6|12|5|
|4|Sem_MG|80|55|7|6|5|0|4|
|5|Com_MG|75|30|9|8|7|5|4|
|6|Sem_MG|63|49|6|6|5|0|4|
|7|Com_MG|79|36|8|7|6|9|3|
|8|Sem_MG|55|68|5|4|4|0|2|
|9|Com_MG|88|33|9|8|7|11|5|
|10|Sem_MG|67|41|7|5|6|0|3|
|11|Com_MG|52|39|7|6|5|7|4|
|12|Sem_MG|74|58|6|5|4|0|3|
|13|Com_MG|83|29|8|7|6|4|5|
|14|Sem_MG|59|71|4|6|3|0|4|
|15|Com_MG|69|31|9|8|7|10|3|
|16|Sem_MG|86|44|5|4|5|0|2|
|17|Com_MG|71|37|8|7|6|6|5|
|18|Sem_MG|54|63|7|5|4|0|3|
|19|Com_MG|78|26|7|6|5|8|4|
|20|Sem_MG|61|52|6|5|4|0|3|
|21|Com_MG|89|34|9|8|7|3|5|
|22|Sem_MG|57|47|5|6|3|0|4|
|23|Com_MG|73|38|8|7|6|12|3|
|24|Sem_MG|82|69|7|5|6|0|2|
|25|Com_MG|66|27|7|6|5|5|5|
|26|Sem_MG|50|59|6|5|4|0|3|
|27|Com_MG|76|32|9|8|7|9|4|
|28|Sem_MG|85|40|4|4|6|0|2|
|29|Com_MG|60|35|8|7|6|7|5|
|30|Sem_MG|70|74|7|5|4|0|3|
|31|Com_MG|87|30|7|6|5|4|4|
|32|Sem_MG|53|51|6|5|4|0|3|
|33|Com_MG|68|39|9|8|7|11|5|
|34|Sem_MG|77|45|5|6|3|0|4|
|35|Com_MG|84|33|8|7|6|6|3|
|36|Sem_MG|56|66|7|5|4|0|2|
|37|Com_MG|62|37|7|6|5|8|5|
|38|Sem_MG|81|58|6|5|4|0|3|
|39|Com_MG|75|28|9|8|7|3|4|
|40|Sem_MG|51|72|4|4|6|0|2|
        """
        df = pd.read_csv(io.StringIO(markdown_table), sep='|', index_col=1, skipinitialspace=True).iloc[1:].reset_index(drop = True)

        # --- Pré-processamento ---
        df.columns = df.columns.str.replace(' ', '')
        for col in ['habilidade_previa', 'tempo_tarefa', 'qualidade_escrita',
                    'resolucao_problemas', 'pensamento_critico', 'num_prompts', 'satisfacao']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(axis=1, how='all', inplace=True)

        # --- Análise Exploratória e Visualizações ---
        print(df.info())
        print(df.head())
        print(df.describe(include='all'))

        # --- Gráficos Anteriores ---
        plot_categorical_count(df, 'grupo', "1. Estudantes por Grupo", "1_contagem_grupos.png", graficos_drive_path)
        plot_numeric_distribution(df, 'habilidade_previa', "2. Habilidade Prévia", "2_dist_habilidade.png", graficos_drive_path, hue='grupo')
        plot_numeric_distribution(df, 'tempo_tarefa', "3. Tempo de Tarefa", "3_dist_tempo.png", graficos_drive_path, hue='grupo')
        plot_boxplot(df, 'grupo', 'qualidade_escrita', "4. Qualidade da Escrita (Boxplot)", "4_boxplot_qualidade.png", graficos_drive_path)
        plot_violin(df, 'grupo', 'qualidade_escrita', "5. Qualidade da Escrita (Violin)", "5_violin_qualidade.png", graficos_drive_path)
        plot_correlation_heatmap(df, "6. Correlação entre Variáveis", "6_heatmap_correlacao.png", graficos_drive_path)
        plot_scatter(df, 'habilidade_previa', 'qualidade_escrita', "7. Habilidade vs. Qualidade", "7_scatter_habilidade_qualidade.png", graficos_drive_path, hue='grupo', trendline=True)
        perform_t_test(df, 'grupo', 'qualidade_escrita')
        if 'satisfacao' in df.columns and df['satisfacao'].nunique() > 1:
            perform_anova(df, 'satisfacao', 'tempo_tarefa')

        # --- Novos Gráficos ---

        # 1. Barras Agrupadas: resolucao_problemas por grupo e satisfacao
        plot_grouped_barplot(df, 'grupo', 'resolucao_problemas', 'satisfacao',
                             "8. Resolução de Problemas por Grupo e Satisfação",
                             "8_grouped_barplot_resolucao_satisfacao.png", graficos_drive_path)

        # 2. Dispersão Colorida: tempo_tarefa vs. qualidade_escrita, cor por num_prompts
        plot_scatter_colored(df, 'tempo_tarefa', 'qualidade_escrita', 'num_prompts',
                              "9. Tempo vs. Qualidade (Cor por Prompts)",
                              "9_scatter_tempo_qualidade_prompts.png", graficos_drive_path)

        # 3. Histograma Empilhado: num_prompts por grupo
        plot_stacked_histogram(df, 'num_prompts', 'grupo',
                                "10. Distribuição de Prompts por Grupo",
                                "10_stacked_histogram_prompts_grupo.png", graficos_drive_path)

        print("Análise completa, gráficos gerados!")

    except Exception as e:
        print(f"Erro geral no processamento: {e}")