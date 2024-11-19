import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from scipy.optimize import linear_sum_assignment
from sklearn.utils import resample
from google.colab import auth
from google.auth import default
import gspread
from google.colab import drive

drive.mount('/content/drive', force_remount=True)

def authenticate_google_drive():
    auth.authenticate_user()
    creds, _ = default()
    gspread_client = gspread.authorize(creds)
    return gspread_client

def load_data_from_sheets(sheet_url):
    gspread_client = authenticate_google_drive()
    sheet = gspread_client.open_by_url(sheet_url).sheet1
    data = pd.DataFrame(sheet.get_all_records())
    return data

def load_and_preprocess(data):
    personality_cols = ['Extroversao', 'Amabilidade', 'Escrupulo', 'Neuroticismo', 'Abertura_A_Experiencia']
    dating_cols = ['Impulso_Proprio', 'Impulso_Casal', 'Sociosexualidade_Geral', 'Satisfacao_Individual']
    X = data[personality_cols + dating_cols].apply(pd.to_numeric, errors='coerce')
    return X, pd.to_numeric(data['Sexo'], errors='coerce')

def determine_factors(X):
    fa = FactorAnalysis()
    fa.fit(X)
    ev, _ = np.linalg.eig(np.corrcoef(X, rowvar=False))
    return sum(ev > 1)

def extract_and_rotate(X, n_factors):
    fa = FactorAnalysis(n_components=n_factors)
    fa.fit(X)
    loadings = pd.DataFrame(
        fa.components_.T,
        columns=[f'Factor{i+1}' for i in range(n_factors)],
        index=X.columns
    )
    return loadings

def calculate_compatibility(X_female, X_male):
    scaler = StandardScaler()
    X_female_scaled = scaler.fit_transform(X_female)
    X_male_scaled = scaler.transform(X_male)
    
    factor = FactorAnalysis(n_components=3)
    factor.fit(np.vstack([X_female_scaled, X_male_scaled]))
    
    female_scores = factor.transform(X_female_scaled)
    male_scores = factor.transform(X_male_scaled)
    
    compatibility = np.zeros((len(X_female), len(X_male)))
    for i in range(len(female_scores)):
        for j in range(len(male_scores)):
            compatibility[i,j] = 1 / (1 + np.sqrt(np.sum((female_scores[i] - male_scores[j])**2)))
    
    return pd.DataFrame(compatibility)

def optimal_matching(compatibility):
    row_ind, col_ind = linear_sum_assignment(-compatibility.values)
    matches = pd.DataFrame({
        'Female_Index': row_ind,
        'Male_Index': col_ind,
        'Compatibility_Score': compatibility.values[row_ind, col_ind]
    })
    return matches.sort_values('Compatibility_Score', ascending=False)

def bootstrap(data, n_iterations=1000):
    bootstrap_results = [resample(data, n_samples=len(data), replace=True) for _ in range(n_iterations)]
    return bootstrap_results

def save_plot(fig, filename):
    drive_path = '/content/drive/My Drive/graficos/'
    if not os.path.exists(drive_path):
        os.makedirs(drive_path)
        
    path = os.path.join(drive_path, filename)
    fig.savefig(path)
    print(f"Gráfico salvo em: {path}")

def visualize_results(compatibility, matches, loadings, bootstrap_samples):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(compatibility, cmap='viridis', ax=ax)
    ax.set_title('Matriz de Compatibilidade de Relacionamento')
    save_plot(fig, 'compatibility_matrix.png')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(loadings, cmap='RdBu', center=0, ax=ax)
    ax.set_title('Cargas Fatoriais')
    save_plot(fig, 'factor_loadings.png')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot([sample['Extroversao'].mean() for sample in bootstrap_samples], ax=ax, kde=True)
    ax.set_title('Distribuição da Extroversão após Bootstrap')
    save_plot(fig, 'extroversao_bootstrap_histogram.png')

def main():
    sheet_url = 'https://docs.google.com/spreadsheets/d/1CNNFeecI-prP06vzS3xnQ8jj3ZY8wABxL4R8BsG35_s/edit?gid=1173412179#gid=1173412179'
    
    data = load_data_from_sheets(sheet_url)
    
    X, sex = load_and_preprocess(data)
    
    n_factors = determine_factors(X)
    
    loadings = extract_and_rotate(X, n_factors)
    
    X_female = X[sex == 0].dropna()
    X_male = X[sex == 1].dropna()
    
    compatibility = calculate_compatibility(X_female, X_male)
    
    matches = optimal_matching(compatibility)
    
    bootstrap_samples = bootstrap(X)
    
    visualize_results(compatibility, matches, loadings, bootstrap_samples)
    
    return {
        'loadings': loadings,
        'compatibility_matrix': compatibility,
        'optimal_matches': matches
    }

if __name__ == "__main__":
    results = main()