import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. Generate Sample Datasets from Different Domains
def generate_sample_data():
    # Computing Domain (e.g., algorithm performance)
    computing_data = pd.DataFrame({
        'Algorithm': ['A', 'B', 'C', 'D', 'E'],
        'Execution Time (ms)': np.random.normal(200, 50, 5),
        'Memory Usage (MB)': np.random.normal(120, 15, 5)
    })

    # Medical Domain (e.g., patient data)
    medical_data = pd.DataFrame({
        'Patient ID': range(1, 11),
        'Age': np.random.randint(20, 80, 10),
        'Blood Pressure': np.random.randint(110, 180, 10),
        'Cholesterol': np.random.randint(150, 250, 10)
    })

    # Social Science Domain (e.g., survey)
    social_data = pd.DataFrame({
        'Respondent ID': range(1, 11),
        'Happiness Score': np.random.uniform(1, 10, 10),
        'Income (k)': np.random.randint(20, 100, 10),
        'Education Level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 10)
    })

    return computing_data, medical_data, social_data

# 2. Define Statistical Analysis Function
def perform_statistics(df, numeric_columns):
    stats_dict = {}
    for col in numeric_columns:
        mode_val = stats.mode(df[col], keepdims=True).mode[0]  # Fixed for SciPy >= 1.9
        stats_dict[col] = {
            'Mean': np.mean(df[col]),
            'Median': np.median(df[col]),
            'Mode': mode_val,
            'Standard Deviation': np.std(df[col]),
            'Variance': np.var(df[col]),
            'Min': np.min(df[col]),
            'Max': np.max(df[col]),
        }
    return pd.DataFrame(stats_dict)

# 3. Plotting Functions
def plot_histograms(df, numeric_columns, title):
    for col in numeric_columns:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"{title}: Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

def plot_correlation(df, numeric_columns, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm')
    plt.title(f"{title}: Correlation Matrix")
    plt.show()

# 4. Main Program
def main():
    computing_data, medical_data, social_data = generate_sample_data()

    print("---------- Computing Domain ----------")
    comp_stats = perform_statistics(computing_data, ['Execution Time (ms)', 'Memory Usage (MB)'])
    print(comp_stats)
    plot_histograms(computing_data, ['Execution Time (ms)', 'Memory Usage (MB)'], "Computing")
    plot_correlation(computing_data, ['Execution Time (ms)', 'Memory Usage (MB)'], "Computing")

    print("\n---------- Medical Domain ----------")
    med_stats = perform_statistics(medical_data, ['Age', 'Blood Pressure', 'Cholesterol'])
    print(med_stats)
    plot_histograms(medical_data, ['Age', 'Blood Pressure', 'Cholesterol'], "Medical")
    plot_correlation(medical_data, ['Age', 'Blood Pressure', 'Cholesterol'], "Medical")

    print("\n---------- Social Sciences Domain ----------")
    social_numeric = ['Happiness Score', 'Income (k)']
    social_stats = perform_statistics(social_data, social_numeric)
    print(social_stats)
    plot_histograms(social_data, social_numeric, "Social Sciences")
    plot_correlation(social_data, social_numeric, "Social Sciences")

# ✅ Correct way to run the script
if __name__ == "__main__":
    main()
