# LAB 2 ASSIGNMENT

import numpy as np
import pandas as pd
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

file_path = "/content/Lab Session Data.xlsx"

# Q1: Extract matrix info 
def extract_matrix_details(path, sheet_index):
    df = pd.read_excel(path, sheet_name=sheet_index).dropna(axis=1)

    features = df.iloc[:, 1:-1].values
    targets = df.iloc[:, -1].values.reshape(-1, 1)

    shape = features.shape
    num_vectors = shape[0]
    matrix_rank = np.linalg.matrix_rank(features)
    cost_estimate = np.linalg.pinv(features) @ targets

    return features, targets, shape, num_vectors, matrix_rank, cost_estimate

# Q2: Classification based on Price
def categorize_products(path, sheet_index):
    df = pd.read_excel(path, sheet_name=sheet_index).dropna(axis=1)

    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)

    category = ["Rich" if val > 200 else "Poor" for val in y]

    estimated = np.linalg.pinv(X) @ y
    predicted = X @ estimated

    return category, y, predicted

#  Q1 & Q2 Output 
result1 = extract_matrix_details(file_path, 0)
print("Matrix A:\n", result1[0])
print("Vector C:\n", result1[1])
print("Shape:", result1[2])
print("Total Vectors:", result1[3])
print("Matrix Rank:", result1[4])
print("Estimated Cost:\n", result1[5])

result2 = categorize_products(file_path, 0)
print("Category Labels:", result2[0])
print("Actual Prices:\n", result2[1])
print("Predicted Prices:\n", result2[2])

# Q3: Stats & Probabilities 
def compute_price_stats(path, sheet_index):
    df = pd.read_excel(path, sheet_name=sheet_index).dropna(axis=1)
    price_data = df["Price"].values
    return statistics.mean(price_data), statistics.variance(price_data)

def analyze_wednesday(path, sheet_index, overall_avg):
    df = pd.read_excel(path, sheet_name=sheet_index).dropna(axis=1)
    wed_prices = df[df["Day"] == "Wed"]["Price"].values
    mean_wed = statistics.mean(wed_prices)
    return mean_wed, mean_wed - overall_avg

def analyze_april(path, sheet_index, overall_avg):
    df = pd.read_excel(path, sheet_name=sheet_index).dropna(axis=1)
    apr_prices = df[df["Month"] == "Apr"]["Price"].values
    mean_apr = statistics.mean(apr_prices)
    return mean_apr, mean_apr - overall_avg

def prob_of_loss(path, sheet_index):
    df = pd.read_excel(path, sheet_name=sheet_index).dropna(axis=1)
    return sum(df["Chg%"] < 0) / len(df["Chg%"])

def prob_profit_on_wed(path, sheet_index):
    df = pd.read_excel(path, sheet_name=sheet_index).dropna(axis=1)
    return sum(df[df["Day"] == "Wed"]["Chg%"] > 0) / len(df[df["Day"] == "Wed"])

def cond_prob_wed_profit(path, sheet_index):
    df = pd.read_excel(path, sheet_name=sheet_index).dropna(axis=1)
    p_wed_profit = prob_profit_on_wed(path, sheet_index)
    p_wed = len(df[df["Day"] == "Wed"]) / len(df)
    return p_wed_profit / p_wed if p_wed > 0 else 0

def scatter_chg_by_day(path, sheet_index=1):
    df = pd.read_excel(path, sheet_name=sheet_index).dropna(axis=1)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df["Day"], y=df["Chg%"], color="blue")
    plt.title("Change % vs Day")
    plt.xlabel("Day")
    plt.ylabel("Chg%")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#  Q3 Output
mean_val, var_val = compute_price_stats(file_path, 1)
print("Mean Price:", mean_val)
print("Price Variance:", var_val)

wed_result = analyze_wednesday(file_path, 1, mean_val)
print("Wednesday Mean:", wed_result[0], " | Difference:", wed_result[1])

apr_result = analyze_april(file_path, 1, mean_val)
print("April Mean:", apr_result[0], " | Difference:", apr_result[1])

print("Loss Probability:", prob_of_loss(file_path, 1))
print("Wednesday Profit Probability:", prob_profit_on_wed(file_path, 1))
print("Conditional Wednesday Profit Probability:", cond_prob_wed_profit(file_path, 1))

scatter_chg_by_day(file_path)

#  Q4: Read Dataset 
def read_dataset(path, sheet_index):
    return pd.read_excel(path, sheet_name=sheet_index).dropna(axis=1)

#  Q5: Split Columns 
def split_columns(df):
    cat = [col for col in df.columns if df[col].dtype == object]
    num = [col for col in df.columns if df[col].dtype != object]
    return cat, num

# Q6: Encode Categorical Columns 
def encode_categoricals(df, cat_cols):
    df_encoded = df.copy()
    encoders = {}

    for col in cat_cols:
        if df[col].nunique() <= 5:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            df_encoded = pd.get_dummies(df_encoded, columns=[col], prefix=col)
    return df_encoded, encoders

# Q7: Detect Missing, Ranges, Outliers, Stats 
def detect_missing(df):
    result = []
    for col in df.columns:
        total_missing = df[col].isnull().sum() + (df[col] == '?').sum()
        if total_missing > 0:
            result.append([col, total_missing])
    return result

def column_ranges(df, num_cols):
    return [[col, df[col].min(), df[col].max()] for col in num_cols]

def identify_outliers(df, num_cols):
    result = {}
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        out_vals = df[col][(df[col] < low) | (df[col] > high)]
        if not out_vals.empty:
            result[col] = out_vals.tolist()
    return result if result else None

def col_stats(df, num_cols):
    return [[col, statistics.mean(df[col]), statistics.variance(df[col])] for col in num_cols]

# Q8: Impute Missing Values 
def imputing_missing_values(df):
    df.replace('?', np.nan, inplace=True)
    cat_cols, num_cols = split_columns(df)

    for col in num_cols:
        if df[col].isnull().sum() > 0:
            skewness = df[col].skew()
            value = df[col].median() if abs(skewness) > 1 else df[col].mean()
            df[col].fillna(value, inplace=True)

    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df

# Q9: Standardize 
def standardize(df, num_cols):
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

# Q10: Similarity Metrics
def jc_smc_metrics(v1, v2):
    v1 = np.where(pd.to_numeric(v1, errors='coerce') > 0, 1, 0)
    v2 = np.where(pd.to_numeric(v2, errors='coerce') > 0, 1, 0)

    a = np.sum((v1 == 1) & (v2 == 1))
    b = np.sum((v1 == 1) & (v2 == 0))
    c = np.sum((v1 == 0) & (v2 == 1))
    d = np.sum((v1 == 0) & (v2 == 0))

    jc = a / (a + b + c) if (a + b + c) > 0 else 0
    smc = (a + d) / (a + b + c + d) if (a + b + c + d) > 0 else 0
    return jc, smc

def cosine_sim(v1, v2):
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot / norm if norm != 0 else 0

def plot_similarity_matrices(df, num_vectors=20):
    df_numeric = df.apply(pd.to_numeric, errors='coerce').iloc[:num_vectors]
    jc_mat = np.zeros((num_vectors, num_vectors))
    smc_mat = np.zeros((num_vectors, num_vectors))
    cos_mat = np.zeros((num_vectors, num_vectors))

    for i in range(num_vectors):
        for j in range(num_vectors):
            jc_val, smc_val = jc_smc_metrics(df_numeric.iloc[i], df_numeric.iloc[j])
            cos_val = cosine_sim(df_numeric.iloc[i], df_numeric.iloc[j])
            jc_mat[i, j] = jc_val
            smc_mat[i, j] = smc_val
            cos_mat[i, j] = cos_val

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    sns.heatmap(jc_mat, ax=axes[0], annot=True, fmt=".2f", cmap="Greens")
    axes[0].set_title("Jaccard Coefficient")
    sns.heatmap(smc_mat, ax=axes[1], annot=True, fmt=".2f", cmap="Oranges")
    axes[1].set_title("Simple Matching Coefficient")
    sns.heatmap(cos_mat, ax=axes[2], annot=True, fmt=".2f", cmap="Purples")
    axes[2].set_title("Cosine Similarity")
    plt.tight_layout()
    plt.show()
#running q4-q10
df = read_dataset(file_path, 2)
cat_cols, num_cols = split_columns(df)
print("Categorical:", cat_cols)
print("Numerical:", num_cols)
print("Missing Data:", detect_missing(df))

encoded_df, enc_map = encode_categoricals(df, cat_cols)
encoded_df = imputing_missing_values(encoded_df)
print("Ranges:", column_ranges(encoded_df, num_cols))
print("Outliers:", identify_outliers(encoded_df, num_cols))
print("Mean & Variance:", col_stats(encoded_df, num_cols))

normalized_df = standardize(encoded_df, num_cols)
print("Normalized Data:\n", normalized_df)

binary_cols = [col for col in encoded_df.columns if encoded_df[col].nunique() == 2]
v1 = encoded_df.iloc[0][binary_cols].values
v2 = encoded_df.iloc[1][binary_cols].values
jc_val, smc_val = jc_smc_metrics(v1, v2)
print("Jaccard Coefficient:", jc_val)
print("SMC:", smc_val)

plot_similarity_matrices(normalized_df, num_vectors=20)
