import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------
# Setup
# -------------------------------
sns.set_style("whitegrid")

# Auto-create output folder
os.makedirs("outputs/plots", exist_ok=True)

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("data/Combined12.csv")

# -------------------------------
# Basic Info
# -------------------------------
print("\nDataset Shape:", df.shape)
print("\nColumns:", df.columns)
print("\nMissing Values:\n", df.isnull().sum())

# -------------------------------
# Data Cleaning
# -------------------------------
df = df.drop_duplicates()

# Fill missing numeric values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Convert date column if exists
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date')

# -------------------------------
# 1. Temperature Trend
# -------------------------------
if 'temperature' in df.columns and 'date' in df.columns:
    plt.figure()
    plt.plot(df['date'], df['temperature'])
    plt.title("Temperature Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/plots/temperature_trend.png")
    plt.close()

# -------------------------------
# 2. Humidity Distribution
# -------------------------------
if 'humidity' in df.columns:
    plt.figure()
    sns.histplot(df['humidity'], kde=True)
    plt.title("Humidity Distribution")
    plt.tight_layout()
    plt.savefig("outputs/plots/humidity_distribution.png")
    plt.close()

# -------------------------------
# 3. Correlation Heatmap
# -------------------------------
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/plots/correlation_heatmap.png")
plt.close()

# -------------------------------
# 4. Boxplots (All Numeric Columns)
# -------------------------------
numeric_cols = df.select_dtypes(include=['number']).columns

for col in numeric_cols:
    plt.figure()
    sns.boxplot(y=df[col])
    plt.title(f"{col} Boxplot")
    plt.tight_layout()
    plt.savefig(f"outputs/plots/{col}_boxplot.png")
    plt.close()

# -------------------------------
# 5. Scatter Plot (Temp vs Humidity)
# -------------------------------
if 'temperature' in df.columns and 'humidity' in df.columns:
    plt.figure()
    sns.scatterplot(x=df['temperature'], y=df['humidity'])
    plt.title("Temperature vs Humidity")
    plt.tight_layout()
    plt.savefig("outputs/plots/temp_vs_humidity.png")
    plt.close()

# -------------------------------
# Done
# -------------------------------
print("\n✅ Analysis complete! Check outputs/plots/ for images.")