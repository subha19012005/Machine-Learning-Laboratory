# ----------------------------------------
# Principal Component Analysis using Python
# ----------------------------------------

# Step 0: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------
# Step 1: Create Dataset (Raw Data)
# ----------------------------------------
data = {
    'Attendance': [85, 90, 75, 80, 95],
    'Internal': [40, 42, 35, 38, 45],
    'Assignment': [8, 9, 7, 8, 9],
    'FinalExam': [78, 85, 70, 75, 88]
}

df = pd.DataFrame(data)
print("RAW DATASET")
print(df)

# ----------------------------------------
# Step 2: Mean of Each Attribute
# (Matches manual mean calculation)
# ----------------------------------------
mean = df.mean()
print("\nMEAN OF EACH ATTRIBUTE")
print(mean)

# ----------------------------------------
# Step 3: Standard Deviation
# (Sample SD used in manual PCA)
# ----------------------------------------
std = df.std(ddof=1)
print("\nSTANDARD DEVIATION")
print(std)

# ----------------------------------------
# Step 4: Standardization (Z-score)
# Z = (X - Mean) / SD
# ----------------------------------------
Z = (df - mean) / std
print("\nSTANDARDIZED DATA (Z-SCORES)")
print(Z)

# ----------------------------------------
# Step 5: Covariance Matrix
# (Computed from standardized data)
# ----------------------------------------
cov_matrix = np.cov(Z.T)
print("\nCOVARIANCE MATRIX")
print(cov_matrix)

# ----------------------------------------
# Step 6: Eigenvalues and Eigenvectors
# ----------------------------------------
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("\nEIGENVALUES")
print(eigenvalues)

print("\nEIGENVECTORS")
print(eigenvectors)

# ----------------------------------------
# Step 7: Explained Variance
# ----------------------------------------
explained_variance = eigenvalues / np.sum(eigenvalues)
cumulative_variance = np.cumsum(explained_variance)

print("\nEXPLAINED VARIANCE (IN %)")
print(explained_variance * 100)

print("\nCUMULATIVE VARIANCE (IN %)")
print(cumulative_variance * 100)

# ----------------------------------------
# Step 8: Scree Plot
# ----------------------------------------
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
plt.xlabel("Principal Component")
plt.ylabel("Eigenvalue")
plt.title("Scree Plot")
plt.grid()
plt.show()

# ----------------------------------------
# Step 9: Principal Component Scores
# ----------------------------------------
PC_scores = Z.dot(eigenvectors)
print("\nPRINCIPAL COMPONENT SCORES")
print(PC_scores)

# ----------------------------------------
# Step 10: Dimensionality Reduction
# Retain only PC1
# ----------------------------------------
PC1 = PC_scores.iloc[:, 0]
print("\nREDUCED DATASET (USING PC1 ONLY)")
print(PC1)

# ----------------------------------------
# END OF PCA PROGRAM
# ----------------------------------------
