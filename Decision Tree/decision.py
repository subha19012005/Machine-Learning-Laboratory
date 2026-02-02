import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Load CSV Dataset
# -------------------------------
df = pd.read_csv("dataset.csv")
print("Dataset Preview:\n", df.head())

# -------------------------------
# Step 2: Handle Missing Values
# -------------------------------
df = df.ffill()

# -------------------------------
# Step 3: Drop Irrelevant Column
# -------------------------------
df.drop('loan_id', axis=1, inplace=True)

# -------------------------------
# Step 4: Encode Categorical Data
# -------------------------------
encoders = {}

for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# -------------------------------
# Step 5: Split Features & Target
# -------------------------------
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# -------------------------------
# Step 6: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------
# Step 7: Train Decision Tree
# -------------------------------
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# Step 8: Accuracy
# -------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# -------------------------------
# Step 9: Sample (Unseen) Input
# -------------------------------
sample_input = {
    'gender': 'male',
    'married': 'yes',
    'dependents': '0',
    'education': 'graduate',
    'self_employed': 'no',
    'applicantincome': 5000,
    'coapplicantincome': 0,
    'loanamount': 150,
    'loan_amount_term': 360,
    'credit_history': 1,
    'property_area': 'urban'
}


sample_df = pd.DataFrame([sample_input])

# Encode sample
for col in sample_df.columns:
    if col in encoders:
        sample_df[col] = encoders[col].transform(sample_df[col])

sample_prediction = model.predict(sample_df)

print("\nSample Prediction")
print("Input:", sample_input)
print("Loan Status ->",
      "Approved" if sample_prediction[0] == 1 else "Rejected")

# -------------------------------
# Step 10: Decision Tree Visualization
# -------------------------------
plt.figure(figsize=(20, 10))
tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Rejected', 'Approved'],
    filled=True
)
plt.title("Decision Tree – Loan Approval (Kaggle Dataset)")
plt.show()
