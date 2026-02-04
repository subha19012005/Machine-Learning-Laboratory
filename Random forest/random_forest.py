import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
df = pd.read_csv("dataset.csv")
df.columns = df.columns.str.strip().str.lower()  # ⭐ FIX
df = df.ffill()

print("Columns:", df.columns)

# -------------------------------
# Step 2: Drop ID Column
# -------------------------------
if 'loan_id' in df.columns:
    df.drop('loan_id', axis=1, inplace=True)

# -------------------------------
# Step 3: Encode Categorical Data
# -------------------------------
encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# -------------------------------
# Step 4: Features & Target
# -------------------------------
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# -------------------------------
# Step 5: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------
# Step 6: Random Forest (3 Trees)
# -------------------------------
rf_model = RandomForestClassifier(
    n_estimators=3,
    criterion='entropy',
    max_depth=4,
    random_state=42
)

rf_model.fit(X_train, y_train)

# -------------------------------
# Step 7: Accuracy
# -------------------------------
y_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))

# -------------------------------
# Step 8: Visualize ALL 3 Trees
# -------------------------------
for i, estimator in enumerate(rf_model.estimators_):
    plt.figure(figsize=(20, 10))
    tree.plot_tree(
        estimator,
        feature_names=X.columns,
        class_names=['Rejected', 'Approved'],
        filled=True,
        max_depth=3
    )
    plt.title(f"Random Forest - Tree {i+1}")
    plt.show()
