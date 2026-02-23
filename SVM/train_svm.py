import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
import joblib

df = pd.read_csv("dataset.csv")
df.columns = df.columns.str.strip().str.lower()

if 'loan_id' in df.columns:
    df.drop('loan_id', axis=1, inplace=True)

# -------- Encode Categorical Columns --------
encoders = {}

for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

X = df.drop('loan_status', axis=1)
y = df['loan_status']

# -------- 🔥 FIX NaN VALUES --------
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# -------- 🔥 SCALE FEATURES --------
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------- FAST SVM --------
model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n✅ SVM Model Accuracy:", round(accuracy, 3))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("SVM Confusion Matrix")
plt.show()

# Save everything
joblib.dump(model, "svm_model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(imputer, "imputer.pkl")

print("\n🎉 Model Saved Successfully!")