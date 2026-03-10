# app.py
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# ---------------------------
# Load dataset
# ---------------------------
data = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")

# Ensure binary target: 0 = No disease, 1 = Disease
data['target'] = data['target'].apply(lambda x: 0 if x == 0 else 1)

# ---------------------------
# Rename columns for easier access
# ---------------------------
data = data.rename(columns={
    'chest pain type': 'chest_pain_type',
    'resting bp s': 'resting_bp_s',
    'max heart rate': 'max_heart_rate'
})

# ---------------------------
# Bin continuous features for DiscreteBayesianNetwork
# ---------------------------
data['age_bin'] = pd.cut(data['age'], bins=[0,40,55,100], labels=['young','middle','old'])
data['resting_bp_bin'] = pd.cut(data['resting_bp_s'], bins=[0,120,140,300], labels=['low','normal','high'])
data['cholesterol_bin'] = pd.cut(data['cholesterol'], bins=[0,200,240,600], labels=['low','normal','high'])
data['max_hr_bin'] = pd.cut(data['max_heart_rate'], bins=[0,120,150,250], labels=['low','normal','high'])

# ---------------------------
# Select only the binned + categorical features
# ---------------------------
features = ['age_bin','sex','chest_pain_type','resting_bp_bin','cholesterol_bin','max_hr_bin','target']
data = data[features]

# ---------------------------
# Define Discrete Bayesian Network
# ---------------------------
model = DiscreteBayesianNetwork([
    ('age_bin','target'),
    ('sex','target'),
    ('chest_pain_type','target'),
    ('resting_bp_bin','target'),
    ('cholesterol_bin','target'),
    ('max_hr_bin','target')
])

# ---------------------------
# Train the model
# ---------------------------
model.fit(data, estimator=MaximumLikelihoodEstimator)

# ---------------------------
# Inference engine
# ---------------------------
inference = VariableElimination(model)

# ---------------------------
# Example patient prediction
# ---------------------------
patient = {
    'age_bin': 'middle',           # e.g., 41-55
    'sex': 1,                      # 0=Female, 1=Male
    'chest_pain_type': 2,          # 0-3
    'resting_bp_bin': 'high',      # low/normal/high
    'cholesterol_bin': 'high',     # low/normal/high
    'max_hr_bin': 'normal'         # low/normal/high
}

result = inference.query(variables=['target'], evidence=patient)
print("Prediction probabilities for heart disease (0=No, 1=Yes):")
print(result)

# ---------------------------
# Plot Bayesian Network
# ---------------------------
G = nx.DiGraph()
G.add_nodes_from(model.nodes())
G.add_edges_from(model.edges())

plt.figure(figsize=(8,6))
pos = nx.spring_layout(G, seed=42)  # fixed seed for consistent layout
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', arrowsize=20, font_size=12)
plt.title("Bayesian Network - Heart Disease Features")
plt.show()