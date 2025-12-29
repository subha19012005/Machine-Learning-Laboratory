import csv

# Load data from CSV file
def load_data(filename):
    concepts = []
    target = []

    with open("rainfall.csv", "r", newline='') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            concepts.append(row[:-1])   # all columns except last
            target.append(row[-1])      # last column is target (Yes/No)
    return concepts, target

# Candidate Elimination Algorithm
def candidate_elimination(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [["?" for _ in range(len(specific_h))]
                 for _ in range(len(specific_h))]

    for i, example in enumerate(concepts):
        if target[i] == "Yes":  # Positive example
            for j in range(len(specific_h)):
                if example[j] != specific_h[j]:
                    specific_h[j] = "?"
                    general_h[j][j] = "?"

        else:  # Negative example
            for j in range(len(specific_h)):
                if example[j] != specific_h[j]:
                    general_h[j][j] = specific_h[j]
                else:
                    general_h[j][j] = "?"

        print(f"\nAfter example {i+1}:")
        print("Specific hypothesis:", specific_h)
        print("General hypothesis:", general_h)

    # Remove fully general hypotheses
    general_h = [h for h in general_h if h != ["?"] * len(specific_h)]

    return specific_h, general_h

# Load data from CSV
concepts, target = load_data("rainfall.csv")  # Replace with your CSV filename

# Run algorithm
s_final, g_final = candidate_elimination(concepts, target)

print("\nFinal Specific Hypothesis:", s_final)
print("Final General Hypothesis:", g_final)
