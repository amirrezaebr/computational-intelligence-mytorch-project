import numpy as np
import pandas as pd

data = pd.read_csv('Iris-Train.csv')
test = pd.read_csv('Iris-Test.csv')

# Encode the species labels to numerical labels
species = data['Species'].unique()
species_to_label = {species_name: idx for idx, species_name in enumerate(species)}
label_to_species = {idx: species_name for species_name, idx in species_to_label.items()}
data['Species'] = data['Species'].map(species_to_label)
test['Species'] = test['Species'].map(species_to_label)

# Extract features and labels
X_train = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y_train = data['Species'].values.astype(np.int32)  # Ensure labels are integers
X_test = test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y_test = test['Species'].values.astype(np.int32)  # Ensure labels are integers

# Shuffle and split the data into training and testing sets
np.random.seed(42)
indices = np.random.permutation(len(X_train))
X_train = X_train[indices]
y_train = y_train[indices]

print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")