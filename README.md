### ENTER YOUR NAME: Vikamuhan Reddy
### ENTER YOUR REGISTER NO.: 212223240181  
### EX. NO.: 6
### DATE: 13/11/25  



# Heart Attack Prediction using MLP

### Aim:
To construct a Multi-Layer Perceptron (MLP) to predict heart attack using Python.



### Algorithm:

1. **Import** the required libraries: `numpy`, `pandas`, `MLPClassifier`, `train_test_split`, `StandardScaler`, `accuracy_score`, and `matplotlib.pyplot`.  
2. **Load** the heart disease dataset using `pd.read_csv()`.  
3. **Separate** the features and labels using `data.iloc` values for features (`X`) and labels (`y`).  
4. **Split** the dataset into training and testing sets using `train_test_split()`.  
5. **Normalize** the feature data using `StandardScaler()` to scale features to zero mean and unit variance.  
6. **Create** an `MLPClassifier` model with the desired architecture and hyperparameters such as `hidden_layer_sizes`, `max_iter`, and `random_state`.  
7. **Train** the MLP model using `mlp.fit(X_train, y_train)` to adjust weights and biases iteratively.  
8. **Predict** on the test data using `mlp.predict(X_test)`.  
9. **Evaluate** model accuracy using `accuracy_score(y_test, y_pred)`.  
10. **Print** the accuracy and performance metrics.  
11. **Plot** the training error convergence using `plt.plot()` and `plt.show()`.  



### Program:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the Heart Disease dataset
data = pd.read_csv('https://raw.githubusercontent.com/Lavanyajoyce/EX-6-NN/main/heart.csv')

# Separate features and labels
X = data.drop('target', axis=1)
y = data['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
training_loss = mlp.fit(X_train_scaled, y_train).loss_curve_

# Make predictions on the testing set
y_pred = mlp.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)

# Plot the error convergence
plt.plot(training_loss)
plt.title("MLP Training Loss Convergence")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.show()

```

### Output:

<img width="500" height="360" alt="Screen Shot 1947-08-22 at 13 34 58" src="https://github.com/user-attachments/assets/bc4d607d-6646-4564-9e17-12d6b2c90a1d" />


### Result:
Thus, an Artificial Neural Network with MLP is constructed and trained successfully to predict heart attack using Python.

