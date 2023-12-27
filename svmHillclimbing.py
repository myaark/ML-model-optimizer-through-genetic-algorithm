from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import random
import pandas as pd
# Load the Iris dataset from a CSV file
iris_df = pd.read_csv('iris.csv')
X = iris_df.drop('Species', axis=1)
y = iris_df['Species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=50)

# Define the Hill Climbing parameters
initial_solution = [50, 'linear']  # Initial hyperparameters
step_size = 10  # The step size for perturbing the hyperparameters
max_iterations = 100  # Maximum number of iterations

def evaluate_solution(solution):
    C, kernel = solution
    svm_classifier = SVC(C=C, kernel=kernel)
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

best_solution = initial_solution
best_accuracy = evaluate_solution(best_solution)

for iteration in range(max_iterations):
    # Generate neighboring solutions by perturbing the hyperparameters
    neighbor_solution = [
        best_solution[0] + random.uniform(-step_size, step_size),
        random.choice(['linear', 'rbf', 'poly'])
    ]
    
    # Evaluate the neighbor solution
    neighbor_accuracy = evaluate_solution(neighbor_solution)
    
    # If the neighbor solution is better, update the best solution
    if neighbor_accuracy > best_accuracy:
        best_solution = neighbor_solution
        best_accuracy = neighbor_accuracy

    print(f"Iteration {iteration + 1}: Best Solution - {best_solution}, Best Accuracy - {best_accuracy * 100:.2f}%")

print("Final Best Solution:", best_solution)
print("Final Best Accuracy:", best_accuracy * 100)
