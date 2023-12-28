import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random

# Load the Iris dataset from a CSV file
iris_df = pd.read_csv('iris.csv')
X = iris_df.drop('Species', axis=1)
Y = iris_df['Species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=50)

# Defining the genetic algorithm parameter
POPULATION_SIZE = 10

def initialize_population(population_size):
    return [
        [random.randint(1, 100), random.randint(1, 20)]
        for _ in range(population_size)
    ]

# Machine Learning Model Evaluation
def evaluate_individual(individual):
    n_estimators, max_depth = individual
    
    classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
 
    
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def select_parents(population, num_parents):
    parents = sorted(population, key=evaluate_individual, reverse=True)[:num_parents]
    return parents

def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def mutate(individual):
    mutated_individual = individual.copy() 

    for i in range(len(mutated_individual)):
        if i == 0:  # Mutation for n_estimators
            mutated_individual[i] = random.randint(1, 100)
        elif i == 1:  # Mutation for max_depth
            mutated_individual[i] = random.randint(1, 20)
    
    return mutated_individual

def genetic_algorithm(population_size, CONSTRAINT_VALUE):
    population = initialize_population(population_size)
    generation = 0
    while True:
        parents = select_parents(population, num_parents=2)
        next_generation = []

        for _ in range(population_size):
            parent1, parent2 = random.sample(parents, k=2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            next_generation.append(child)
        
        population = next_generation

        best_individual = max(population, key=evaluate_individual)
        print(f"Generation {generation + 1}: Best Individual - {best_individual}, Fitness - {evaluate_individual(best_individual)}")
        generation += 1
        if evaluate_individual(best_individual) * 100 > CONSTRAINT_VALUE:
            break

    return best_individual

best_solution = genetic_algorithm(population_size=10, CONSTRAINT_VALUE=96.0)
print("Best Solution:", best_solution)
