import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import random

# Load the Iris dataset from a CSV file
iris_df = pd.read_csv('iris.csv')
X = iris_df.drop('Species', axis=1)
y = iris_df['Species']

#X -> values of data i.e sepal length petal length etc
#y -> the resultant values
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=50)


# Defining the genetic algorithm parameter
POPULATION_SIZE = 10


def initialize_population(population_size):
    return [[random.uniform(0.1, 100), random.choice(['linear', 'rbf', 'poly'])] for _ in range(population_size)]


# SVC -> support vector classifier
def evaluate_individual(individual):
    C, kernel = individual
    svm_classifier = SVC(C=C, kernel=kernel)
    svm_classifier.fit(X_train, y_train)  #training model
    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def select_parents(population, num_parents):
    parents = sorted(population, key=evaluate_individual, reverse=True)[:num_parents]
    #print("PARENTS:",parents)
    return parents

def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def mutate(individual):
    mutated_individual = individual.copy() 

    for i in range(len(mutated_individual)):
            if i == 0:  # Mutation for C parameter
                mutated_individual[i] = random.uniform(0.1, 100)
            elif i == 1:  # Mutation for kernel parameter
                mutated_individual[i] = random.choice(['linear', 'rbf', 'poly'])
    return mutated_individual

def genetic_algorithm(population_size, CONSTRAINT_VALUE):
    population = initialize_population(population_size)
    best_solution = [50,'linear']
    generation = 0
    while True:
        # Select parents for reproduction
        parents = select_parents(population, num_parents=2)
        next_generation = []

        # Create the next generation using crossover and mutation
        for _ in range(population_size):
            parent1, parent2 = random.sample(parents, k=2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            next_generation.append(child)
        # Replace the old population with the new generation
        population = next_generation

        best_individual = max(population, key=evaluate_individual)
        print(f"Generation {generation + 1}: Best Individual - {best_individual}, Fitness - {evaluate_individual(best_individual)}")
        generation+=1
        if(evaluate_individual(best_individual)>evaluate_individual(best_solution)):
            best_solution = best_individual
        print(evaluate_individual(best_solution))
        if evaluate_individual(best_solution)*100>CONSTRAINT_VALUE:
            break

    return best_solution

best_solution = genetic_algorithm(population_size=10,CONSTRAINT_VALUE=96.0)
print("Best Solution:", best_solution)
