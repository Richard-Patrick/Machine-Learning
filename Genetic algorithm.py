import random

items = [
    {'weight': 2, 'value': 3},
    {'weight': 3, 'value': 4},
    {'weight': 4, 'value': 5},
    {'weight': 5, 'value': 8},
    {'weight': 9, 'value': 10}
]
knapsack_capacity = 16

population_size = 50
generations = 100
mutation_rate = 0.1


def generate_population():
    population = []
    for _ in range(population_size):
        chromosome = [random.choice([0, 1]) for _ in range(len(items))]
        population.append(chromosome)
    return population

def calculate_fitness(chromosome):
    total_weight = 0
    total_value = 0
    for i in range(len(chromosome)):
        if chromosome[i] == 1:
            total_weight += items[i]['weight']
            total_value += items[i]['value']
    if total_weight > knapsack_capacity:
        total_value = 0
    return total_value

def selection(population):
    parents = []
    for _ in range(2):
        tournament = random.sample(population, 5)  # Tournament size = 5
        best_chromosome = max(tournament, key=calculate_fitness)
        parents.append(best_chromosome)
    return parents

def crossover(parents):
    crossover_point = random.randint(1, len(items) - 1)
    offspring = []
    for i in range(len(parents[0])):
        if i < crossover_point:
            offspring.append(parents[0][i])
        else:
            offspring.append(parents[1][i])
    return offspring

def mutation(chromosome):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]  # Flip bit
    return chromosome

def genetic_algorithm():
    population = generate_population()
    best_fitness = 0
    best_solution = None

    for _ in range(generations):
        new_population = []
        for _ in range(population_size // 2):
            parents = selection(population)
            offspring = crossover(parents)
            offspring = mutation(offspring)
            new_population.extend([parents[0], parents[1], offspring])

        population = new_population
        best_chromosome = max(population, key=calculate_fitness)
        fitness = calculate_fitness(best_chromosome)

        if fitness > best_fitness:
            best_fitness = fitness
            best_solution = best_chromosome

    return best_solution

best_solution = genetic_algorithm()
best_value = calculate_fitness(best_solution)

selected_items = [item for item, is_selected in zip(items, best_solution) if is_selected == 1]

print("Best Solution:")
for item in selected_items:
    print(item)

print("Best Value:", best_value)
