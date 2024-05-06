import numpy as np
import random
from collections import Counter
import time
import copy

# Function to generate a random Sudoku puzzle with a specified number of filled entries
def generate_sudoku(num_filled_entries):
    base = 3
    side = base * base
    board = [[0] * side for _ in range(side)]

    # Fill the main diagonal of each 3x3 subgrid with random numbers
    for i in range(0, side, base):
        nums = random.sample(range(1, side + 1), base)
        for j in range(base):
            board[i + j][i + j] = nums[j]

    # Solve the Sudoku puzzle
    if solve_sudoku(board):
        # Randomly remove entries to create the puzzle
        empty_cells = 81 - num_filled_entries
        empty_indices = random.sample(range(81), empty_cells)
        for idx in empty_indices:
            row, col = divmod(idx, 9)
            board[row][col] = 0
        return board
    else:
        # If the generated puzzle cannot be solved, regenerate it
        return generate_sudoku(num_filled_entries)

# Function to solve a Sudoku puzzle using backtracking
def solve_sudoku(board):
    empty_cell = find_empty_cell(board)
    if not empty_cell:
        return True  # Puzzle solved
    row, col = empty_cell
    for num in range(1, 10):
        if is_safe(board, row, col, num):
            board[row][col] = num
            if solve_sudoku(board):
                return True
            board[row][col] = 0  # Backtrack
    return False

# Function to find an empty cell in the Sudoku puzzle
def find_empty_cell(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)
    return None

# Function to check if a number can be placed in a cell
def is_safe(board, row, col, num):
    # Check row and column
    if num in board[row] or num in [board[i][col] for i in range(9)]:
        return False
    # Check 3x3 subgrid
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num:
                return False
    return True

# Function to evaluate fitness of a Sudoku solution
def sudoku_fitness(solution):
    fitness = 0
    # Check rows and columns
    for i in range(9):
        row_counts = Counter(solution[i])
        col_counts = Counter(solution[j][i] for j in range(9))
        fitness += sum(1 for count in row_counts.values() if count == 1)
        fitness += sum(1 for count in col_counts.values() if count == 1)

    # Check 3x3 subgrids
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            subgrid_counts = Counter(solution[x][y] for x in range(i, i+3) for y in range(j, j+3))
            fitness += sum(1 for count in subgrid_counts.values() if count == 1)

    return fitness

# Function to perform selection
def select(population, fitness_scores, num_selected, initial_population):
    if len(initial_population) == 0:
        selected_indices = np.argsort(fitness_scores)[-num_selected:]
    else:
        selected_indices = np.argsort(fitness_scores[len(initial_population):])[-num_selected:]
    return [population[i] for i in selected_indices]

# Function to perform crossover
def crossover(parent1, parent2, initial_board):
    crossover_point = random.randint(1, 8)
    child1 = copy.deepcopy(parent1[:crossover_point]) + copy.deepcopy(parent2[crossover_point:])
    child2 = copy.deepcopy(parent2[:crossover_point]) + copy.deepcopy(parent1[crossover_point:])
    # Adjust offspring to preserve initially filled values
    for i in range(9):
        for j in range(9):
            if initial_board[i][j] != 0:
                child1[i][j] = parent1[i][j]
                child2[i][j] = parent2[i][j]
    return child1, child2

# Function to perform mutation
def mutate(solution, initial_board):
    mutated_solution = copy.deepcopy(solution)
    row = random.randint(0, 8)
    col = random.randint(0, 8)
    # Check if the cell is initially filled
    if initial_board[row][col] == 0:
        mutated_solution[row][col] = random.randint(1, 9)
    return mutated_solution

# Function to generate initial population
def initialize_population(population_size):
    return [generate_sudoku(num_filled_entries=random.randint(25, 30)) for _ in range(population_size)]

# Main genetic algorithm function
def genetic_algorithm(population_size, max_generations, mutation_rate):
    start_time = time.time()
    initial_population = initialize_population(population_size)
    population = initial_population[:]
    best_fitness = float('-inf')
    best_solution = None
    generation = 0
    stagnation_count = 0
    mutation_count = 0
    crossover_count = 0

    while generation < max_generations and stagnation_count < 100:
        fitness_scores = [sudoku_fitness(solution) for solution in population]
        max_fitness = max(fitness_scores)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_solution = population[np.argmax(fitness_scores)]
            stagnation_count = 0
            print(f"Generation {generation}: Found new best solution with fitness {max_fitness}")
            print("Best solution so far:")
            for row in best_solution:
                print(row)
            filled_grid_spots = sum(row.count(0) for row in best_solution)
            print(f"Filled grid spots: {81 - filled_grid_spots}")
        else:
            stagnation_count += 1

        selected = select(population, fitness_scores, num_selected=10, initial_population=initial_population)

        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.choices(selected, k=2)
            if random.random() < mutation_rate:
                mutation_count += 1
                offspring = mutate(parent1 if sudoku_fitness(parent1) > sudoku_fitness(parent2) else parent2, initial_board=initial_population[0])
            else:
                crossover_count += 1
                offspring1, offspring2 = crossover(parent1, parent2, initial_board=initial_population[0])
                offspring = offspring1 if sudoku_fitness(offspring1) > sudoku_fitness(offspring2) else offspring2
            new_population.append(offspring)

        population = new_population
        generation += 1

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time} seconds")
    print(f"Best solution found with fitness: {best_fitness}")
    print("Final solution:")
    for row in best_solution:
        print(row)
    print(f"Mutation count: {mutation_count}")
    print(f"Crossover count: {crossover_count}")

# Run the genetic algorithm
genetic_algorithm(population_size=20, max_generations=100, mutation_rate=0.2)
