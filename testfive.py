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
    # Check rows and columns for duplicates
    for i in range(9):
        row_counts = Counter(solution[i])
        col_counts = Counter(solution[j][i] for j in range(9))
        fitness -= sum((count - 1) ** 2 for count in row_counts.values() if count > 1)
        fitness -= sum((count - 1) ** 2 for count in col_counts.values() if count > 1)

    # Check 3x3 subgrids for duplicates
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            subgrid_counts = Counter(solution[x][y] for x in range(i, i+3) for y in range(j, j+3))
            fitness -= sum((count - 1) ** 2 for count in subgrid_counts.values() if count > 1)

    return fitness


# Function to perform selection
def select(population, fitness_scores, num_selected):
    selected_indices = np.argsort(fitness_scores)[-num_selected:]
    return [population[i] for i in selected_indices]

def mutate(solution, initial_board, mutation_rate):
    mutated_solution = copy.deepcopy(solution)
    for i in range(9):
        for j in range(9):
            if initial_board[i][j] == 0 and random.random() < mutation_rate:
                mutated_solution[i][j] = random.randint(1, 9)
            elif initial_board[i][j] != 0:
                mutated_solution[i][j] = initial_board[i][j]  # Preserve initial values
    return mutated_solution


def crossover(parent1, parent2, initial_board):
    crossover_point = random.randint(1, 8)
    child1 = copy.deepcopy(parent1[:crossover_point]) + copy.deepcopy(parent2[crossover_point:])
    child2 = copy.deepcopy(parent2[:crossover_point]) + copy.deepcopy(parent1[crossover_point:])
    # Adjust offspring to preserve initially filled values
    for i in range(9):
        for j in range(9):
            if initial_board[i][j] != 0:
                child1[i][j] = initial_board[i][j]
                child2[i][j] = initial_board[i][j]
    return child1, child2

# Function to generate initial population
def initialize_population(population_size):
    return [generate_sudoku(num_filled_entries=random.randint(25, 30)) for _ in range(population_size)]

# Main genetic algorithm function
def genetic_algorithm(population_size, max_generations, mutation_rate, initial_board):
    start_time = time.time()
    population = initialize_population(population_size)
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
            filled_grid_spots = sum(row.count(0) for row in best_solution)
            print(f"Filled grid spots: {81 - filled_grid_spots}")
        else:
            stagnation_count += 1

        selected = select(population, fitness_scores, num_selected=10)

        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.choices(selected, k=2)
            if random.random() < mutation_rate:  # Favor mutation
                offspring = mutate(parent1, initial_board, mutation_rate)
                mutation_count += 1
            else:  # Use remaining probability for crossover
                offspring1, offspring2 = crossover(parent1, parent2, initial_board)
                offspring = random.choice([offspring1, offspring2])
                crossover_count += 1
            new_population.append(offspring)

        population = new_population
        generation += 1

        # Check elapsed time and stop if exceeds 30 seconds
        if time.time() - start_time > 30:
            break

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal generations: {generation}")
    print(f"Total time taken: {total_time} seconds")
    print(f"Total mutations occurred: {mutation_count}")
    print(f"Total crossovers occurred: {crossover_count}")

    return best_solution



# Function to print the Sudoku board
def print_sudoku(board):
    for row in board:
        print(row)

# Example usage
print("Randomly generated Sudoku puzzle:")
random_puzzle = generate_sudoku(num_filled_entries=random.randint(25, 30))
initial_board = copy.deepcopy(random_puzzle)  # Make a deep copy of the initial puzzle
print_sudoku(random_puzzle)
print("\nSolving using genetic algorithm...\n")
best_solution = genetic_algorithm(population_size=50, max_generations=1000, mutation_rate=0.4, initial_board=initial_board)
print("\nBest solution found:")
print_sudoku(best_solution)

