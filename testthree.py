import numpy as np
import random
from collections import Counter
import time

# Function to generate a random Sudoku puzzle with a specified number of filled entries
def generate_sudoku(num_filled_entries):
    board = np.zeros((9, 9), dtype=int)
    solve_sudoku(board)  # Fill the board with a valid solution
    empty_cells = 81 - num_filled_entries
    empty_indices = random.sample(range(81), empty_cells)
    for idx in empty_indices:
        row, col = divmod(idx, 9)
        board[row][col] = 0
    return board.tolist()

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
def select(population, fitness_scores, num_selected):
    selected_indices = np.argsort(fitness_scores)[-num_selected:]
    return [population[i] for i in selected_indices]

# Function to perform crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(1, 8)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Function to perform mutation
def mutate(solution):
    mutated_solution = solution.copy()
    row = random.randint(0, 8)
    col = random.randint(0, 8)
    mutated_solution[row][col] = random.randint(1, 9)
    return mutated_solution

# Function to generate initial population
def initialize_population(population_size):
    return [generate_sudoku(num_filled_entries=random.randint(25, 30)) for _ in range(population_size)]

# Main genetic algorithm function
def genetic_algorithm(population_size, max_generations, mutation_rate):
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
            print("Best solution so far:")
            for row in best_solution:
                print(row)
            filled_grid_spots = sum(row.count(0) for row in best_solution)
            print(f"Filled grid spots: {81 - filled_grid_spots}")
        else:
            stagnation_count += 1

        selected = select(population, fitness_scores, num_selected=10)

        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.choices(selected, k=2)
            if random.random() < mutation_rate:
                offspring = mutate(random.choice(selected))
                mutation_count += 1
            else:
                offspring1, offspring2 = crossover(parent1, parent2)
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

# def print_sudoku_board(board):
#     for i in range(len(board)):
#         if i % 3 == 0 and i != 0:
#             print("-" * 21)  # Print horizontal line after every 3 rows
#         for j in range(len(board[i])):
#             if j % 3 == 0 and j != 0:
#                 print("|", end=" ")  # Print vertical line after every 3 columns
#             if j == 8:
#                 print(board[i][j])
#             else:
#                 print(str(board[i][j]) + " ", end="")


def test_initial_values_changed(initial_board, generation_boards):
    for gen_board in generation_boards:
        for i in range(9):
            for j in range(9):
                # Check if the cell is initially filled
                if initial_board[i][j] != 0:
                    # If the value in the same cell in the generation board is different
                    # and the value in the initial board is not 0 (indicating an empty cell),
                    # and the value in the generation board is not 0 (indicating it was not mutated)
                    if initial_board[i][j] != gen_board[i][j] and gen_board[i][j] != 0:
                        return False
    return True

# Example usage
print("Randomly generated Sudoku puzzle:")
random_puzzle = generate_sudoku(num_filled_entries=random.randint(25, 30))
for row in random_puzzle:
    print(row)
print("\nSolving using genetic algorithm...\n")
best_solution = genetic_algorithm(population_size=50, max_generations=1000, mutation_rate=0.1)
print("\nBest solution found:")
for row in best_solution:
    print(row)
    
    # Test if initially filled values remained unchanged
generation_boards = [random_puzzle]  # Store initial board
for _ in range(10):
    generation_boards.append(mutate(best_solution.copy()))  # Store 10 mutated boards
test_result = test_initial_values_changed(random_puzzle, generation_boards)
print("\nTest Result:", "Passed" if test_result else "Failed")
