import numpy as np
import random
from collections import Counter
import time
import copy

# generate a random sudoku puzzle
def generate_sudoku(numFilledEntries):
    base = 3
    side = base * base
    board = [[0] * side for _ in range(side)]

    # fill diagonal of 3x3 with random nums
    for i in range(0, side, base):
        nums = random.sample(range(1, side + 1), base)
        for j in range(base):
            board[i + j][i + j] = nums[j]

    # Solve the puzzle to check if the puzzle is solvable. Remove random entries to create the puzzle
    if solve_sudoku(board):
        empty_cells = 81 - numFilledEntries
        empty_indices = random.sample(range(81), empty_cells)
        for idx in empty_indices:
            row, col = divmod(idx, 9)
            board[row][col] = 0
        return board
    else:
        # if it's not solvable, then regenerate it
        return generate_sudoku(numFilledEntries)

# backtrack method to solve sudolku, for checking validity of the generated board.
def solve_sudoku(board):
    empty_cell = find_empty_cell(board)
    if not empty_cell:
        return True 
    row, col = empty_cell
    for num in range(1, 10):
        if is_safe(board, row, col, num):
            board[row][col] = num
            if solve_sudoku(board):
                return True
            board[row][col] = 0
    return False

# check for empty cell
def find_empty_cell(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)
    return None

# check if a number can be placed in a cell
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

# fitness function for sudoku, penalties for having the same entry listed more than once to
#encourage the algorithm to select better fit generations.
def sudoku_fitness(solution):
    fitness = 0
    # check for duplicates
    for i in range(9):
        row_counts = Counter(solution[i])
        col_counts = Counter(solution[j][i] for j in range(9))
        fitness -= sum((count - 1) ** 2 for count in row_counts.values() if count > 1)
        fitness -= sum((count - 1) ** 2 for count in col_counts.values() if count > 1)

    # heck 3x3 subgrid
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            subgrid_counts = Counter(solution[x][y] for x in range(i, i+3) for y in range(j, j+3))
            fitness -= sum((count - 1) ** 2 for count in subgrid_counts.values() if count > 1)

    return fitness


# tournament selection
def tournament_selection(population, fitness_scores, tournament_size):
    selected = []
    while len(selected) < len(population):
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        selected.append(population[winner_index])
    return selected
     
# selection
def select(population, fitness_scores, num_selected):
    selected_indices = np.argsort(fitness_scores)[-num_selected:]
    return [population[i] for i in selected_indices]

def mutate(solution, initial_board, mutation_rate):
    mutated_solution = copy.deepcopy(solution)
    for i in range(9):
        for j in range(9):
            if initial_board[i][j] == 0 and random.random() < mutation_rate:
                valid_values = [num for num in range(1, 10) if num not in mutated_solution[i] and num not in [mutated_solution[x][j] for x in range(9)]]
                if valid_values:
                    mutated_solution[i][j] = random.choice(valid_values)
    return mutated_solution

def crossover(parent1, parent2, initial_board):
    crossover_point = random.randint(1, 8)
    child1 = copy.deepcopy(parent1[:crossover_point]) + copy.deepcopy(parent2[crossover_point:])
    child2 = copy.deepcopy(parent2[:crossover_point]) + copy.deepcopy(parent1[crossover_point:])
    #need to make sure the initially designated values of the puzzle don't change
    for i in range(9):
        for j in range(9):
            if initial_board[i][j] != 0:
                child1[i][j] = initial_board[i][j]
                child2[i][j] = initial_board[i][j]
    # try to resolve occurences of duplicate values
    child1 = resolve_duplicates(child1)
    child2 = resolve_duplicates(child2)
    return child1, child2
#doesn't quite get rid of duplicate values, needs better logic
def resolve_duplicates(solution):
    for i in range(9):
        row_counts = Counter(solution[i])
        for num, count in row_counts.items():
            if count > 1:
                empty_indices = [j for j in range(9) if solution[i][j] == 0]
                for j in empty_indices:
                    if num not in [solution[x][j] for x in range(9)]:
                        solution[i][j] = num
                        break
    return solution


# set initial population for gen 0
def initialize_population(population_size):
    return [generate_sudoku(numFilledEntries=random.randint(25, 30)) for _ in range(population_size)]

def genetic_algorithm(population_size, max_generations, mutation_rate, crossover_rate, tournament_size, initial_board):
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

        selected = tournament_selection(population, fitness_scores, tournament_size)

        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.choices(selected, k=2)
            #checks mutation, assigns probability to mutation and the rest is used for crossover
            if random.random() < mutation_rate: 
                offspring = mutate(parent1, initial_board, mutation_rate)
                mutation_count += 1
            else:  
                offspring1, offspring2 = crossover(parent1, parent2, initial_board)
                offspring = random.choice([offspring1, offspring2])
                crossover_count += 1
            new_population.append(offspring)

        population = new_population
        generation += 1

        # time check to make sure this thing doesn't run for hours on end
        if time.time() - start_time > 30:
            break

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal generations: {generation}")
    print(f"Total time taken: {total_time} seconds")
    print(f"Total mutations occurred: {mutation_count}")
    print(f"Total crossovers occurred: {crossover_count}")

    return best_solution

# hill climbing local search takes over after genetic mutations to further refine the final solution.
def is_valid_solution(solution):
    # Check rows and columns for duplicates
    for i in range(9):
        row_counts = Counter(solution[i])
        if any(count > 1 for count in row_counts.values()):
            return False
        col_counts = Counter(solution[j][i] for j in range(9))
        if any(count > 1 for count in col_counts.values()):
            return False

    # Check 3x3 subgrids for duplicates
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            subgrid_counts = Counter(solution[x][y] for x in range(i, i+3) for y in range(j, j+3))
            if any(count > 1 for count in subgrid_counts.values()):
                return False

    return True
     
def hill_climbing(initial_solution, initial_board):
    current_solution = copy.deepcopy(initial_solution)
    current_fitness = sudoku_fitness(current_solution)

    # initialize the best solution and the fitness
    best_solution = copy.deepcopy(current_solution)
    best_fitness = current_fitness

    while True:
        # go through each cell
        for i in range(9):
            for j in range(9):
                # skip initial values from the puzzle
                if initial_board[i][j] != 0:
                    continue

                # try all values possible
                for value in range(1, 10):
                    if current_solution[i][j] == value:
                        continue

                    current_solution[i][j] = value

                    #check validity
                    if is_valid_solution(current_solution):
                        # update fitness
                        new_fitness = sudoku_fitness(current_solution)

                        # change best solution of new solution is better
                        if new_fitness > best_fitness:
                            best_solution = copy.deepcopy(current_solution)
                            best_fitness = new_fitness

                    # revert
                    current_solution[i][j] = initial_solution[i][j]

        # if no improvements, break
        if best_fitness <= current_fitness:
            break

        # update solution and fitness
        current_solution = copy.deepcopy(best_solution)
        current_fitness = best_fitness

    return best_solution


def print_sudoku(board):
    for row in board:
        print(row)

print("Randomly generated Sudoku puzzle:")
random_puzzle = generate_sudoku(numFilledEntries=random.randint(25, 30))
#copy the initial puzzle to preserve the values
initial_board = copy.deepcopy(random_puzzle) 
print_sudoku(random_puzzle)
print("\nSolving using genetic algorithm...\n")
best_solution = genetic_algorithm(population_size=50, max_generations=1000, mutation_rate=0.4, crossover_rate=0.8, tournament_size=10, initial_board=initial_board)
print("\nBest solution found:")
print_sudoku(best_solution)

print("\nApplying hill climbing local search...\n")
final_solution = hill_climbing(best_solution, initial_board)
print("\nBest solution after hill climbing:")
print_sudoku(final_solution)
