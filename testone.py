import random
import time

def generate_sudoku():
    """
    Generates a random Sudoku puzzle.
    """
    # Initialize an empty 9x9 Sudoku grid
    sudoku_grid = [[0 for _ in range(9)] for _ in range(9)]
    
    # Fill the diagonal blocks (3x3) with random numbers
    for i in range(0, 9, 3):
        block_values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        random.shuffle(block_values)
        for j in range(3):
            for k in range(3):
                sudoku_grid[i + j][i + k] = block_values.pop()
    
    # Fill the rest of the grid using backtracking algorithm
    solve_sudoku(sudoku_grid)
    
    # Remove some numbers to create a puzzle
    remove_count = random.randint(40, 55)  # Adjust this range for the difficulty level
    for _ in range(remove_count):
        row = random.randint(0, 8)
        col = random.randint(0, 8)
        sudoku_grid[row][col] = 0
    
    return sudoku_grid

def fitness(solution):
    """
    Evaluates the fitness of a Sudoku solution.
    """
    # Calculate fitness based on how close the solution's row and column sums are to 45
    row_sums = [sum(row) for row in solution]
    col_sums = [sum(col) for col in zip(*solution)]
    row_fitness = sum(abs(45 - sum(row)) for row in solution)
    col_fitness = sum(abs(45 - sum(col)) for col in zip(*solution))
    return row_fitness + col_fitness

def solve_sudoku(grid):
    """
    Solves the Sudoku puzzle using backtracking algorithm.
    """
    # Find empty cell
    empty_cell = find_empty_cell(grid)
    if not empty_cell:
        return True  # Puzzle solved
    
    row, col = empty_cell
    
    # Try placing numbers 1-9
    for num in range(1, 10):
        if is_valid_move(grid, row, col, num):
            grid[row][col] = num
            
            # Recursively solve the puzzle
            if solve_sudoku(grid):
                return True
            
            # Backtrack
            grid[row][col] = 0
    
    # No solution found
    return False

def find_empty_cell(grid):
    """
    Finds an empty cell in the Sudoku grid.
    """
    for i in range(9):
        for j in range(9):
            if grid[i][j] == 0:
                return (i, j)
    return None

def is_valid_move(grid, row, col, num):
    """
    Checks if the given number can be placed in the specified position.
    """
    # Check row and column
    for i in range(9):
        if grid[row][i] == num or grid[i][col] == num:
            return False
    
    # Check 3x3 box
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if grid[i][j] == num:
                return False
    
    return True

def generate_population(size):
    """
    Generates a population of random Sudoku solutions.
    """
    population = []
    for _ in range(size):
        sudoku = generate_sudoku()
        population.append(sudoku)
    return population

def sort_population(population):
    """
    Sorts the population according to fitness.
    """
    return sorted(population, key=fitness)

def replace_worst(population, new_solution):
    """
    Replaces the worst solution in the population with a new solution.
    """
    population[-1] = new_solution
    return population

def check_global_best(population, global_best):
    """
    Checks if there is a new global best fitness in the population.
    """
    best_fitness = fitness(population[0])
    if best_fitness < global_best[0]:
        global_best[0] = best_fitness
        global_best[1] = population[0]
    return global_best

def introduce_crossover(population):
    """
    Introduces crossover by selecting two parents and creating a new solution through crossover.
    """
    # Select two parents (can be randomly or based on fitness)
    parent1, parent2 = random.choices(population, k=2)
    
    # Perform crossover (e.g., two-point crossover)
    crossover_point1 = random.randint(0, 8)
    crossover_point2 = random.randint(crossover_point1 + 1, 9)
    
    child = []
    for i in range(9):
        if i < crossover_point1 or i >= crossover_point2:
            child.append(parent1[i][:])
        else:
            child.append(parent2[i][:])
    
    return child

def print_sudoku(grid):
    """
    Prints the Sudoku grid.
    """
    for row in grid:
        print(" ".join(map(str, row)))

# Example usage
if __name__ == "__main__":
    start_time = time.time()
    population_size = 10
    max_iterations = 1000
    iteration = 0
    global_best_fitness = [float('inf'), None]
    no_improvement_count = 0
    max_no_improvement = 100  # Adjust this value as needed

    print("Generated Random Puzzle:")
    random_puzzle = generate_sudoku()
    print_sudoku(random_puzzle)
    print("\n")

    while iteration < max_iterations and no_improvement_count < max_no_improvement:
        # Generate initial population
        population = generate_population(population_size)
        
        # Sort population according to fitness
        population = sort_population(population)
        
        # Replace worst solution with a new solution
        new_solution = generate_sudoku()
        population = replace_worst(population, new_solution)
        
        # Check for new global best fitness
        global_best_fitness = check_global_best(population, global_best_fitness)
        
        # Introduce crossover if needed
        if iteration % 10 == 0:
            population.append(introduce_crossover(population))
        
        # Check for improvement
        if fitness(population[0]) >= global_best_fitness[0]:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
        
        iteration += 1

    end_time = time.time()
    print("Solution:")
    print_sudoku(global_best_fitness[1])
    print("Time taken:", end_time - start_time, "seconds")
    print("Number of iterations:", iteration)
