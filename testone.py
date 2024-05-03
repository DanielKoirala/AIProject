import random

def mutate(board, mutation_rate):
    # Iterate through each cell and potentially mutate it
    for i in range(9):
        for j in range(9):
            if random.random() < mutation_rate:
                # Mutate the cell by randomly selecting a new value
                board[i][j] = random.randint(1, 9)

def genetic_solve_sudoku(board, population_size, mutation_rate, max_generations):
    for generation in range(max_generations):
        # Create a population of Sudoku boards
        population = [board] * population_size

        # Mutate each Sudoku board in the population
        for i in range(population_size):
            mutate(population[i], mutation_rate)

        # Evaluate the fitness of each board in the population
        # (For Sudoku, fitness can be the number of correct cells)

        # Select the best boards for reproduction
        # (You can use various selection methods like roulette wheel or tournament selection)

        # Reproduce the selected boards to create the next generation

        # Check if any board in the population is solved
        for board in population:
            if is_solved(board):
                return board

    # If no solution is found after the maximum generations, return None
    return None

def is_valid(board, row, col, num):
    # Check if the number is already present in the row
    if num in board[row]:
        return False
    
    # Check if the number is already present in the column
    for i in range(9):
        if board[i][col] == num:
            return False
    
    # Check if the number is already present in the 3x3 grid
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num:
                return False
    
    return True

def find_empty_location(board):
    # Find the first empty location in the board
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return i, j
    return -1, -1

def solve_sudoku(board):
    # Find an empty location
    row, col = find_empty_location(board)
    
    # If no empty location is found, the sudoku is solved
    if row == -1 and col == -1:
        return True
    
    # Try placing numbers from 1 to 9
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num
            
            # Recursively solve the rest of the sudoku
            if solve_sudoku(board):
                return True
            
            # If placing num does not lead to a solution, backtrack
            board[row][col] = 0
    
    # If no number can be placed, the sudoku is unsolvable
    return False

def print_board(board):
    for row in board:
        print(" ".join(map(str, row))) 


def main():
    # Example Sudoku board (0 represents empty cells)
    board = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]

    print("Sudoku to solve:")
    print_board(board)

    solution = genetic_solve_sudoku(board, population_size=100, mutation_rate=0.1, max_generations=1000)
    if solution:
        print("\nSudoku Solved:")
        print_board(solution)
    else:
        print("\nNo solution found within the maximum generations")

if __name__ == "__main__":
    main()
