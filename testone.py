
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

    if solve_sudoku(board):
        print("\nSudoku Solved:")
        print_board(board)
    else:
        print("\nNo solution exists")

if __name__ == "__main__":
    main()


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

    if solve_sudoku(board):
        print("\nSudoku Solved:")
        print_board(board)
    else:
        print("\nNo solution exists")

if __name__ == "__main__":
    main()
