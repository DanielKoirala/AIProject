import numpy
import random
import time

# Global variables
Nd = 9  # Number of digits (in the case of standard Sudoku puzzles, this is 9).

class Population(object):
    """ A set of candidate solutions to the Sudoku puzzle. These candidates are also known as the chromosomes in the population. """

    def __init__(self):
        self.candidates = []

    def seed(self, Nc, given):
        self.candidates = []
        helper = Candidate()
        helper.values = [[[] for j in range(0, Nd)] for i in range(0, Nd)]
        for row in range(0, Nd):
            for column in range(0, Nd):
                for value in range(1, 10):
                    if ((given.values[row][column] == 0) and not (given.is_column_duplicate(column, value) or given.is_block_duplicate(row, column, value) or given.is_row_duplicate(row, value))):
                        helper.values[row][column].append(value)
                    elif (given.values[row][column] != 0):
                        helper.values[row][column].append(given.values[row][column])
                        break

        for p in range(0, Nc):
            g = Candidate()
            for i in range(0, Nd):
                row = numpy.zeros(Nd)
                for j in range(0, Nd):
                    if (given.values[i][j] != 0):
                        row[j] = given.values[i][j]
                    elif (given.values[i][j] == 0):
                        row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j]) - 1)]

                while (len(list(set(row))) != Nd):
                    for j in range(0, Nd):
                        if (given.values[i][j] == 0):
                            row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j]) - 1)]

                g.values[i] = row

            self.candidates.append(g)

        self.update_fitness()

    def update_fitness(self):
        """ Update fitness of every candidate/chromosome. """
        for candidate in self.candidates:
            candidate.update_fitness()

    def sort(self):
        """ Sort the population based on fitness. """
        self.candidates = [candidate for candidate in self.candidates if candidate.fitness is not None]
        self.candidates.sort(key=lambda x: x.fitness, reverse=True)


class Candidate(object):
    """ A candidate solutions to the Sudoku puzzle. """

    def __init__(self):
        self.values = numpy.zeros((Nd, Nd), dtype=int)
        self.fitness = None

    def update_fitness(self):
        """ The fitness of a candidate solution is determined by how close it is to being the actual solution to the puzzle. The actual solution (i.e. the 'fittest') is defined as a 9x9 grid of numbers in the range [1, 9] where each row, column and 3x3 block contains the numbers [1, 9] without any duplicates (see e.g. http://www.sudoku.com/); if there are any duplicates then the fitness will be lower. """

        row_count = numpy.zeros(Nd)
        column_count = numpy.zeros(Nd)
        block_count = numpy.zeros(Nd)
        row_sum = 0
        column_sum = 0
        block_sum = 0

        for i in range(0, Nd):  # For each row...
            for j in range(0, Nd):  # For each number within it...
                row_count[self.values[i][j] - 1] += 1  # ...Update list with occurrence of a particular number.

            row_sum += (1.0 / len(set(row_count))) / Nd
            row_count = numpy.zeros(Nd)

        for i in range(0, Nd):  # For each column...
            for j in range(0, Nd):  # For each number within it...
                column_count[self.values[j][i] - 1] += 1  # ...Update list with occurrence of a particular number.

            column_sum += (1.0 / len(set(column_count))) / Nd
            column_count = numpy.zeros(Nd)

        # For each block...
        for i in range(0, Nd, 3):
            for j in range(0, Nd, 3):
                block_count[self.values[i][j] - 1] += 1
                block_count[self.values[i][j + 1] - 1] += 1
                block_count[self.values[i][j + 2] - 1] += 1

                block_count[self.values[i + 1][j] - 1] += 1
                block_count[self.values[i + 1][j + 1] - 1] += 1
                block_count[self.values[i + 1][j + 2] - 1] += 1

                block_count[self.values[i + 2][j] - 1] += 1
                block_count[self.values[i + 2][j + 1] - 1] += 1
                block_count[self.values[i + 2][j + 2] - 1] += 1

                block_sum += (1.0 / len(set(block_count))) / Nd
                block_count = numpy.zeros(Nd)

        # Calculate overall fitness.
        if (int(row_sum) == 1 and int(column_sum) == 1 and int(block_sum) == 1):
            fitness = 1.0
        else:
            fitness = column_sum * block_sum

        self.fitness = fitness

    def mutate(self, mutation_rate, given):
        """ Mutate a candidate by picking a row, and then picking two values within that row to swap. """

        r = random.uniform(0, 1.1)
        while (r > 1):  # Outside [0, 1] boundary - choose another
            r = random.uniform(0, 1.1)

        success = False
        if (r < mutation_rate):  # Mutate.
            while (not success):
                row1 = random.randint(0, 8)
                row2 = random.randint(0, 8)
                row2 = row1

                from_column = random.randint(0, 8)
                to_column = random.randint(0, 8)
                while (from_column == to_column):
                    from_column = random.randint(0, 8)
                    to_column = random.randint(0, 8)

                # Check if the two places are free...
                if (given.values[row1][from_column] == 0 and given.values[row1][to_column] == 0):
                    # ...and that we are not causing a duplicate in the rows' columns.
                    if (not given.is_column_duplicate(to_column, self.values[row1][from_column])
                            and not given.is_column_duplicate(from_column, self.values[row2][to_column])
                            and not given.is_block_duplicate(row2, to_column, self.values[row1][from_column])
                            and not given.is_block_duplicate(row1, from_column, self.values[row2][to_column])):

                        # Swap values.
                        temp = self.values[row2][to_column]
                        self.values[row2][to_column] = self.values[row1][from_column]
                        self.values[row1][from_column] = temp
                        success = True

        return success


class Given(Candidate):
    """ The grid containing the given/known values. """

    def __init__(self, values):
        self.values = values

    def is_row_duplicate(self, row, value):
        """ Check whether there is a duplicate of a fixed/given value in a row. """
        for column in range(0, Nd):
            if (self.values[row][column] == value):
                return True
        return False

    def is_column_duplicate(self, column, value):
        """ Check whether there is a duplicate of a fixed/given value in a column. """
        for row in range(0, Nd):
            if (self.values[row][column] == value):
                return True
        return False

    def is_block_duplicate(self, row, column, value):
        """ Check whether there is a duplicate of a fixed/given value in a 3 x 3 block. """
        i = 3 * (int(row / 3))
        j = 3 * (int(column / 3))

        if ((self.values[i][j] == value)
                or (self.values[i][j + 1] == value)
                or (self.values[i][j + 2] == value)
                or (self.values[i + 1][j] == value)
                or (self.values[i + 1][j + 1] == value)
                or (self.values[i + 1][j + 2] == value)
                or (self.values[i + 2][j] == value)
                or (self.values[i + 2][j + 1] == value)
                or (self.values[i + 2][j + 2] == value)):
            return True
        else:
            return False


class Tournament(object):
    """ The crossover function requires two parents to be selected from the population pool. The Tournament class is used to do this.

    Two individuals are selected from the population pool and a random number in [0, 1] is chosen. If this number is less than the 'selection rate' (e.g. 0.85), then the fitter individual is selected; otherwise, the weaker one is selected.
    """

    def __init__(self):
        return

    def compete(self, candidates):
        """ Pick 2 random candidates from the population and get them to compete against each other. """
        c1 = candidates[random.randint(0, len(candidates) - 1)]
        c2 = candidates[random.randint(0, len(candidates) - 1)]
        if c1.fitness > c2.fitness:
            return c1
        else:
            return c2


def sudoku_solver(Nc, Ng, Mr, given):
    population = Population()
    population.seed(Nc, given)

    print("Randomly Generated Board:")
    print_sudoku_board(given.values)

    # Evolution loop
    for g in range(0, Ng):
        # Display message for each iteration
        print(f"\nGeneration {g + 1}:")

        # Sort candidates
        population.sort()

        # Print the best solution
        if population.candidates:
            best_solution = population.candidates[0]
            print("Best Solution:")
            print_sudoku_board(best_solution.values)

            # Create a new population
            new_population = Population()

            # Reproduction
            while len(new_population.candidates) < Nc:
                tournament = Tournament()
                parent1 = tournament.compete(population.candidates)
                parent2 = tournament.compete(population.candidates)
                child = crossover(parent1, parent2)
                if random.uniform(0, 1) < Mr:
                    child.mutate(Mr, given)
                new_population.candidates.append(child)

            # Update population
            population = new_population

        else:
            print("Population is empty. Unable to produce new candidates.")
            break

    # Sort candidates
    population.sort()

    # Print the final best solution
    if population.candidates:
        best_solution = population.candidates[0]
        print("\nFinal Best Solution:")
        print_sudoku_board(best_solution.values)
    else:
        print("No solution found.")


def crossover(parent1, parent2):
    child = Candidate()
    for i in range(Nd):
        for j in range(Nd):
            if random.randint(0, 1) == 0:
                child.values[i][j] = parent1.values[i][j]
            else:
                child.values[i][j] = parent2.values[i][j]
    return child

def print_sudoku_board(board):
    for i in range(len(board)):
        if i % 3 == 0 and i != 0:
            print("-" * 21)  # Print horizontal line after every 3 rows
        for j in range(len(board[i])):
            if j % 3 == 0 and j != 0:
                print("|", end=" ")  # Print vertical line after every 3 columns
            if j == 8:
                print(board[i][j])
            else:
                print(str(board[i][j]) + " ", end="")

# Random Sudoku puzzle generator
def generate_sudoku():
    base = 3
    side = base * base

    # pattern for a baseline valid solution
    def pattern(r, c): return (base * (r % base) + r // base + c) % side

    # randomize rows, columns and numbers (of valid base pattern)
    from random import sample

    def shuffle(s): return sample(s, len(s))
    rBase = range(base)
    rows = [g * base + r for g in shuffle(rBase) for r in shuffle(rBase)]
    cols = [g * base + c for g in shuffle(rBase) for c in shuffle(rBase)]
    nums = shuffle(range(1, base * base + 1))

    # produce board using randomized baseline pattern
    board = [[nums[pattern(r, c)] for c in cols] for r in rows]

    # Make a copy of the complete puzzle
    puzzle = numpy.array(board)

    # Generate a puzzle by removing some of the numbers
    for _ in range(40):
        row = random.randint(0, 8)
        col = random.randint(0, 8)
        puzzle[row][col] = 0

    return puzzle


if __name__ == "__main__":
    # Parameters
    Nc = 10  # Number of candidates in the population.
    Ng = 50  # Number of generations.
    Mr = 0.01  # Mutation rate.

    # Generate random Sudoku puzzle
    given = Given(generate_sudoku())

    # Solve Sudoku puzzle
    sudoku_solver(Nc, Ng, Mr, given)
