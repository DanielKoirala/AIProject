import numpy as np
import random

Nd = 9  # Number of digits (in the case of standard Sudoku puzzles, this is 9).


class Population(object):
    """ A set of candidate solutions to the Sudoku puzzle. These candidates are also known as the chromosomes in the population. """

    def __init__(self):
        self.candidates = []
        return

    def seed(self, Nc, given):
        self.candidates = []
        
        # Determine the legal values that each square can take.
        helper = Candidate()
        helper.values = [[[] for j in range(0, Nd)] for i in range(0, Nd)]
        for row in range(0, Nd):
            for column in range(0, Nd):
                for value in range(1, 10):
                    if((given.values[row][column] == 0) and not (given.is_column_duplicate(column, value) or given.is_block_duplicate(row, column, value) or given.is_row_duplicate(row, value))):
                        # Value is available.
                        helper.values[row][column].append(value)
                    elif(given.values[row][column] != 0):
                        # Given/known value from file.
                        helper.values[row][column].append(given.values[row][column])
                        break

        # Seed a new population.       
        for p in range(0, Nc):
            g = Candidate()
            for i in range(0, Nd): # New row in candidate.
                row = np.zeros(Nd)
                
                # Fill in the givens.
                for j in range(0, Nd): # New column j value in row i.
                
                    # If value is already given, don't change it.
                    if(given.values[i][j] != 0):
                        row[j] = given.values[i][j]
                    # Fill in the gaps using the helper board.
                    elif(given.values[i][j] == 0):
                        row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j])-1)]

                # If we don't have a valid board, then try again. There must be no duplicates in the row.
                while(len(list(set(row))) != Nd):
                    for j in range(0, Nd):
                        if(given.values[i][j] == 0):
                            row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j])-1)]

                g.values[i] = row

            self.candidates.append(g)
        
        # Compute the fitness of all candidates in the population.
        self.update_fitness()
        
        print("Seeding complete.")
        
        return
        
    def update_fitness(self):
        """ Update fitness of every candidate/chromosome. """
        for candidate in self.candidates:
            candidate.update_fitness()
        return
        
    def sort(self):
        """ Sort the population based on fitness. """
        self.candidates.sort(self.sort_fitness)
        return

    def sort_fitness(self, x, y):
        """ The sorting function. """
        if(x.fitness < y.fitness):
            return 1
        elif(x.fitness == y.fitness):
            return 0
        else:
            return -1


class Candidate(object):
    """ A candidate solutions to the Sudoku puzzle. """
    def __init__(self):
        self.values = np.zeros((Nd, Nd), dtype=int)
        self.fitness = None
        return

    def update_fitness(self):
        """ The fitness of a candidate solution is determined by how close it is to being the actual solution to the puzzle. The actual solution (i.e. the 'fittest') is defined as a 9x9 grid of numbers in the range [1, 9] where each row, column and 3x3 block contains the numbers [1, 9] without any duplicates (see e.g. http://www.sudoku.com/); if there are any duplicates then the fitness will be lower. """
        
        row_count = np.zeros(Nd)
        column_count = np.zeros(Nd)
        block_count = np.zeros(Nd)
        row_sum = 0
        column_sum = 0
        block_sum = 0

        for i in range(0, Nd):  # For each row...
            for j in range(0, Nd):  # For each number within it...
                row_count[self.values[i][j]-1] += 1  # ...Update list with occurrence of a particular number.

            row_sum += (1.0/len(set(row_count)))/Nd
            row_count = np.zeros(Nd)

        for i in range(0, Nd):  # For each column...
            for j in range(0, Nd):  # For each number within it...
                column_count[self.values[j][i]-1] += 1  # ...Update list with occurrence of a particular number.

            column_sum += (1.0 / len(set(column_count)))/Nd
            column_count = np.zeros(Nd)


        # For each block...
        for i in range(0, Nd, 3):
            for j in range(0, Nd, 3):
                block_count[self.values[i][j]-1] += 1
                block_count[self.values[i][j+1]-1] += 1
                block_count[self.values[i][j+2]-1] += 1
                
                block_count[self.values[i+1][j]-1] += 1
                block_count[self.values[i+1][j+1]-1] += 1
                block_count[self.values[i+1][j+2]-1] += 1
                
                block_count[self.values[i+2][j]-1] += 1
                block_count[self.values[i+2][j+1]-1] += 1
                block_count[self.values[i+2][j+2]-1] += 1

                block_sum += (1.0/len(set(block_count)))/Nd
                block_count = np.zeros(Nd)

        # Calculate overall fitness.
        if (int(row_sum) == 1 and int(column_sum) == 1 and int(block_sum) == 1):
            fitness = 1.0
        else:
            fitness = column_sum * block_sum
        
        self.fitness = fitness
        return
        
    def mutate(self, mutation_rate, given):
        """ Mutate a candidate by picking a row, and then picking two values within that row to swap. """

        r = random.uniform(0, 1.1)
        while(r > 1): # Outside [0, 1] boundary - choose another
            r = random.uniform(0, 1.1)
    
        success = False
        if (r < mutation_rate):  # Mutate.
            while(not success):
                row1 = random.randint(0, 8)
                row2 = random.randint(0, 8)
                row2 = row1
                
                from_column = random.randint(0, 8)
                to_column = random.randint(0, 8)
                while(from_column == to_column):
                    from_column = random.randint(0, 8)
                    to_column = random.randint(0, 8)   

                # Check if the two places are free...
                if(given.values[row1][from_column] == 0 and given.values[row1][to_column] == 0):
                    # ...and that we are not causing a duplicate in the rows' columns.
                    if(not given.is_column_duplicate(to_column, self.values[row1][from_column])
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
        return
        
    def is_row_duplicate(self, row, value):
        """ Check whether there is a duplicate of a fixed/given value in a row. """
        for column in range(0, Nd):
            if(self.values[row][column] == value):
               return True
        return False

    def is_column_duplicate(self, column, value):
        """ Check whether there is a duplicate of a fixed/given value in a column. """
        for row in range(0, Nd):
            if(self.values[row][column] == value):
               return True
        return False

    def is_block_duplicate(self, row, column, value):
        """ Check whether there is a duplicate of a fixed/given value in a 3 x 3 block. """
        i = 3*(int(row/3))
        j = 3*(int(column/3))

        if((self.values[i][j] == value)
           or (self.values[i][j+1] == value)
           or (self.values[i][j+2] == value)
           or (self.values[i+1][j] == value)
           or (self.values[i+1][j+1] == value)
           or (self.values[i+1][j+2] == value)
           or (self.values[i+2][j] == value)
           or (self.values[i+2][j+1] == value)
           or (self.values[i+2][j+2] == value)):
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
        c1 = candidates[random.randint(0, len(candidates)-1)]
        c2 = candidates[random.randint(0, len(candidates)-1)]
        f1 = c1.fitness
        f2 = c2.fitness

        # Find the fittest and the weakest.
        if(f1 > f2):
            fittest = c1
            weakest = c2
        else:
            fittest = c2
            weakest = c1

        selection_rate = 0.85
        r = random.uniform(0, 1.1)
        while(r > 1):  # Outside [0, 1] boundary. Choose another.
            r = random.uniform(0, 1.1)
        if(r < selection_rate):
            return fittest
        else:
            return weakest
    
class CycleCrossover(object):
    """ Crossover relates to the analogy of genes within each parent candidate mixing together in the hopes of creating a fitter child candidate. Cycle crossover is used here (see e.g. A. E. Eiben, J. E. Smith. Introduction to Evolutionary Computing. Springer, 2007). """

    def __init__(self):
        return
    
    def crossover(self, parent1, parent2, crossover_rate):
        """ Create two new child candidates by crossing over parent genes. """
        child1 = Candidate()
        child2 = Candidate()
        
        # Make a copy of the parent genes.
        child1.values = np.copy(parent1.values)
        child2.values = np.copy(parent2.values)

        r = random.uniform(0, 1.1)
        while(r > 1):  # Outside [0, 1] boundary. Choose another.
            r = random.uniform(0, 1.1)
            
        # Perform crossover.
        if (r < crossover_rate):
            # Pick a crossover point. Crossover must have at least 1 row (and at most Nd-1) rows.
            crossover_point1 = random.randint(0, 8)
            crossover_point2 = random.randint(1, 9)
            while(crossover_point1 == crossover_point2):
                crossover_point1 = random.randint(0, 8)
                crossover_point2 = random.randint(1, 9)
                
            if(crossover_point1 > crossover_point2):
                temp = crossover_point1
                crossover_point1 = crossover_point2
                crossover_point2 = temp
                
            for i in range(crossover_point1, crossover_point2):
                child1.values[i], child2.values[i] = self.crossover_rows(child1.values[i], child2.values[i])

        return child1, child2

    def crossover_rows(self, row1, row2): 
        child_row1 = np.zeros(Nd)
        child_row2 = np.zeros(Nd)

        remaining = range(1, Nd+1)
        cycle = 0
        
        while((0 in child_row1) and (0 in child_row2)):  # While child rows not complete...
            if(cycle % 2 == 0):  # Even cycles.
                # Assign next unused value.
                index = self.find_unused(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row1[index]
                child_row2[index] = row2[index]
                next = row2[index]
                
                while(next != start):
                    index = np.where(row1 == next)[0][0]
                    child_row1[index] = row1[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row2[index]
                    next = row2[index]
            else:  # Odd cycles.
                # Assign next unused value.
                index = self.find_unused(row2, remaining)
                start = row2[index]
                remaining.remove(row2[index])
                child_row1[index] = row1[index]
                child_row2[index] = row2[index]
                next = row1[index]
                
                while(next != start):
                    index = np.where(row2 == next)[0][0]
                    child_row1[index] = row1[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row2[index]
                    next = row1[index]

            cycle += 1
            
        return child_row1, child_row2

    def find_unused(self, row, remaining):
        """ Find next unused value in row. """
        for index in range(0, Nd):
            if(row[index] in remaining):
                return index
        return None


def solve(givens, Nc, mutation_rate, crossover_rate, Nm):
    """ Evolve generations of candidates. """
    population = Population()
    population.seed(Nc, givens)
    generation = 1
    
    while(True):
        for i in range(0, Nc):
            population.candidates[i].mutate(mutation_rate, givens)
        
        for i in range(0, Nc, 2):
            parent1 = tournament.compete(population.candidates)
            parent2 = tournament.compete(population.candidates)
            population.candidates[i], population.candidates[i+1] = cycle_crossover.crossover(parent1, parent2, crossover_rate)
        
        population.update_fitness()
        population.sort()
        print("Generation:", generation, " Fittest:", population.candidates[0].fitness)
        if(population.candidates[0].fitness == 1):
            print("Solution found.")
            return population.candidates[0]
        elif(generation == Nm):
            print("Maximum generations reached.")
            return population.candidates[0]

        generation += 1
        

# Random Sudoku Puzzle Generation Code
def generate_sudoku():
    """ Generate a random valid Sudoku puzzle. """
    grid = np.zeros((Nd, Nd), dtype=int)
    
    for i in range(Nd):
        for j in range(Nd):
            num = random.randint(1, 9)
            while not is_valid_move(grid, i, j, num):
                num = random.randint(1, 9)
            grid[i][j] = num
    
    return grid


def is_valid_move(grid, row, col, num):
    """ Check if placing a number in a specific position is valid. """
    # Check if the number exists in the row or column.
    for i in range(Nd):
        if grid[row][i] == num or grid[i][col] == num:
            return False
    
    # Check if the number exists in the 3x3 box.
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if grid[start_row + i][start_col + j] == num:
                return False
    
    return True


# Initialize Sudoku puzzle
random_sudoku = generate_sudoku()
print("Random Sudoku Puzzle:")
print(random_sudoku)

# Convert Sudoku puzzle into a 'Given' object
given_sudoku = Given(random_sudoku)

# Parameters for genetic algorithm
Nc = 100  # Number of candidates
mutation_rate = 0.05
crossover_rate = 0.8
Nm = 1000  # Maximum number of generations

# Initialize genetic algorithm components
population = Population()
tournament = Tournament()
cycle_crossover = CycleCrossover()

# Solve Sudoku puzzle using genetic algorithm
solution = solve(given_sudoku, Nc, mutation_rate, crossover_rate, Nm)

print("\nSolution:")
print(np.array(solution.values))
