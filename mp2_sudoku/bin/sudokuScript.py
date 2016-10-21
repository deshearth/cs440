#!usr/bin/env python
import string
import os
import numpy as np
from sudokuProblem import Sudoku, solveSudoku 
#from sudokuTest import Sudoku, solveSudoku 

import time

def script():
    binPath = os.getcwd()
    mpPath = os.path.dirname(binPath)
    inGridPath = mpPath + '/tests/inputs/grids/'
    inBankPath = mpPath + '/tests/inputs/banks/'
    outputsPath = mpPath + '/tests/outputs/'

    # read file
    gridBase = 'grid1'
    bankBase = 'bank1'
    suffix = '.txt'
    grid = np.array(readFile(inGridPath + gridBase + suffix))
    wordBank = [''.join(line) for line in 
                readFile(inBankPath + bankBase + suffix)]

    # solve problem
    sudoku = Sudoku(grid, wordBank, False)
    #sudoku = Sudoku(grid, wordBank, True)
    result = solveSudoku(sudoku)

def readFile(fileName):
    f = open(fileName, 'r')
    content = [list(eachLine.strip().lower()) for eachLine in f]
    f.close()
    return content


if __name__ == '__main__':
    start = time.time()
    script()
    end = time.time()
    print end - start
