#!/usr/bin/env python
import string
import os
import numpy as np
from sudokuProblem import Sudoku, solveSudoku 
#from sudokuTest import Sudoku, solveSudoku 
#from fastSudoku import Sudoku, solveSudoku 

import time

def script():
    binPath = os.getcwd()
    mpPath = os.path.dirname(binPath)
    inGridPath = mpPath + '/tests/inputs/grids/'
    inBankPath = mpPath + '/tests/inputs/banks/'
    outputsPath = mpPath + '/tests/outputs/'

    # read file
    gridBase = 'grid2'
    bankBase = 'bank2'
    suffix = '.txt'
    grid = np.array(readFile(inGridPath + gridBase + suffix))
    wordBank = [''.join(line) for line in 
                readFile(inBankPath + bankBase + suffix)]

    # solve problem
    sudoku = Sudoku(grid, wordBank, False)
    #sudoku = Sudoku(grid, wordBank, True)
    start = time.time()
    solutions = solveSudoku(sudoku)
    end = time.time()
    #print solutions.grid
    #assignSeq(soltions)
    print start - end
    

def assignSeq(solutions):
    prev = solutions.prevAssign
    assigned = []
    recycle = set()
    while prev:
        #print prev.assigned
        assigned.append(prev.assigned)
        prev = prev.prevAssign
    recycle = set()
    while assigned:
        seq = assigned.pop()
        varss = seq.keys()
        for var in varss:
            if var not in recycle:
                print var, seq[var]
            recycle.add(var)




def readFile(fileName):
    f = open(fileName, 'r')
    content = [list(eachLine.strip().lower()) for eachLine in f]
    f.close()
    return content


if __name__ == '__main__':
    script()
