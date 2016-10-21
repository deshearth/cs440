import string
import os
import numpy as np
from sudokuProblem import Sudoku, solveSudoku 
from sudokuProblem import sliceBlock, selectUnassignedVar
import numpy as np
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
    

def readFile(fileName):
    f = open(fileName, 'r')
    content = [list(eachLine.strip()) for eachLine in f]
    f.close()
    return content
def testSliceBlock():

    # sliceBlock test
    grid = np.ones((9,9))
    var1 = 'abcd'
    vals =  [('V', (0,0)), ('V', (3,4)), ('H', (1,2))]
    print var1, vals
    #print sliceBlock(grid, var1, vals[2], 'word')
    print ['word', 'row', 'col']
    print [sliceBlock(grid, var1, val, option) for val in vals 
        for option in ['word', 'row', 'col']]

def testMRV():
    pass

    
