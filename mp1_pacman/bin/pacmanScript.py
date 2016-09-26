#!/usr/bin/env python
import os
import string
from pacmanProblem import Pacman
from pacmanProblem import State
from pacmanProblem import solvePacman

def script(fileName, searchStrategy):

    binPath = os.getcwd()
    mpPath = os.path.dirname(binPath)
    inputsPath = mpPath + '/tests/inputs/'
    outputsPath = mpPath + '/tests/outputs/'

    f = open(inputsPath + fileName,'r')
    maze = [eachLine.strip() for eachLine in f]
    f.close()
    wallInfo = []
    foodInfo = []
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i][j] == '%':
                wallInfo.append((i,j))
            if maze[i][j] == 'P':
                startLoc = (i,j)
            if maze[i][j] == '.':
                foodInfo.append((i,j))  
    initialState = State(startLoc,foodInfo)
    actions = ['E', 'S', 'W', 'N']
    initialPathCost = 0
    pman = Pacman(initialState, actions, wallInfo, list(foodInfo))
    solution,nodeNum = solvePacman(pman, searchStrategy)
    solutionInfo = {'solution': solution,'nodeNum': nodeNum}
    
    
    writeSolution(outputsPath, inputsPath, fileName, startLoc, solutionInfo,
                  foodInfo, searchStrategy)
 
def writeSolution(outputsPath,inputsPath, fileName, startLoc, solutionInfo,
                  foodInfo, searchStrategy):

    fName = fileName[:-4] + searchStrategy + 'Solution.txt'
    f = open(inputsPath + fileName,'r')
    maze = [list(eachLine.strip()) for eachLine in f]
    f.close()
    pLoc = startLoc
    gth = 0
    label = string.digits + string.letters
    solution = solutionInfo['solution']
    solutionPath = list(solution)

    maze[startLoc[0]][startLoc[1]] = 'P'
    while solution:
        action = solution.pop(0)
        if action == 'E':
            row,col = pLoc[0], pLoc[1] + 1
        if action == 'S':
            row,col = pLoc[0] + 1, pLoc[1]
        if action == 'W':
            row,col = pLoc[0], pLoc[1] - 1
        if action == 'N':
            row,col = pLoc[0] - 1, pLoc[1]
        
        
        if (row,col) not in foodInfo:
            maze[row][col] = '.'
        elif maze[row][col] not in label:
            maze[row][col] = label[gth % len(label)]
            gth += 1
        else:
            pass
      
        pLoc = (row,col)
    f = open(outputsPath + fName,'w')
    [f.write(''.join(line[:]+['\n'])) for line in maze]

    f.close()
    pathCost = len(solutionPath)
    solutionPath = '->'.join(solutionPath)
    
    if pathCost > 10:

        solutionPath = ''.join(solutionPath[i:i+30] + \
                '\n' for i in range(0,len(solutionPath),30))
    nodeNum = solutionInfo['nodeNum']
    
    f = open(outputsPath + fName,'a')

    printInfo = '''%sSolution Path:\n%s\nPath Cost: %d\nPop Node Number: %d''' \
                    % ('\n'*3,solutionPath, pathCost, nodeNum) 

    f.write(printInfo)
     
    f.close()
            
if __name__=='__main__':


    fileName = raw_input("Input the file name: ")
    print '''
    [a] Breadth First Search
    [b] Depth First Search
    [c] Greedy Search
    [d] AStar Search
    [e] Nearest First(for Multiple Dots)
    '''
    choices = raw_input("Enter Your Choice(separate by ','): ")
    for choice in choices.split(','):
        choice = choice.strip(' ')
        
        if choice == 'a':
            strategy = 'BreadthFirstSearch'
        if choice == 'b':
            strategy = 'DepthFirstSearch'
        if choice == 'c':
            strategy = 'GreedySearch'
        if choice == 'd':
            strategy = 'AStarSearch'
        if choice == 'e':
            strategy = 'NearestFood'
        print "Running %s Strategy Now..." % (strategy)
        script(fileName, strategy)
        print "%s Strategy Done!\n\n" % (strategy)

