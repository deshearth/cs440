
import operator
import math
import bisect
from dataStructure import (FIFOQueue, Stack, PriorityQueue)
import copy 

class State:
    def __init__(self, location, foodLeft):
        self.location = location
        self.foodLeft = foodLeft

    def newLoc(self, action):
        east = (0,1) 
        south = (1,0)
        west = (0,-1)
        north = (-1,0)
        if action == 'E':
            return self.tupleSum(self.location, east)
        elif action == 'S':
            return self.tupleSum(self.location, south)
        elif action == 'W':
            return self.tupleSum(self.location, west)
        else:
            return self.tupleSum(self.location, north)

    def tupleSum(self, t1, t2):
        return tuple(sum(x) for x in zip(t1,t2))
            
    def moveToWall(self, action, wallInfo):
        return self.newLoc(action) in wallInfo

    def __eq__(self, other):
        return (isinstance(other, State)) and self.location == other.location\
                                          and self.foodLeft == other.foodLeft
    def __repr__(self):
        return "State: %s,%s" % (str(self.location), str(self.foodLeft))

    def __hash__(self):
        return hash(self.__repr__())
class Node:
    
    def __init__(self, state, parent=None, action=None, pathCost=0, popth=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.pathCost = pathCost

    def childNode(self, problem, action):

        state = problem.actionResult(self.state, action)
        pathCost = self.pathCost + problem.pathCost(self.state, action)
        return Node(state, self, action, pathCost)

    def expand(self, problem):
        legalActions = problem.getLegalActions()
        children = [self.childNode(problem, action) for action in legalActions]
        return children

    def solution(self):
        return [node.action for node in self.path()[1:]] 
        
    def path(self):
        node, pathBack = self, []
        while node:
            pathBack.append(node)
            #print node
            node = node.parent
        return list(reversed(pathBack))

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

    def __repr__(self):
        return "[ Node: <%s> <PathCost:%s>]" % (self.state.__repr__(), 
                                                self.pathCost)
                                  


def ManhattanDistance(posA, posB):
    return abs(posA[0]-posB[0]) + abs(posA[1] - posB[1])

class Pacman:

    def __init__(self, initialState, actions, wallInfo, foodInfo):
        self.initialState = initialState
        self.state = initialState
        self.wallInfo = wallInfo
        self.actions = actions
        self.foodInfo = foodInfo
        # some parameters
        self.mazeSize = (wallInfo[-1][0] - 1, wallInfo[-1][1] - 1)
        self.mazeArea = self.mazeSize[0] * self.mazeSize[1]
        edgeWall = (self.mazeSize[0] + self.mazeSize[1]) * 2 + 4
        self.wallPercent = float((len(wallInfo) - edgeWall)) / self.mazeArea 
        self.foodPercent = float(len(foodInfo)) / self.mazeArea
         
        

    def getLegalActions(self):
        return [action for action in self.actions 
                if not self.state.moveToWall(action, self.wallInfo)]
     
    def actionResult(self, state, action):
        newState = State(state.newLoc(action), list(state.foodLeft))
        if newState.location in state.foodLeft:
            newState.foodLeft.remove(newState.location)
        return newState
        
    def goalTest(self, state):
        #print state.location, state.location in self.foodInfo,state.foodLeft
        return (state.location in self.foodInfo) and not state.foodLeft
     
    def pathCost(self, state, action):
        return 1

    #def heuristic(self, node, choose):
    #    measure = ManhattanDistance
    #    agent = node.state.location
    #    foodLeft = node.state.foodLeft
    #    if choose == 'sumDis':
    #        hval = self.sumDistance(measure, agent, foodLeft)
    #    if choose =='nearestDis':
    #        hval = self.nearestFood(measure, agent, foodLeft)
    #        
    #    return hval

    def sumDistance(self, node):
        measure = ManhattanDistance
        agent = node.state.location
        foodLeft = node.state.foodLeft

        return 0 if not foodLeft else sum(measure(agent,food) \
                for food in foodLeft)/len(foodLeft) + len(foodLeft) - 1


    def nearestFood(self, node):

        measure = ManhattanDistance
        agent = node.state.location
        foodLeft = node.state.foodLeft
        #print '====MST====',self.MST(node)
        if node.parent and (not foodLeft or agent in node.parent.state.foodLeft):
        #if not foodLeft or agent in self.foodInfo:
            #print "===Nearest===,0"
            return 0
        else:
            #h = nearestVertex(agent, foodLeft)[1]
            #if (node.pathCost > 2 and
            #        node.parent.state.location == node.state.location):
            #    return h
            #else:
            #    return h + 0.5
            h = min(measure(agent, food) for food in foodLeft)
            #print '===Nearest====',h
            return h

        

    def MST(self, node):
        h = 0.0
        agent = node.state.location
        connected = [agent]
        candidates = list(node.state.foodLeft)
        elect = nearestVertex
        firstRoundWin = []
        #n = 0
        while candidates:
        # first election is to pick the nearest node for each vertex
        # in connect vertices
            for vertex in connected:
                firstRoundWin.append(elect(vertex, candidates))

            winner = min(firstRoundWin, key=operator.itemgetter(1))
            #print '++++++++',winner
            del firstRoundWin[:]
            
            h += winner[1]
            #n += 1
            connected.append(winner[0])
            candidates.remove(winner[0])
            #if n > 4:
            #    break
       
        #if (node.pathCost > 2 and 
        #    node.parent.parent.state.location == node.state.location):

        #    return h + 0.5
        #else:
        #    return h
        #h *= 1 + float(len(node.state.foodLeft)) / len(self.foodInfo)
        return h
    def nnNearest(self, node):
        h = 0.0
        vertex = node.state.location
        candidates = list(node.state.foodLeft)
        elect = nearestVertex
        while candidates:
        # first election is to pick the nearest node for each vertex
        # in connect vertices
             
            winner = nearestVertex(vertex, candidates)
            h += winner[1]
            vertex = winner[0]
            candidates.remove(vertex)
            
        #h *= 1 + pow(math.e, float(len(node.state.foodLeft)) / len(self.foodInfo))
        return h

def nearestVertex(vertex, candidates):
    measure = ManhattanDistance
    winner = {candidate: measure(vertex, candidate) for candidate in candidates}
    return min(winner.iteritems(), key=operator.itemgetter(1))
        

        

#======================================================#

def solvePacman(pman, searchStrategy):
    if searchStrategy == 'BreadthFirstSearch':
        goalNode, nodeNum = BFS(pman)
    elif searchStrategy == 'DepthFirstSearch':
        goalNode, nodeNum = DFS(pman)
    elif searchStrategy == 'GreedySearch':
        goalNode, nodeNum  = greedy(pman)
    elif searchStrategy == 'AStarSearch':
        goalNode, nodeNum  = AStar(pman)
    elif searchStrategy == 'NearestFood':
        goalNode, nodeNum  = nearest(pman)
    else:
        print 'No implementation of this strategy'
        goalNode = None
    return (None, nodeNum) if not goalNode else (goalNode.solution(),nodeNum)
 

def BFS(problem):
   
    return graphSearch(problem, FIFOQueue) 

def DFS(problem):
    return graphSearch(problem, Stack)

def nearest(problem):
    h = problem.nnNearest
    return fSearch(problem, lambda node: node.pathCost + h(node), 'nearest')
    


def graphSearch(problem, Frontier):
    nodeNum = 0
    frontier = Frontier()
    frontier.append(Node(problem.initialState))
    explored = set()
    while frontier:
        node = frontier.pop()
        nodeNum += 1
        problem.state = node.state
        if problem.goalTest(node.state):
            return node, nodeNum
        explored.add(node.state)
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and
                        child not in frontier)
    return None, nodeNum


def greedy(problem):
    h = problem.MST
    return fSearch(problem, h)

def AStar(problem):
    h = problem.MST
    return fSearch(problem, lambda node: node.pathCost + h(node))

def fSearch(problem, f, *nearest):

    nodeNum = 0
    frontier = PriorityQueue(min, f) 
    frontier.append(Node(problem.initialState))
    explored = set()
    parentFoodLeft = problem.foodInfo
    threshold = float('inf')
    #threshold = 5
    #record = len(problem.foodInfo)
    while frontier:
        node = frontier.pop()
        #if f(node) > threshold and len(frontier) > 3000:
        #    print "================cut branch============"
        #    continue
        nodeNum += 1
        if node.state.location in parentFoodLeft and nearest:
            frontier.clear()
        #if nodeNum > 50000 and node.state.location in parentFoodLeft:
        #    print "++++++++++++frontier cleared++++++++++"
            frontier.clear()
        #if record - len(node.state.foodLeft) > threshold:
        #    frontier.clear()
        #    record = len(node.state.foodLeft)
        #if node.state.location in parentFoodLeft:
        #    threshold = f(node)
            
        #print '=============\n',node
        print '(loc: %s, cost: %d, h: %f, f: %f, foodLeft: %d)' % \
                                                     (str(node.state.location),
                                                     node.pathCost,
                                                     f(node)-node.pathCost,
                                                     f(node),
                                                     len(node.state.foodLeft))
        
        parentFoodLeft = node.state.foodLeft
        problem.state = node.state
        if problem.goalTest(node.state):
            return node, nodeNum
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
            
            
    return None, nodeNum

            
        
