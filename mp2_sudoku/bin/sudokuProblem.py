import numpy as np
import operator
import string
import time
# some constant
failure = -1
inf = float('inf')
decoy = ('N/A',(inf, inf))

class Sudoku:

    def __init__(self, grid, wordBank, option):
        # get the stat of word bank, which contains the 
        # number of all letters and count of each letter
        #self.stat = getStat(wordBank)
        self.grid = grid
        self.variables = wordBank
        self.decoy = option  
        #self.stat = self.getStat()
         




    # helper functions for initialization
    def getStat(self):
        'variable is a list of string, count the letters occurence'
        stat = {}
        wordStat = {}
        gridStat = {}
        #seq = ''.join(''.join(line) for line in subject).lower()
        for letter in string.lowercase:
            words = [var for var in self.variables if letter in var]
            if words:
                wordStat[letter] = words
        stat['word'] = wordStat
        if not emptyGrid(self.grid):
            rows, cols = self.grid.shape
            letters = self.grid[self.grid != '_']
            for letter in letters:
                if gridStat.has_key(letter):
                    continue
                gridStat[letter] = np.char.count(letters, letter)
            stat['grid'] = gridStat

        return stat
    
    def assignComplete(self, assignment):
        return False if np.char.count(assignment.grid, '_').sum() \
                     else True

    def constraints(self, assignment, var, val):
        return True if (self.noWordCollision(assignment, var, val) and
            self.ableToFillAll(assignment, var, val) and
            self.allDiff(assignment, var, val)) else False

    def noWordCollision(self, assignment, var, val):
        if val == decoy:
            return True
        pieceFromGrid = fetchBlock(assignment.grid, var, val, 'word').reshape(-1)
        for i in range(len(var)):
            if pieceFromGrid[i] == '_':
                continue
            if pieceFromGrid[i] != var[i]:
                return False
        return True

    def ableToFillAll(self, assignment, var, val):
        #if not self.decoy:
        #    return True
        if self.decoy:
            remainLetters = len(''.join(assignment.unassigned.keys()))
            emptySpot = np.char.count(assignment.grid, '_').sum()
            if remainLetters < emptySpot:
                return False
        #almost useless
        #if self.stat.has_key('grid'):
        #    for letter in self.stat['grid'].keys():
        #        print 'from stat ',letter, len(self.stat['word'][letter])
        #        print 'from counting\n', sum((letter in v 
        #                    for v in assignment.assigned.keys()
        #                    if assignment.assigned[v] == decoy)) 
        #        if (len(self.stat['word'][letter]) - sum((letter in v 
        #                    for v in assignment.assigned.keys()
        #                    if assignment.assigned[v] == decoy))) < 0:
        #            print "messeage from able to fill all"
        #            return False
        return True

                
    def allDiff(self, assignment, var, val):
        if val == decoy:
            return True
        grid = assignment.grid
        u, d, l, r = sliceBlock(grid, var, val, 'word')
        lVals = [(val[0],(val[1][0]+i, val[1][1]+j)) for i in range(0, d-u)
                                                     for j in range(0, r-l)]
        lVars = list(var)
        for option in ['row', 'col', 'square']:
            for i in range(len(var)):
                pieceFromGrid = fetchBlock(grid, lVars[i], lVals[i], option) 
                if lVars[i] in pieceFromGrid and lVars[i] != grid[lVals[i][1]]:
                    return False
        return True
        

class Assign:

    def __init__(self, assigned, unassigned, grid, prevAssign=None):
        self.assigned = assigned
        self.unassigned = unassigned
        self.grid = grid
        self.prevAssign = prevAssign


    def modify(self, var, val, operation):
        self.assigned[var] = val
        del self.unassigned[var]
        if val != decoy:
            writeBlock(self.grid, var, val, operation)

    def operation(self, pairs, operation):
        while pairs:
            var, val = pairs.pop()
            self.modify(var, val, operation)

    def update(self, pairs):
        self.operation(pairs, 'add')

    def reverse(self, pairs):
        self.operator(pairs, 'remove')

    def simulation(self, var, val):
        unassigned = dict(self.unassigned)
        unassigned[var] = [val,]
        grid = np.array(self.grid)
        simAssignment = Assign(dict(self.assigned), unassigned, grid, self)
        return Assign(self.assigned, unassigned, grid, self)
        
    def remove(self, var, val):
        self.assigned[var].remove(val)

    def __del__(self):
        self.prevAssign, self.grid = None, None
        del self.assigned, self.unassigned
        

def sliceBlock(grid, var, val, option):
    'up, down for row, left, right for col'
    rows, cols = grid.shape
    roffset, coffset = (len(var), 1) if val[0] == 'V' else \
                       (1, len(var))
    try:
        u, d, l, r = (val[1][0], val[1][0]+roffset,
                      val[1][1], val[1][1]+coffset)
    except IndexError:
        print 'error at ', var, val, type(val) 

    if option == 'row':
        l, r = (0, cols)
    if option == 'col':
        u, d = (0, rows)
    if option == 'square':
        ULadjust = lambda x: 3 * (x / 3)
        DRadjust = lambda x: 3 * ((x - 1) / 3) + 3 
        u, l = ULadjust(u), ULadjust(l)
        d, r = DRadjust(d), DRadjust(r)
    return u, d, l, r

def fetchBlock(grid, var, val, option):
    u, d, l, r = sliceBlock(grid, var, val, option)
    return grid[u:d, l:r]
                     
def writeBlock(grid, var, val, operation):
    u, d, l, r = sliceBlock(grid, var, val, 'word') 
    content = np.array(list(var)).reshape(d-u, r-l) if operation == 'add' \
            else np.array(['_']*len(var)).reshape(d-u, r-l)
    grid[u:d, l:r] = content

def solveSudoku(csp):
    assigned = {}
    unassigned = getDomains(csp)
    grid = csp.grid
    assignment = Assign(assigned, unassigned, grid) 
    if not emptyGrid(csp.grid):
        #refineDomains(assignment,csp)    
        infer(assignment, csp)
    result = backtrack(assignment, csp)

def refineDomains(assignment, csp):
    for var in assignment.unassigned.keys():
        recycle = set()
        for val in assignment.unassigned[var]:
            if not csp.constraints(assignment, var, val):
                recycle.add(val)
        while recycle:
            assignment.unassigned[var].remove(recycle.pop())

def getDomains(csp): 
    '''reduce the domain for each variable, which is like first
    check in node consistency'''
    row, col = csp.grid.shape
    unassigned = {}
    for var in csp.variables:
        unassigned[var] = []
        unassigned[var].extend([('H', (i, j)) for i in range(row) 
                                          for j in range(col - len(var) + 1)])
        unassigned[var].extend([('V', (i, j)) for i in range(row - len(var) + 1)
                                          for j in range(col)])
        if csp.decoy:
        #unassigned[var].append(('N/A',(inf, inf)))
            unassigned[var].append(decoy)
    return unassigned

def emptyGrid(grid):
    height, width = grid.shape
    return np.char.count(grid, '_').sum() == height * width

def backtrack(assignment, csp):
    #print 'len of assigned', len(assignment.assigned.keys())
    if csp.assignComplete(assignment):
        print assignment.grid
        return assignment
    if not assignment.unassigned:
        return failure
    #print assignment.grid
    #wait = raw_input('''.....''')
    var = selectUnassignedVar(assignment)
    for val in orderDomainVal(assignment,csp, var):
        if csp.constraints(assignment, var, val): # node consistancy
            simAssignment = assignment.simulation(var, val)
            #refineDomains(simAssignment, csp)
            inferences = infer(simAssignment,csp, var)
            #print 'refer time', refertimee - refertimes
            if len(inferences) > 1:
                print "inferences are useful!"
            if inferences != failure:
                simAssignment.update(inferences)
                result = backtrack(simAssignment, csp)
                if result != failure:
                    return result
            del simAssignment
    return failure


    
        
def selectUnassignedVar(assignment):
    '''MRV: find out which variable has fewest remaining variable
    no need to tie breaking since in most cases their remaining variable
    is different even at the beginning'''

    return min(assignment.unassigned.iteritems(),
               key=compose(len, operator.itemgetter(1))
               )[0]

def orderDomainVal(assignment, csp, var):
    sortMethod = lambda val: overlappingLetter(assignment, var, val)
    revOpt = False if csp.decoy else True
    return sorted(assignment.unassigned[var], key=sortMethod, reverse=revOpt)

def overlappingLetter(assignment, var, val):
    if val == decoy:
        return inf
    else:
        pieceFromGrid = fetchBlock(assignment.grid, var, val, 'word').reshape(-1)
        return sum((var[i] == pieceFromGrid[i] for i in range(len(var))))

    

def compose(f,g):
    'compose two functions'
    return lambda x:f(g(x))

def infer(assignment, csp, var=None):
    'maintain arc consistency'
    arcTails = set()
    if not var:
        arcTails |= set(assignment.unassigned.keys())
        initial = True
    else:
        arcTails |= set(dynamicLink(assignment, var))
        initial = False
    while arcTails:
        arcTail = arcTails.pop()
        arcHeads = dynamicLink(assignment, arcTail)
        if reviseInMAC(csp, assignment, arcTail, arcHeads, initial):
            if not assignment.unassigned[arcTail]:
                #if arcTailVal is empty, return failure
                return failure
            arcTails.union(set(arcHeads))
    
    # return the var who only has one value left    
    if var:
        #for (var, val) in assignment.unassigned.iteritems():
        #    print len(val),type(val)
        #return [(var, assignment.unassigned[var][0]) 
        #        for var in assignment.unassigned.keys() if len(val) == 1]

        return [(var, val[0]) for var, val in assignment.unassigned.iteritems()
                                            if len(val) == 1]

def reviseInMAC(csp, assignment, arcTail, arcHeads, initial):
    revised = False
    recycle = set() # maintain a set to record the removed value in tail
    for arcTailVal in assignment.unassigned[arcTail]:
        if not csp.constraints(assignment, arcTail, arcTailVal):
            recycle.add(arcTailVal)
            revised = not initial
            continue
        if not initial:
            #only do the forward checking
            attempt = Assign(dict(assignment.assigned),
                             dict(assignment.unassigned),
                             np.array(assignment.grid),
                             assignment
                            )
            attempt.modify(arcTail, arcTailVal, 'add')
            for arcHead in arcHeads:
                consistant = False
                for arcHeadVal in attempt.unassigned[arcHead]:
                    if csp.constraints(attempt, arcHead, arcHeadVal):

                        consistant = True
                        break # once find existing consisting value in head
                              # move on to check another arc
                if not consistant:
                    recycle.add(arcTailVal)
                    revised = True
                    break # no need to check other heads, 
                          # come back the outmost loop, check other tail values
            del attempt
    # done checking all the neighbors
    # means that all the values in arc tail is illegal
    while recycle:
        assignment.unassigned[arcTail].remove(recycle.pop())
    return revised
    

def dynamicLink(assignment, arcTail):
    '''the dynamicLink between variables is dependent on their 
    remaining values'''
    return [] 
    if len(assignment.unassigned[arcTail]) > 40:
        return [var for var in assignment.unassigned.keys() if var != arcTail]
    sliceToCoord = lambda s: set([(x,y) for x in range(s[0],s[1]) 
                                        for y in range(s[2],s[3])])
    inPowerRange = lambda subjects, powerRange: any(subject in powerRange 
                                                for subject in subjects)  
    powerRange = reduce(lambda x, y : x.union(y),
                    (sliceToCoord(
                        sliceBlock(assignment.grid, arcTail, arcTailVal, option))
                            for option in ['row', 'col', 'square'] 
                                for arcTailVal in assignment.unassigned[arcTail]
                                 if arcTailVal != decoy), set())
    return [var for var in assignment.unassigned.keys() 
            if (inPowerRange(assignment.unassigned[var], powerRange) and 
                var != arcTail)]

        
        


'''
the most constrained variable: having least candidates
the most constraining variable: takes up most space
the least contraining value: takes up least space
'''

'''
whether the variables are constraining each is determined 
by assignment, so their constraining relationship is dynamic
'''
