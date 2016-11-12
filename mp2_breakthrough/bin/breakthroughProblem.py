import numpy as np
inf = float('inf')
white = 1
black = -1

class Breakthrough:
    
    def __init__(self, boardSize=(8, 8), order=(white, black)):
    #def __init__(self, boardSize, order):
        self.boardSize = boardSize 
        self.initial = self.initialize()
        self.state = self.initial
        self.actions = 'only move forward or diagnolly forward'
        self.clock = 0 
        self.order = order 
        

    def initialize(self):
        'white is always at the top and black is always at the bottom'
        board = np.zeros(self.boardSize)
        board[0:2,:] = white # white
        board[-1:-3:-1,:] = black #black
        state = State(board)
        state.update()
        return state        
    


    def player(self, node):
        #if node.gameNode:
        #    return order[(node.depth + node.)]
        return order[(node.depth + node.gameNode.depth) % 2]
         
        
    def legalActions(self, node):
        state = node.state
        direction = self.player(node)
        player = self.player(node)
        pieces = tupleIdx(state.piecePos[player])
        actions = {}
        #for piece in pieces:
        #    actions[piece] = self.availableMove(state, piece)

        #return actions
            
        actions = {piece: self.availableMove(node, piece) 
                    for piece in tupleIdx(state.piecePos[player])}
        return [(loc, move) for loc in actions.keys() 
                for move in actions[loc] if actions[loc] ]
        #return dict(((k, v) for k, v in actions.iteritems() if v))
    def availableMove(self, node, piece):
        'return a tuple containing boolen value liek (True False False)' 
        state = node.state
        rows, cols = self.boardSize
        player = self.player(node)
        oppo = -player
        currR, currC = piece #current row and col piece is in 
        march = player
        # left diag is true iff currC is within the boundary and 
        # there is not comrade
        move = []
        if currC > 0  and state.board[currR+march, currC-1] != player:
            move.append(-1)
        if not state.board[currR+march, currC]:
            move.append(0)
            #print currC
        if currC < cols- 1 and state.board[currR+march, currC+1] != player:
            
            move.append(1)
        return move
        #leftDiag = False if currC < leftEdge else \
        #        not state.piecePos[player][currR+march, currC-1]
        ## right diag is true iff currC is within the boundary and 
        ## there is not comrade
        #rightDiag = False if currC < rightEdge else \
        #        not state.piecePos[player][currR+march, currC+1]
        ## if piece is present in forward position, forward is false
        #forward = not bool(state.board[currR+march, currC])

        #return (leftDiag, forward, rightDiag)



    def result(self, node, action):
        'return a new state'
        state = node.state
        player = self.player(node)
        oppo = -player
        #print player
        march = player
        currR, currC = action[0]
        moveTo = (currR+march, currC+action[1]) 
        #if state.board[moveTo] == oppo:
        #    print 'in action %s, %d is captured by %d' % (action, player, oppo)
        #    print 'moveTo', moveTo

        #print action, action[1]
        #print type(moveTo)
        #print moveTo
        newState = state.copy() 
        newState.board[currR, currC], newState.board[moveTo] = 0, player
        newState.update() 
        #print 'in result', player
        #print action
        #print newState.board
        return newState
            

    def terminalTest(self, node):
        state = node.state
        player = self.player(node)
        return (any(leftPiece == 0 for leftPiece in state.pieceNum.values()) or
              np.any(state.board[0] == -1) or
              np.any(state.board[-1] == 1))



    def utility(self, node):
        state = node.state
        player = self.player(node)
        defenseLine = min(player, 0)
        return -50 if (any(state.board[defenseLine] == -player) or 
                state.pieceNum[player] == 0) else 50

    def evaluation(self, node):
        state = node.state
        player = self.player(node)
        oppo = -player 
        if self.terminalTest(node):
            return utility(node)
       
        return state.pieceNum[player] - state.pieceNum[oppo]
        #if evaluation:
        #    print 'evaluation %d, player%d node dep %d, gameNode%d, action: %s, %s, %s' % (
        #                                                evaluation, player,
        #                                                node.depth,
        #                                                node.gameNode.depth,
        #                                                node.parent.parent.action, 
        #                                                node.parent.action,
        #                                                node.action)
        #return evaluation

    def offensive(self, node):
        state = node.state
        player = self.player(node)
        oppo = -player



    def defensive(self, state):
        return 0

def tupleIdx(boolIdx):
    'convert bool index to tuple'
    ridx, cidx = np.where(boolIdx)
    return zip(tuple(ridx), tuple(cidx))

    

class State:
    
    def __init__(self, board, piecePos={}, pieceNum={}):
        self.board = board
        self.piecePos = piecePos
        self.pieceNum = pieceNum

    def copy(self):
        return State(np.array(self.board),
                     dict(self.piecePos), 
                     dict(self.pieceNum)) 
    def update(self):
        for player in [white, black]:
            self.piecePos[player] = self.board == player
            self.pieceNum[player] = np.count_nonzero(self.piecePos[player])
        

class Node:
    
    def __init__(self, state, action=None, parent=None, gameNode=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.gameNode = gameNode
        self.depth = self.getDepth()

    def childNode(self, game, action):
        return Node(game.result(self, action), action, self, self.gameNode)
    def getDepth(self):
        depth = 0
        parent = self.parent
        while parent:
            depth += 1
            parent = parent.parent
        return depth
    def searchTreeRoot(self):
        return Node(self.state, self.action, None, self)
        


def gaming(game):
    gameRoot = Node(game.initial)
    node = Node(gameRoot.state, gameRoot.action, gameRoot.parent, gameRoot )
    while not game.terminalTest(node):
        node = firstPlayer(game, node)
        print "first player's move done"
        print node.state.board
        wait()
        node = secondPlayer(game, node)
        print "second player's move done"
        print node.state.board
        wait()
    print node.state.board

def wait():
    wait = raw_input('')

def firstPlayer(game, gameNode):
    actions = game.legalActions(gameNode)
    #print '++++actions', actions
    node = gameNode.searchTreeRoot()
    action = minimax(game, node, actions) 
    print 'first player action', action
    return gameNode.childNode(game, action)

def secondPlayer(game, gameNode):
    actions = game.legalActions(gameNode)
    node = gameNode.searchTreeRoot()
    action = minimax(game, node, actions)
    print 'second player action', action
    return gameNode.childNode(game, action)

def minimax(game, node, actions):
    depth = 0
    limit = 2
    #print 'in minimax, nodedepth', node.depth
    return max(actions, 
            key=lambda action: minValue(game,
                node.childNode(game, action), depth, limit))

def minValue(game, node, depth, limit):
    'must be opponent'
    depth += 1
    #print 'in min val, nodedepth', node
    #print 'in min value depth', depth
    if game.terminalTest(node):
        return game.utility(node)
    if depth == limit:
        #print 'evaluation!!'
        return game.evaluation(node)
    v = inf
    #print 'in minVal', game.legalActions(node.state)
    for action in game.legalActions(node):
        #print 'minVal action', action
        #print node.state.board
        #wait = raw_input('...')
        v = min(v, maxValue(game, node.childNode(game, action), depth, limit))
    return v

def maxValue(game, node, depth, limit):
    depth += 1
    #print 'in maxVal, nodedepth', node.depth
    #print 'in max value depth', depth
    if game.terminalTest(node):
        return game.utility(node)
    if depth == limit:
        #print 'evaluation!!'
        return game.evaluation(node)

    v = -inf
    #print 'in macVal', game.legalActions(node.state)
    for action in game.legalActions(node):
        #print 'max val action', action
        #print node.state.board
        #wait = raw_input('...')

        v = max(v, minValue(game, node.childNode(game, action), depth, limit))
    return v 

def alphabeta():
    pass 

boardSize = (8,8)
order = (white, black)
#game = Breakthrough(boardSize, order)
game = Breakthrough()

gaming(game)






