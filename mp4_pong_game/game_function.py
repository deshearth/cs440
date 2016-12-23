from collections import defaultdict
import math
import random

#constants
paddle_height = 0.2
MOVESTEP = 0.04
inf = float('inf')
TERMINALSTATE = (inf, inf, inf, inf, inf)

def discretize(tag):
    "parameterized decorator for discretization"
    def decorator(func):
        def wrapper(*args, **kwargs):
            if tag is "percept":
                state, reward = func(*args, **kwargs)
                ball_x, ball_y, v_x, v_y, paddle_y = state
                d_ball_x = min(11, round(ball_x*12))
                d_ball_y = min(11, round(ball_y*12))
                d_v_x = sgn(v_x)
                d_v_y = 0 if abs(v_y) < 0.015 else sgn(v_y)
                d_paddle_y = min(11,math.floor(12 * paddle_y / (1-paddle_height)))
                d_state = (d_ball_x, d_ball_y, d_v_x, d_v_y, d_paddle_y)
                return (d_state, reward)
            if tag is "action":
                return map(sgn, func(*args, **kwargs)) if func(*args, **kwargs) != [None] \
                        else [None]
        return wrapper
    return decorator

class QLearningAgent:
    
    def __init__(self, mdp, Ne, Rplus, alpha=None):
        self.gamma = mdp.gamma
        self.actions = mdp.actions
        self.Ne = Ne
        self.Rplus = Rplus
        self.Q = defaultdict(float)
        self.Nsa = defaultdict(float)
        self.s = None
        self.a = None
        self.r = None
        self.states = set()
        
         
        if alpha:
            self.alpha = alpha
        else:
            self.alpha = lambda n: 1./(1+n)

    def game_init(self):
        self.s = None
        self.a = None
        self.r = None
    
    def f(self, u, n):
        if n < self.Ne:
            return self.Rplus
        else:
            return u
            
    @discretize("action")
    def actions_in_state(self, mdp):
        if mdp.isterminal():
            return [None]
        else:
            #random.shuffle(self.actions)
            return self.actions

    def __call__(self, mdp, percept):

        sprime, rprime = self.update_state(percept)
        self.r = rprime
        Q, Nsa, s, a = self.Q, self.Nsa, self.s, self.a
        alpha, gamma, actions_in_state, isterminal = self.alpha, self.gamma, \
                                                     self.actions_in_state, mdp.isterminal

        if isterminal():
            sprime = TERMINALSTATE
        if s is not None:
            Nsa[s, a] += 1
            Q[s, a] += alpha(Nsa[s, a]) * (rprime + gamma * max(Q[sprime, aprime] for aprime in actions_in_state(mdp))
                                             - Q[s, a])
        self.s = sprime
        if isterminal():
            self.a = None
            return None
        else:
            self.a = max(actions_in_state(mdp), key=lambda aprime: self.f(Q[sprime, aprime], Nsa[sprime, aprime]))
            self.states.add(sprime)
            return self.a * MOVESTEP

    @discretize("percept")
    def update_state(self, percept):
        "percept is (state, reward)"
        return percept

    
class PongMDP:

    def __init__(self, init=(0.5, 0.5, 0.03, 0.01, 0.5-paddle_height/2), 
                 actions=[0, 0.04, -0.04], gamma=.9):
        self.init = init
        self.gamma = gamma
        self.actions = actions
        self.prev_state = init
        self.state = init
        self.bounceCnt = 0

    
    def R(self, mdp):
        Nsa, s, a = mdp.Nsa, mdp.s, mdp.a
        punish_factor = (Nsa[s, a]+2.) / (Nsa[s, a]+10.)
        reward_factor = (Nsa[s, a]+5.) / (Nsa[s, a]+10.)
        if self.isterminal():
            return -1 * punish_factor 
            #return -1
        elif self.isbounce():
            self.bounceCnt += 1
            #return 1
            return 1 * reward_factor
        else:
            return 0

    def T(self, action):
        if action is not None:
            self.move(action)

    def move(self, action):
        self.prev_state = self.state
        ball_x, ball_y, v_x, v_y, paddle_y = self.state
        ball_x += v_x
        ball_y += v_y
        paddle_y += action
        if paddle_y < 0:
            paddle_y = 0
        if paddle_y + paddle_height > 1:
            paddle_y = 1 - paddle_height

        if ball_y < 0:
            ball_y = -ball_y
            v_y = -v_y
        if ball_y > 1:
            ball_y = 2 - ball_y
            v_y = -v_y
        if ball_x < 0:
            ball_x = -ball_x
            v_x = -v_x
        if ball_x >= 1:
            if paddle_y <= ball_y <= paddle_y+paddle_height:
                ball_x = 2 - ball_x
                v_x = -v_x + random.uniform(-0.015, 0.015)
                #v_x = -v_x
                v_y = v_y + random.uniform(-0.03, 0.03)
                #v_y = v_y
                if abs(v_x) < 0.03:
                    v_x = sgn(v_x) * 0.03
                if abs(v_x) > 1:
                    v_x = sgn(v_x)
                if abs(v_y) > 1:
                    v_y = sgn(v_y)
        self.state = tuple(map(lambda x: round(x, 3), 
                           (ball_x, ball_y, v_x, v_y, paddle_y)))



    def isterminal(self):
        return True if self.state[0] > 1 else False
    
    def isbounce(self):

        return True if (self.prev_state[2] > 0 and self.state[2] < 0) else False

#-------------
#helper function
#-------------
def sgn(d):
    if d < 0:
        return -1
    elif d > 0:
        return 1
    else:
        return 0



def run_single_trial(agent_program):

    mdp = PongMDP()
    agent_program.game_init()
    #print len(agent_program.Q.keys())
    while True:
        current_reward = mdp.R(agent_program)
        current_state = mdp.state
        #print current_state
        percept = (current_state, current_reward)
        next_action = agent_program(mdp, percept)
        if next_action is None:
            break
        mdp.T(next_action)
    return mdp.bounceCnt


