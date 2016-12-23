from game_function import QLearningAgent, PongMDP, run_single_trial
def script():
    q_agent = QLearningAgent(sequential_decision_env, Ne=5, Rplus=2,
                             alpha=lambda n: 60./(59+n))
    for i in range(200):
            run_single_trial(q_agent)

if __name__=='__main__':
    script()
