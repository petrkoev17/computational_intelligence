import numpy as np

from QLearning import QLearning


class MyQLearning(QLearning):
    def update_q(self, state, action, r, state_next, possible_actions, alfa, gamma):
        # TODO Auto-generated method stub
        max_q_next = np.max([self.get_q(state_next, a) for a in possible_actions])
        old_q = self.get_q(state, action)
        new_q = old_q + alfa * (r + gamma * max_q_next - old_q)
        self.set_q(state, action, new_q)
        return
