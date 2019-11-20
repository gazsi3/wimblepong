from wimblepong import Wimblepong
import numpy as np

class Agent(object):
    def __init__(self, env, player_id=2):
        if type(env) is not Wimblepong:
            raise TypeError("I'm not a very smart AI. All I can play is Wimblepong.")
        self.env = env
        self.player_id = player_id         
        self.name = "buzibogyo"

    def get_name(self):

        return self.name

    def get_action(self, ob=None):

        action = 0
        ac = np.random.rand(1)
        if(ac>0.2):
            action = self.env.MOVE_UP  # Up
        if(ac>0.6):
            action = self.env.MOVE_DOWN  # Down

        return action

    def load_model(self):
        # tbd
        return

    def reset(self):
        # tbd
        return
    