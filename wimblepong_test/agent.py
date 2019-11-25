from wimblepong import Wimblepong
import numpy as np
import gym

class Agent(object):
    def __init__(self):
        self.env = env = gym.make("WimblepongVisualMultiplayer-v0")
        self.name = "buzibogyo"

    def prepro(self,I):
        """ prepro 200x200x3 uint8 frame into 10000 (10x10) 1D float vector """
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 43] = 0 # erase background (background type 1)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        return I.astype(np.float).ravel()

    def get_name(self):

        return self.name

    def get_action(self, ob=None):

        prepro_ob = self.prepro(ob)
        print(ob.shape, prepro_ob.shape)

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
    