import numpy as np
import pickle
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    # player = self.env.player1 if self.player_id == 1 else self.env.player2
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 43] = 0 # erase background (background type 1)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


class Agent(object):
    def __init__(self):
        self.H = 200  # number of hidden layer neurons
        # self.D = 100 * 100  # input dimensionality: 100x100 grid
        self.D = 6  # input dimensionality: 100x100 grid
        self.prev_x = None
        self.model = {}
        self.init_model()
        self.name = "6ComboDestroyer"
        self.rewards = []
        self.model_file = "start.p"
        self.reward_file = "running_rewards.p"
        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
        self.xs,self.hs,self.dlogps,self.drs = [],[],[],[]
        self.running_reward = None
        self.reward_sum = 0
        self.episode_number = 0
        self.batch_size = 10 # every how many episodes to do a param update?
        self.grad_buffer = {}
        self.rmsprop_cache = {}
        self.plot_rewards = []
        self.env = []

        #supervised model params
        self.input_dim = 10000
        self.output_dim = 4
        self.net = {}
        self.sup_model_file = "./supervised_model.pth"
        self.optimizer = None
        self.loss_func = None

    def add_reward(self,reward):
        self.reward_sum += reward
        self.drs.append(reward)

    def count_episode(self):
        self.episode_number += 1

    def train(self,reward):
        self.count_episode()

        epx = np.vstack(self.xs)
        eph = np.vstack(self.hs)
        epdlogp = np.vstack(self.dlogps)
        epr = np.vstack(self.drs)
        self.xs,self.hs,self.dlogps,self.drs = [],[],[],[] # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = self.discount_rewards(epr)
        discounted_epr = discounted_epr.astype('float64')

        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        grad = self.policy_backward(eph, epx, epdlogp)
        for k in self.model: self.grad_buffer[k] += grad[k] # accumulate grad over batch

         # perform rmsprop parameter update every batch_size episodes
        if self.episode_number % self.batch_size == 0:
            for k,v in self.model.items():
                g = self.grad_buffer[k] # gradient
                self.rmsprop_cache[k] = self.decay_rate * self.rmsprop_cache[k] + (1 - self.decay_rate) * g**2
                self.model[k] += self.learning_rate * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
                self.grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

        # boring book-keeping
        self.running_reward = self.reward_sum if self.running_reward is None else self.running_reward * 0.99 + self.reward_sum * 0.01
        #print(self.running_reward)
        self.plot_rewards.append(self.running_reward)

        if self.episode_number % 500 == 0:
            pickle.dump(self.model, open(self.model_file, 'wb'))
            print("weights saved", self.episode_number)

            #plt.plot(self.plot_rewards, 'b')
            pickle.dump(self.plot_rewards, open(self.reward_file, 'wb'))
            print ('running mean: %f' % (self.running_reward))

        
        if self.episode_number % 1000 == 0:

            #print ('resetting env. episode reward total was %f. running mean: %f' % (self.reward_sum, self.running_reward))
            plt.plot(self.plot_rewards, 'b')
            plt.savefig('./plots/running_rewards_' + str(self.episode_number) + '.png')

        self.reward_sum = 0

        #if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
            #print ('ep %d: game finished, reward: %f' % (self.episode_number, reward) + ('' if reward == -1 else ' !!!!!!!!'))

    def get_name(self):
        return self.name

    def load_model(self):

        try:
            with open(('../' + self.model_file), "rb") as input_file:
                self.model = pickle.load(input_file)
        except (OSError, IOError) as e:
            self.init_model()
            with open(('../' + self.model_file), "wb") as output_file:
                pickle.dump(self.model, output_file)

        try:
            with open(('../' + self.reward_file), "rb") as input_file:
                self.plot_rewards = pickle.load(input_file)
        except (OSError, IOError) as e:
            with open(('../' + self.reward_file), "wb") as output_file:
                pickle.dump(self.plot_rewards, output_file)
        
        self.grad_buffer = { k : np.zeros_like(v) for k,v in self.model.items() } # update buffers that add up gradients over a batch
        self.rmsprop_cache = { k : np.zeros_like(v) for k,v in self.model.items() } # rmsprop memory


        # another way to define a network
        self.net = torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, 200),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(200, 100),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(100, self.output_dim),
            )

        self.net.load_state_dict(torch.load('../' + self.sup_model_file))

        self.net.eval()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
        self.loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss    

    def init_model(self):
        self.model.clear()
        self.model['W1'] = np.random.randn(self.H, self.D) / np.sqrt(self.D)
        # GOOD print("1: ", self.model['W1'].shape)
        self.model['W2'] = np.random.randn(self.H) / np.sqrt(self.H)
        # GOOD print("1: ", self.model['W1'].shape)

        self.grad_buffer = { k : np.zeros_like(v) for k,v in self.model.items() } # update buffers that add up gradients over a batch
        self.rmsprop_cache = { k : np.zeros_like(v) for k,v in self.model.items() } # rmsprop memory

    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def policy_forward(self, x):
        h = np.dot(self.model['W1'], x)
        h[h < 0] = 0  # ReLU nonlinearity
        logp = np.dot(self.model['W2'], h)
        p = sigmoid(logp)

        # pickle.dump(self.model, open(self.model_file, 'wb'))
        # print("weights saved", self.episode_number)

        return p, h  # return probability of taking action 2, and hidden state

    def policy_backward(self, eph, epx, epdlogp):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.model['W2'])
        dh[eph <= 0] = 0 # backpro prelu
        dW1 = np.dot(dh.T, epx)
        return {'W1':dW1, 'W2':dW2}

    def get_action(self, observation):

        my_obs = prepro(observation)
        my_obs = np.array(my_obs)
        my_obs = torch.Tensor(my_obs)

        prediction = self.net(my_obs).detach().numpy()
        prediction = np.delete(prediction,1)

        #print(prediction)

        speeds = prediction - self.prev_x if self.prev_x is not None else np.zeros(int(self.D/2))

        #print(speeds)

        x = np.concatenate((prediction, speeds))

        #print(x)

        #print("=========")

        self.prev_x = prediction

        # forward the policy network and sample an action from the returned probability

        aprob, h = self.policy_forward(x)
        action = 1 if np.random.uniform() < aprob else 2  # roll the dice!
        
        self.xs.append(x)
        self.hs.append(h)

        y = 1 if action == 1 else 0 # a "fake label"
        self.dlogps.append(y - aprob)

        return action

    def reset(self):
        # Reset previous observation
        self.prev_x = None

    def set_environment(self, env):
        self.env = env

