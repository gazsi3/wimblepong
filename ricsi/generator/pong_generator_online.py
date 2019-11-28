import wimblepong
import gym
from wimblepong.simple_ai import SimpleAi
import matplotlib.pyplot as plt
import numpy as np
import pickle

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data



class PongTrainbench(object):
    def __init__(self, render=False):
        self.player1 = None
        self.player2 = None
        self.total_games = 0
        self.wins1 = 0
        self.wins2 = 0
        self.render = render
        self.env = gym.make("WimblepongVisualMultiplayer-v0")
        self.me = 0
        self.sample_count = 0

    def init_players(self, player1, player2=None):
        if player1:
            self.player1 = player1
        else:
            self.player1 = SimpleAi(self.env, player_id=1)
        if player2:
            self.player2 = player2
        else:
            self.player2 = SimpleAi(self.env, player_id=2)
        self.set_names()

    def switch_sides(self):
        def switch_simple_ai(player):
            if type(player) is SimpleAi:
                player.player_id = 3 - player.player_id

        op1 = self.player1
        ow1 = self.wins1
        self.player1 = self.player2
        self.wins1 = self.wins2
        self.player2 = op1
        self.wins2 = ow1

        self.me += 1

        # Ensure SimpleAi knows where it's playing
        switch_simple_ai(self.player1)
        switch_simple_ai(self.player2)

        self.env.switch_sides()
        print("Switching sides.")

    def prepro(self,I):
        """ prepro 200x200x3 uint8 frame into 10000 (10x10) 1D float vector """
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 43] = 0 # erase background (background type 1)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        return I.astype(np.float).ravel()

    def play_game(self):
        self.player1.reset()
        self.player2.reset()
        obs1, obs2 = self.env.reset()
        done = False
        samples = []
        while not done:
            action1 = self.player1.get_action(obs1)
            action2 = self.player2.get_action(obs2)


            if(self.me%2==0):
                player = self.env.player1
                opponent = self.env.player2
            else:
                player = self.env.player2
                opponent = self.env.player1
            
            my_y = player.y
            op_y = opponent.y
            ball_x = self.env.ball.x
            ball_y = self.env.ball.y

            #print(my_y, op_y, ball_x, ball_y)

            #plt.imshow(obs1)
            #plt.show()

            my_obs = self.prepro(obs1)
            my_obs = np.array(my_obs)
            #print(my_obs.shape)

            sample = [my_obs, my_y, op_y, ball_x, ball_y]
            #print(sample)


            samples.append(sample)

            self.sample_count += 1

            

            (obs1, obs2), (rew1, rew2), done, info = self.env.step((action1, action2))

            if self.render:
                self.env.render()

            if done:

                if rew1 > 0:
                    self.wins1 += 1
                elif rew2 > 0:
                    self.wins2 += 1
                else:
                    raise ValueError("Game finished but no one won?")
                self.total_games += 1
                # print("Game %d finished." % self.total_games)

        return samples

    def run_test(self, no_games=100, switch_freq=-1):
        # Ensure the testbench is in clear state
        assert self.wins1 is 0 and self.wins2 is 0 and self.total_games is 0

        if switch_freq == -1:
            # Switch once in the middle
            switch_freq = no_games // 2
        elif switch_freq in (None, 0):
            # Don't switch sides at all
            switch_freq = no_games*2

        print("Running test: %s vs %s." % (self.player1.get_name(), self.player2.get_name()))

        input_dim = 10000
        output_dim = 4
        #print(input_dim, output_dim)

        # another way to define a network
        net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 200),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(200, 100),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(100, output_dim),
            )

        model_file = "./supervised_model.pth"
        loss_file = "./losses.p"

        try:
            net.load_state_dict(torch.load(model_file))
        #model.eval()
        except (OSError, IOError) as e:
            pass

        try:
            with open((loss_file), "rb") as input_file:
                plot_losses = pickle.load(input_file)

        except (OSError, IOError) as e:

            plot_losses = []
            with open((loss_file), "wb") as output_file:
                pickle.dump(plot_losses, output_file)

        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss        

        games = 0
        running_loss = 0
        print_freq = 1

        while games <= (no_games):

            samples = self.play_game()
            #print("samples ",self.sample_count)

            samples = np.array(samples)

            x = samples[:,0]
            x = x.reshape(x.shape[0],-1)
            #print(x.shape)
            y = samples[:,1:]
            y = y.reshape(x.shape[0],-1)
            #print(y.shape)

            new_x = np.zeros((x.shape[0], (x[0][0]).shape[0]))
            new_y = np.zeros((y.shape[0], (y).shape[1]))

            #print(y)

            for i in range(x.shape[0]):
                new_x[i,:] = x[i][0]
                new_y[i,:] = y[i]

            x = torch.Tensor(new_x)
            y = torch.Tensor(new_y)

            prediction = net(x)
            loss = loss_func(prediction, y)
            running_loss += loss

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients   

            if(games % print_freq == 0 and games != 0):

                print("game", games, " | avg loss in last 100 games: ", (running_loss/print_freq))
                plot_losses.append(running_loss)
                running_loss = 0
                torch.save(net.state_dict(), model_file)


                plt.plot(plot_losses, 'b')
                plt.savefig('./losses.png')

                with open((loss_file), "wb") as output_file:
                    pickle.dump(plot_losses, output_file)

            games += 1  

        torch.save(net.state_dict(), model_file)

        with open((loss_file), "wb") as output_file:
                pickle.dump(plot_losses, output_file)

        plt.plot(plot_losses, 'b')
        plt.savefig('./losses.png')

        plt.show()

        plt.plot(plot_losses[10:], 'b')

        plt.show()

    def set_names(self):
        def verify_name(name):
            # TODO: some ASCII/profanity checks?
            return type(name) is str and 0 < len(name) <= 26

        name1 = self.player1.get_name()
        name2 = self.player2.get_name()

        if not verify_name(name1):
            raise ValueError("Name", name1, "not correct")
        if not verify_name(name2):
            raise ValueError("Name", name2, "not correct")

        self.env.set_names(name1, name2)

    def get_agent_score(self, agent):
        if agent is self.player1:
            return self.wins1, self.total_games
        elif agent is self.player2:
            return self.wins2, self.total_games
        else:
            raise ValueError("Agent not found in the testbench!")


