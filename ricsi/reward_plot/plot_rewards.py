import numpy as np
import pickle
import matplotlib.pyplot as plt


with open(('./' + "running_rewards.p"), "rb") as input_file2:
    plot_rewards = pickle.load(input_file2)

plot_rewards = np.array(plot_rewards)

plot_rewards = plot_rewards[100000:]
plot_rewards = plot_rewards[0:(plot_rewards.size - plot_rewards.size % 1000)]
print(plot_rewards.shape)



thousands = np.mean(plot_rewards.reshape(-1, 1000), axis=1)
print(thousands.shape)

plt.plot(thousands)
plt.show()