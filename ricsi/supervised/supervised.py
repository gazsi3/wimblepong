import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt

import numpy as np
import pickle

torch.manual_seed(1)    # reproducible

def load_samples(file):
    with open(('../generator/' + file), "rb") as input_file:
        samples = pickle.load(input_file)
    samples = np.array(samples)
    #print(samples.shape)
    #print(samples)

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

    print(new_x.shape)
    print(new_y.shape)

    x = torch.Tensor(new_x)
    y = torch.Tensor(new_y)


    # torch can only train on Variable, so convert them to Variable
    x, y = Variable(x), Variable(y)
    return x, y

x, y = load_samples("train_samples.p")
x_test, y_test = load_samples("test_samples.p")

input_dim = x.shape[1]
output_dim = y.shape[1]
print(input_dim, output_dim)

model_file = "./supervised_model.pth"

# another way to define a network
net = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 200),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(200, 100),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100, output_dim),
    )

net.load_state_dict(torch.load(model_file))
#model.eval()

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

BATCH_SIZE = 100
EPOCH = 1

torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, num_workers=2,)

train_losses, test_losses = [],[]

# start training
for epoch in range(EPOCH):
    epoch_train_loss = 0
    i = 0
    for step, (batch_x, batch_y) in enumerate(loader): # for each training step
        
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        prediction = net(b_x)     # input x and predict based on x

        loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)

        epoch_train_loss += loss

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        i += 1

    print("epoch: " + str(epoch+1) + "/" + str(EPOCH))
    epoch_train_loss = epoch_train_loss/i
    train_losses.append(epoch_train_loss)

    test_prediction = net(x_test)
    test_loss = loss_func(test_prediction, y_test)
    test_losses.append(test_loss)

    print("avg train loss: " + str(epoch_train_loss))
    print("avg test loss: " + str(test_loss))

plt.plot(train_losses, label = 'train losses')
plt.plot(test_losses, label = 'test losses')
plt.legend(loc='best')
plt.show()

torch.save(net.state_dict(), model_file)

for i in range(5):
    print(y_test[i,:])
    print(net(x_test[i,:]))
    print("===========")