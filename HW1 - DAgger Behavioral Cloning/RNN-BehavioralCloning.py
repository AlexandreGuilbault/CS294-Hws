import pickle

import torch
from torch import nn, utils

import torch.optim as optim
import torch.utils.data

import gym
import numpy as np

from run_expert_function import run_expert_with_rollouts


######################
# Settings

envname = 'Reacher-v2'
expert_data_path = r'./expert_data/'

# Behavioral model
batch_size = 64
n_tests = 42
n_epochs = 20
lr = 0.001
max_steps = 500
seq_length = 3

# Expert Policy
num_rollouts = 20
max_timesteps = 21
render = False

######################


######################
# Functions

def prepare_rolling_window_dataset(pre_X, pre_Y, seq_len):   
    # Divide observations in multiple seq_length examples
    X = []
    Y = []
    for i_rollout in range(pre_X.shape[0]):
        for i_obs in range(pre_X.shape[1]-seq_len+1) :
            # inputs.shape == [batch_size, seq_length, obs_size]
            X.append(np.array(pre_X[i_rollout, i_obs:i_obs+seq_len,:]))
            Y.append(np.array(pre_Y[i_rollout, i_obs+seq_len-1,:]))
    
    return torch.Tensor(X), torch.Tensor(Y)



print("Running Expert for {}".format(envname))
run_expert_with_rollouts(envname=envname, num_rollouts=num_rollouts, render=render, max_timesteps=max_timesteps, export_path=expert_data_path)
print("\n- Finished gathering expert examples\n")


######################
# Defining Network

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_probability):
        super(SimpleRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_probability = dropout_probability
        self.num_layers = num_layers

        if num_layers > 1:
            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout_probability, batch_first=True)
        else : # No dropout for last RNN layer
            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.ll = nn.Linear(self.hidden_size, output_size)
        
    def init_hidden(self, batch_size):
        return torch.zeros((self.num_layers, batch_size, self.hidden_size))
        
    def forward(self, X):
        
        h = self.init_hidden(X.size(0))
        X, h = self.rnn(X, h) # X ->  (batch, seq, feature)
        
        X = self.ll(X[:,-1,:].contiguous()) # Output only last RNN cell output (t=t+seq_length)

        return X

#####################
# Generating trainset
        
pickle_in = open(expert_data_path+envname+'.pkl', 'rb')
expert_data = pickle.load(pickle_in)

num_rollouts, num_examples, size_observations = expert_data['observations'].shape
num_rollouts, num_examples, size_actions = expert_data['actions'].squeeze().shape

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
new_expert_data = {}

# Observations in one sequence

X, Y = prepare_rolling_window_dataset(np.array(expert_data['observations']), np.array(expert_data['actions']).squeeze(), seq_length)

    
model = SimpleRNN(input_size=size_observations, hidden_size=size_observations*10, output_size=size_actions, num_layers=2, dropout_probability=0.5).to(device)
    
trainset = torch.utils.data.TensorDataset(X,Y)
trainloader = utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

##########################
# Train Behavioral cloning

print("Training model\n")

model.train()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(1, n_epochs+1):
    train_running_loss = 0.0
    train_acc = 0.0
    
    for i, data in enumerate(trainloader):
        optimizer.zero_grad()
        
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
    
        outputs = model(inputs)
    
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.detach().item()
         
    print('Epoch: {}/{} | Loss: {:.4}'.format(epoch, n_epochs, train_running_loss/i))

print("\n- Model trained\n")
    



print("\nTesting model\n")

model.eval()

env = gym.make(envname)

for i in range(n_tests):
    steps = 0
    obs = env.reset()

    while True:
    
        action = model(torch.Tensor(obs))
    
        obs, r, done, _ = env.step(action.detach().numpy())
        steps += 1

        env.render()
        
        if steps % 100 == 0: 
            print("[{}/{}] steps".format(steps, max_steps))
        if steps >= max_steps: break