import pickle

import torch
from torch import nn, utils

import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import gym

from run_expert_function import run_expert, run_expert_on_obsv


######################
# Settings

envname = 'Hopper-v2'
expert_data_path = r'./expert_data/'

# Behavioral model
batch_size = 64
n_tests = 50
n_epochs = 20
lr = 0.001

# Expert Policy
num_rollouts = 20
max_timesteps = 1000
render = False

# DAgger
n_dagger_iterations = 15 # 1 is no DAgger

######################

print("Running Expert for {}".format(envname))
run_expert(envname=envname, num_rollouts=num_rollouts, render=render, max_timesteps=max_timesteps, export_path=expert_data_path)
print("\n- Finished gathering expert examples\n")


######################
# Defining Network

class SimpleMLP(nn.Module):
    def __init__(self, n_inputs, n_outputs, dropout_probability):
        super(SimpleMLP, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.dropout_probability = dropout_probability
            
        self.l1 = nn.Linear(self.n_inputs,  self.n_inputs*2)
        self.l2 = nn.Linear(self.n_inputs*2,  self.n_inputs*4)
        self.l3 = nn.Linear(self.n_inputs*4,  self.n_inputs*8)
        self.l4 = nn.Linear(self.n_inputs*8, self.n_outputs)
        
        self.dropout = nn.Dropout(self.dropout_probability)
        
    def forward(self, X):
        
        X = F.relu(self.l1(X))
        X = self.dropout(X)
        X = F.relu(self.l2(X))
        X = self.dropout(X)
        X = F.relu(self.l3(X))
        X = self.dropout(X)     
        X = self.l4(X)
        
        return X

#####################
# Generating trainset
        
pickle_in = open(expert_data_path+envname+'.pkl', 'rb')
expert_data = pickle.load(pickle_in)

num_examples, size_observation = expert_data['observations'].shape
num_examples, size_actions = expert_data['actions'].squeeze().shape

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
new_expert_data = {}

X = torch.Tensor(expert_data['observations'])
Y = torch.Tensor(expert_data['actions'].squeeze())

models = []

for dagger_i in range(n_dagger_iterations):
    
    print("\n*******************************")
    print("* DAgger iteration {}/{}".format(dagger_i, n_dagger_iterations))
    print("*******************************\n")
    
    model = SimpleMLP(size_observation, size_actions, 0.5).to(device)
        
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
    
    
    #######################################################
    # Gather more observations from behavioral cloning model
    
    print("\nGather more observations from model\n")

    model.eval()
    env = gym.make(envname)
    
    observations = []
    
    max_steps = max_timesteps or env.spec.timestep_limit
    
    for i in range(n_tests):
        steps = 0
        obs = env.reset()

        while True:
        
            action = model(torch.Tensor(obs))
        
            obs, r, done, _ = env.step(action.detach().numpy())
            steps += 1
            observations.append(obs)
            
            if steps % 100 == 0: 
                print("[{}/{}] steps".format(steps, max_steps))
            if steps >= max_steps: break
    
    new_expert_data = run_expert_on_obsv(envname=envname, observations=observations)
    
    X = torch.cat((X, torch.Tensor(new_expert_data['observations'])))
    Y = torch.cat((Y, torch.Tensor(new_expert_data['actions']).squeeze()))
    
    print("\n -> Dataset size : {}".format(X.size()))
    models.append(model)

print("\nTesting model\n")


env = gym.make(envname)
model = models[-1]
model.eval()

max_steps = max_timesteps or env.spec.timestep_limit

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