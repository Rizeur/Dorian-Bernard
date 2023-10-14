import numpy as np
import torch as torch
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import gymnasium as gym

#Set up the data and parameters
torch.manual_seed(3)
#load data
# Read y
datay= np.loadtxt("y2d.dat")
yp_wall=datay[1]/2  #middle between cells
yp_first_cell = np.loadtxt('yp-instant-IDDES.dat')

# Read local vel
u_first_cells_local = np.loadtxt('u-instant-IDDES-hist.dat')
u_wall=u_first_cells_local[:,0]
# Create Data Loader for both train and validating

viscos=1/5200
ustar=(viscos*np.abs(u_wall)/yp_wall)**0.5  #calculate u_tau from a discretized derivative

# Sampling down
Sample_size = 500
Index_sampledown = np.random.choice(range(len(u_first_cells_local)),Sample_size)
#print(len(u_first_cells_local))    len(u_first_cells_local)=27999
#print(Index_sampledown)    800 random numbers between 0 and 27999 
u_sampled=u_first_cells_local[Index_sampledown,1:]  #take the velocity in the channel, not the wall (begin to 1) for the wanted index
ustar_sampled=ustar[Index_sampledown]   #take the u_tau for the wanted index
uplus=u_sampled[:,0:9] / ustar_sampled[:, np.newaxis]   #calculate u+ for every wanted value
yplus=np.tile(yp_first_cell, (len(Index_sampledown), 1))    #no clue but yplus is equivalent to y(x,t) here
yplus=yplus[:,:9]/viscos*ustar_sampled[:, np.newaxis]   #calculate y+ for every wanted value
up_flat=uplus.reshape(-1, 1)
yp_flat=yplus.reshape(-1, 1)
Lenght = len(yp_flat)

#print(yplus)
#print(uplus)
#print(yp_flat)
#print(up_flat)

#for i in range(Sample_size):
#    if i % 80 == 0 :
#        plt.scatter(yplus[i],uplus[i])
#plt.title("u+ in function of y+")
#plt.xlabel("y+")
#plt.ylabel("u+")
#plt.show()

#Split the training data adn the test data
Ratio = 8/10
Split_size = int(Lenght * Ratio)
Rest = Lenght - Split_size
uplus_test = []
yplus_test = []
uplus_train = []
yplus_train = []

for i in range(Split_size):
    uplus_train.append(up_flat[i][0])
    yplus_train.append(yp_flat[i][0])

for i in range(Split_size,Lenght):
    uplus_test.append(up_flat[i][0])
    yplus_test.append(yp_flat[i][0])


#def average(list):
#    list_sorted = sorted(list)
#    lenght = len(list_sorted)
#    list_average =  [list_sorted[0]]
#    for i in range(1,lenght):
#        list_average.append(list_sorted[i-1]/2+list_sorted[i]/2)
#    return(list_average)

#up_sa_train = average(uplus_train)
#yp_sa_train = average(yplus_train)
#up_sa_test = average(uplus_test)
#yp_sa_test = average(yplus_test)

def average_bis(list_yp,list_up, precision):
    copy_yp = list_yp
    copy_up = list_up
    lenght = len(copy_yp)
    list_with_index = []
    index = []
    new_up = []
    new_yp = sorted(copy_yp)
    for i in range(lenght):
        list_with_index.append([copy_yp[i],i])
    sorted_list = sorted(list_with_index, key=lambda x: x[0])
    for i in range(lenght):
        index.append(sorted_list[i][1])    
    for i in index:
        new_up.append(copy_up[i])

    j = precision
    i = 0
    count = 0
    loop_up, loop_yp = [], []
    total_up, total_yp = [], []
    while i < lenght and j < new_yp[-1]:
        if j-precision <= new_yp[i] <= j+precision :
            count += 1
            loop_up.append(new_up[i])
            loop_yp.append(new_yp[i])
            i += 1
        else :
            j += 2 * precision
            count = 0
            loop_up, loop_yp = [], []
        if count != 0 :
            total_up.append(loop_up)
            total_yp.append(loop_yp)
    num = len(total_up)
    average_up, average_yp = [], []
    for i in range(num):
        average_up.append(sum(total_up[i])/len(total_up[i]))
        average_yp.append(sum(total_yp[i])/len(total_yp[i]))   

    return(average_up, average_yp)
#    average = [new_yp[0],(new_yp[0]+new_yp[1])/2, (new_yp[0]+new_yp[1]+new_yp[2])/3, 
#              (new_yp[0]+new_yp[1]+new_yp[2]+new_yp[3])/4]
#    for i in range(4, lenght):
#        average.append((new_yp[i-4]+new_yp[i-3]+new_yp[i-2]+new_yp[i-1]+new_yp[i])/5)
#    return(sorted(copy_up), average)

up_sa_train, yp_sa_train = average_bis(yplus_train, uplus_train, 1/10)
up_sa_test, yp_sa_test = average_bis(yplus_test, uplus_test, 1/10)

plt.scatter(yplus_train, uplus_train, c = 'coral', label = 'average')
plt.scatter(yp_sa_train, up_sa_train, c = 'blue', label = 'real')
plt.xlabel("u+")
plt.ylabel("y+")
plt.title("real and average u+ in function of y+")
plt.show()





# Définition de l'environnement
class Environment:
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data
        self.current_index = 0

    def reset(self):
        self.current_index = 0

    def step(self):
        if self.current_index < len(self.input_data):
            observation = self.input_data[self.current_index]
            target = self.target_data[self.current_index]
            self.current_index += 1
            return observation, target
        else:
            return None, None

# Définition de l'agent
class Agent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Paramètres
input_size = 1
hidden_size = 16
output_size = 1
learning_rate = 0.001
num_epochs = 1500

# Données d'entraînement
input_data = up_sa_train
target_data = yp_sa_train

# Création de l'environnement et de l'agent
env = Environment(input_data, target_data)
agent = Agent(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

# Entraînement de l'agent
for epoch in range(num_epochs):
    total_loss = 0.0
    env.reset()
    n_loop = 0.0

    while True:
        observation, target = env.step()

        if observation is None:
            break

        observation_tensor = torch.tensor([observation], dtype=torch.float32)
        target_tensor = torch.tensor([target],  dtype=torch.float32)
        
        output = agent(observation_tensor)
        optimizer.zero_grad()
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_loop +=1

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/n_loop:.4f}')

#inputs = torch.tensor(uplus_test, dtype=torch.float32)
predictions = []
with torch.no_grad():
    for i in range(Rest):
        inputs = torch.tensor([uplus_test[i]], dtype=torch.float32)
        predictions.append(agent(inputs).numpy())
        predicted_y = predictions

plt.scatter(yplus_test, uplus_test, c = 'blue', label = 'real')
plt.scatter(predictions, uplus_test, c = 'coral', label = 'prediction')
plt.scatter(yp_sa_train, up_sa_train, c = 'green', label = 'real')
plt.xlabel("u+")
plt.ylabel("y+")
plt.title("real, average and predicted y+ in function of u+")
plt.show()