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
Sample_size = 2000
Index_sampledown = np.random.choice(range(len(u_first_cells_local)),Sample_size)
#print(len(u_first_cells_local))    len(u_first_cells_local)=27999
#print(Index_sampledown)    Sample_size random numbers between 0 and 27999 
u_sampled=u_first_cells_local[Index_sampledown,1:]  #take the velocity in the channel, not the wall (begin to 1) for the wanted index
ustar_sampled=ustar[Index_sampledown]   #take the u_tau for the wanted index
uplus=u_sampled[:,0:9] / ustar_sampled[:, np.newaxis]   #calculate u+ for every wanted value
yplus=np.tile(yp_first_cell, (len(Index_sampledown), 1))    #no clue but yplus is equivalent to y(x,t) here
yplus=yplus[:,:9]/viscos*ustar_sampled[:, np.newaxis]   #calculate y+ for every wanted value
up_flat=uplus.reshape(-1, 1)
yp_flat=yplus.reshape(-1, 1)
Lenght = len(yp_flat)

Ratio = 5/10
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

    return(average_yp, average_up)

yp_sa_train, up_sa_train = average_bis(yplus_train, uplus_train, 1.2)
yp_sa_test, up_sa_test = average_bis(yplus_test, uplus_test, 1.2)

if len(up_sa_train) < len(up_sa_test) :
    up_sa_train.append(up_sa_train[-1])
    yp_sa_train.append(yp_sa_train[-1])
elif len(up_sa_train) > len(up_sa_test) :
    up_sa_test.append(up_sa_test[-1])
    yp_sa_test.append(yp_sa_test[-1])

lenght_test = len(up_sa_test)
lenght_train = len(up_sa_train)

def samesize(list1, list2):
    lenght1 = len(list1)
    lenght2 = len(list2)
    i = 0
    j = 0
    new_list = []
    while i < lenght2:
        if j < lenght1:
            new_list.append(list1[j])
            j += 1
        else :
            j = 0
            i -=1
        i += 1
    return(sorted(new_list))

up_test = samesize(uplus_test, up_sa_train)
yp_test = samesize(yplus_test, yp_sa_train)

plt.scatter(yplus_train, uplus_train, c = 'coral', label = 'average')
plt.scatter(yp_sa_train, up_sa_train, c = 'blue', label = 'real')
plt.xlabel("y+")
plt.ylabel("u+")
plt.title("u+ in function of y+")
plt.show()




class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

input_size = lenght_train # Dimension de x
hidden_size = 256
output_size = lenght_train # Dimension de y
learning_rate = 0.001

q_net = QNet(input_size, hidden_size, output_size)
optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

loss_list = []
num_epochs = 500
for epoch in range(num_epochs):
    optimizer.zero_grad()
    inputs = torch.tensor(up_sa_train, dtype=torch.float32)
    #inputs = inputs.view(lenght_train, 1)
    targets = torch.tensor(yp_sa_train, dtype=torch.float32)
    #targets = targets.view(lenght_train, 1)


    # Prédiction du modèle
    predictions = q_net(inputs)
    loss = criterion(predictions, targets)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    inputs_test = torch.tensor(up_sa_test, dtype=torch.float32)
    inputs_train = torch.tensor(up_sa_train, dtype=torch.float32)
    predictions_test = q_net(inputs_test)
    predicted_y_test = predictions_test.numpy()
    #inputs = inputs.view(lenght_test, 1)    
    predictions_train = q_net(inputs_train)
    predicted_y_train = predictions_train.numpy()


plt.scatter(yplus_test, uplus_test, c = 'blue', label = 'real')
plt.scatter(predicted_y_test, up_sa_test, c = 'coral', label = 'prediction')
plt.scatter(yp_sa_test, up_sa_test, c = 'green', label = 'average')

plt.xlabel("y+")
plt.ylabel("u+")
plt.title("real, average and predicted y+ in function of u+")
plt.show()
