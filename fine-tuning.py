#Fine-tuning Stage: use Z-direction for example

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
import matplotlib.pyplot as plt
from scipy import io

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_record = []

#hyper-parameters
LR2 = 0.005
EPOCHS_2 = 300

#normalization，make the data in the range of [0, 1]
def MinMaxScaler(data):
    for i in range(data.shape[0]):
        min = np.amin(data[i])
        max = np.amax(data[i])
        data[i] = (data[i] - min)/(max - min)
    return data

#Feature Extraction Network: MUST BE SAME WITH THE MODEL GOT FROM PRE-TRAINING STAGE
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(21, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 10)
    def forward(self,x):
        x = self.fc4(self.fc3(self.fc2(self.fc1(x))))
        x = F.normalize(x, p=2, dim=1)
        return x

#load pre-trained model
net1 = torch.load('YOUR PRE-TRAINED MODEL.pth')
net1 = net1.to(device)

#load support set
zl_support = np.array(pd.read_excel("YOUR PATH")) #8*21
zl_support = MinMaxScaler(zl_support)
zl_support = torch.tensor(torch.from_numpy(zl_support), dtype=torch.float32, requires_grad=True)
zr_support = np.array(pd.read_excel("YOUR PATH"))
zr_support = MinMaxScaler(zr_support)
zr_support = torch.tensor(torch.from_numpy(zr_support), dtype=torch.float32, requires_grad=True)

#use features of support set as initial parameter
with torch.no_grad():
    matrix_m = (net1(zl_support).detach().numpy() + net1(zr_support).detach().numpy()) / 2 #8*10

#softmax classification network + initialization
class CLASSIFIER(nn.Module):
    def __init__(self, matrix_m):
        super(CLASSIFIER,self).__init__()
        self.softmax_classifier_w = torch.nn.Parameter(torch.from_numpy(matrix_m))
        self.softmax_classifier_b = torch.nn.Parameter(torch.full([8, 1], 0.0)) #zero vector
    def forward(self,x):
        x = torch.matmul(self.softmax_classifier_w, torch.transpose(x, 0, 1))
        x += self.softmax_classifier_b
        x = x.reshape(1, 8)
        return x

net2 = CLASSIFIER(matrix_m=matrix_m)
net2 = net2.to(device)

#we find that joint fine-tuning will be better
optimizer2 = optim.SGD(itertools.chain(net1.parameters(), net2.parameters()), lr=LR2)

#make label
labels = torch.full([zl_support.shape[0]], 1, dtype=torch.long)
for i in range(zl_support.shape[0]):
    labels[i] *= i

#fine-tuning
for epoch in range(EPOCHS_2):
    running_loss = 0.0
    for i in range(zl_support.shape[0]):
        temp_input = (net1(zl_support[i].reshape([1, -1])) + net1(zr_support[i].reshape([1, -1]))) / 2
        new_input = temp_input.to(device)

        optimizer2.zero_grad()
        new_output = net2(new_input)
        loss = F.cross_entropy(new_output, labels[i].reshape(1))

        #entropy regularization
        entropy = 0
        softmaxed_output = torch.softmax(new_output, dim=1)
        for j in range(len(softmaxed_output)):
            entropy += 0 - softmaxed_output[0][j]*torch.log(softmaxed_output[0][j])
        loss += entropy

        loss.backward()
        optimizer2.step()

        running_loss += loss.item()

    loss_record.append(running_loss)

    if (epoch+1) % 10 == 0:
        print("Stage 2: EPOCH = " + str(epoch + 1))
        print("Total regular_loss = " + str(running_loss))

    if (epoch+1) == EPOCHS_2:
        print('Finished Training Stage 2')

#save model
torch.save(net1, "YOUR PATH AND MODEL.pth")
torch.save(net2, "YOUR PATH AND MODEL.pth")

#save loss
data_loss = np.array(loss_record)
io.savemat('total_regular_loss.mat', {'array': data_loss})

#Figure-loss
plt.figure(1)
plt.plot(data_loss)
plt.xlabel('Epoch')
plt.ylabel('Total regular loss')
plt.title('Stage 2')
plt.show()

#results after fine-tuning
with torch.no_grad():
    print("Result：")
    sp = np.array(pd.read_excel("YOUR PATH"))
    sp = MinMaxScaler(sp) #1*21
    sp = torch.tensor(torch.from_numpy(sp), dtype=torch.float32)
    sp = net1(sp)
    result = net2(sp)
    result = torch.softmax(result, dim=1).reshape(8).numpy()
    print("Z-direction：")
    print(result)

    #Figure-Z
    plt.figure(2)
    plt.plot(result)
    plt.xlabel('Position')
    plt.ylabel('Probability')
    plt.title('Second Pred Result: Z direction')
    plt.show()

    io.savemat('result-z after fine-tuning.mat', {'array': result}) #save result-z