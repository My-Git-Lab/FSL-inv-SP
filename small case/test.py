# A TEST CASE USE X-direction FOR EXAMPLE

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

normalized_data = np.array(pd.read_excel("normalized data.xlsx")) #print(normalized_data.shape) 22*22
Query_SP = np.array(normalized_data[0, 1:], dtype=np.float32).reshape(1, 21) #print(Query_SP) 1*21
Support_X_direction = np.array(normalized_data[1:, 1:], dtype=np.float32) #print(Support_X_direction.shape) 21*21

#Feature Extraction Network
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
net = torch.load('pre-trained model.pth')
net = net.to(device)

#results in X-direction
with torch.no_grad():
    print("Result:")
    sp = torch.tensor(torch.from_numpy(Query_SP), dtype=torch.float32) #1*21
    sp = net(sp) #1*10
    sp = torch.transpose(sp, 0, 1) #10*1

    #X-direction
    x_support = torch.tensor(torch.from_numpy(Support_X_direction), dtype=torch.float32) #21*21
    x_temp = net(x_support) #21*10
    x_temp = torch.matmul(x_temp, sp)  #21*10 * 10*1 = 21*1
    result_x = torch.softmax(torch.transpose(x_temp, 0, 1), dim=1)
    result_x = result_x.reshape(21).numpy()
    print("Probability distribution of X-direction：")
    print(result_x)

    #Figure-X
    plt.figure(1)
    plt.plot(result_x)
    plt.xlabel('Position')
    plt.ylabel('Probability')
    plt.title('First Pred Result: X direction')
    plt.show()