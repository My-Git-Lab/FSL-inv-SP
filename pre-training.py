#Pre-training Stage

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy import io

num_class = 9 #class of training dataset
num_dianji = 21 #electrodes on the surface
total_sample_size = 1800 #pairs of selected training data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_record = []
accuracy_record = []

#hyper-parameters
batch_size = 10
LR1 = 0.002
EPOCHS_1 = 50

#normalization，make the data in the range of [0, 1]
def MinMaxScaler(data):
    for i in range(data.shape[0]):
        min = np.amin(data[i])
        max = np.amax(data[i])
        data[i] = (data[i] - min)/(max - min)
    return data

#SP data: one leak position one class
pos_1 = np.array(pd.read_excel("YOUR PATH")) #245*21 samples * num of electrodes
data_x1 = MinMaxScaler(pos_1)
pos_2 = np.array(pd.read_excel("YOUR PATH"))
data_x2 = MinMaxScaler(pos_2)
pos_3 = np.array(pd.read_excel("YOUR PATH"))
data_x3 = MinMaxScaler(pos_3)
pos_4 = np.array(pd.read_excel("YOUR PATH"))
data_x4 = MinMaxScaler(pos_4)
pos_5 = np.array(pd.read_excel("YOUR PATH"))
data_x5 = MinMaxScaler(pos_5)
pos_6 = np.array(pd.read_excel("YOUR PATH"))
data_x6 = MinMaxScaler(pos_6)
pos_7 = np.array(pd.read_excel("YOUR PATH"))
data_x7 = MinMaxScaler(pos_7)
pos_8 = np.array(pd.read_excel("YOUR PATH"))
data_x8 = MinMaxScaler(pos_8)
pos_9 = np.array(pd.read_excel("YOUR PATH"))
data_x9 = MinMaxScaler(pos_9)

#sampler
def get_data(total_sample_size):
    count = 0
    x_pair = np.zeros([total_sample_size, 3, num_dianji])

    for i in range(total_sample_size): #random select
        anchor_class = np.random.randint(num_class)
        while True:
            neg_class = np.random.randint(num_class)
            if anchor_class != neg_class:
                o1 = np.random.randint(200) #use the first 200 samples for training
                o2 = np.random.randint(200)
                o3 = np.random.randint(200)
                if o1 != o2:
                    break
        anchor = globals()['data_x%d' % (anchor_class + 1)][o1]
        data_pos = globals()['data_x%d' % (anchor_class + 1)][o2]
        data_neg = globals()['data_x%d' % (neg_class + 1)][o3]

        x_pair[count, 0, :] = anchor
        x_pair[count, 1, :] = data_pos
        x_pair[count, 2, :] = data_neg

        count += 1

    return x_pair

#training data
X = get_data(total_sample_size) #1800*3*21
new_x = X.transpose((1, 0, 2)) #3*1800*21
x0 = torch.tensor(torch.from_numpy(new_x[0]), dtype=torch.float32, requires_grad=True) #1800*21
x1 = torch.tensor(torch.from_numpy(new_x[1]), dtype=torch.float32, requires_grad=True)
x2 = torch.tensor(torch.from_numpy(new_x[2]), dtype=torch.float32, requires_grad=True)

#use the last 9*45 samples for test
test_class_1 = torch.tensor(torch.from_numpy(data_x1[200:,:]), dtype=torch.float32) #45*21
test_class_2 = torch.tensor(torch.from_numpy(data_x2[200:,:]), dtype=torch.float32)
test_class_3 = torch.tensor(torch.from_numpy(data_x3[200:,:]), dtype=torch.float32)
test_class_4 = torch.tensor(torch.from_numpy(data_x4[200:,:]), dtype=torch.float32)
test_class_5 = torch.tensor(torch.from_numpy(data_x5[200:,:]), dtype=torch.float32)
test_class_6 = torch.tensor(torch.from_numpy(data_x6[200:,:]), dtype=torch.float32)
test_class_7 = torch.tensor(torch.from_numpy(data_x7[200:,:]), dtype=torch.float32)
test_class_8 = torch.tensor(torch.from_numpy(data_x8[200:,:]), dtype=torch.float32)
test_class_9 = torch.tensor(torch.from_numpy(data_x9[200:,:]), dtype=torch.float32)

#Feature Extraction Network: YOU CAN CHANGE
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(21, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 10)
    def forward(self,x):
        x = self.fc4(self.fc3(self.fc2(self.fc1(x))))
        x = F.normalize(x, p=2, dim=1) #get unit vector
        return x

net1 = MLP()
net1 = net1.to(device)
optimizer = optim.Adam(net1.parameters(), lr=LR1)

#pre-training
for epoch in range(EPOCHS_1):
    print("Stage 1: EPOCH = " + str(epoch+1))
    running_loss = 0.0
    num_of_batch = int(total_sample_size / batch_size)
    for i in range(num_of_batch):
        input1 = x0[i*batch_size:(i+1)*batch_size]
        input2 = x1[i*batch_size:(i+1)*batch_size]
        input3 = x2[i*batch_size:(i+1)*batch_size]
        input1, input2, input3 = input1.to(device), input2.to(device), input3.to(device) #batchsize*21

        optimizer.zero_grad()

        output1 = net1(input1)
        output2 = net1(input2)
        output3 = net1(input3) #batchsize * 10(dimension of unit vector)

        loss = F.triplet_margin_loss(output1, output2, output3) #return mean_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size #total loss of each batch

    #if epoch % 100 == 0:
    print("Total loss = " + str(running_loss))
    loss_record.append(running_loss)

    if (epoch+1) == EPOCHS_1:
        print('Finished Training Stage 1')

#get accuracy after training
with torch.no_grad():
    correct_count = 0
    matrix_f = net1(test_class_1[0].reshape(1,21)).numpy() #1*10
    matrix_f = np.append(matrix_f, net1(test_class_2[0].reshape(1,21)).numpy(), axis = 0)
    matrix_f = np.append(matrix_f, net1(test_class_3[0].reshape(1,21)).numpy(), axis = 0)
    matrix_f = np.append(matrix_f, net1(test_class_4[0].reshape(1,21)).numpy(), axis = 0)
    matrix_f = np.append(matrix_f, net1(test_class_5[0].reshape(1,21)).numpy(), axis = 0)
    matrix_f = np.append(matrix_f, net1(test_class_6[0].reshape(1,21)).numpy(), axis = 0)
    matrix_f = np.append(matrix_f, net1(test_class_7[0].reshape(1,21)).numpy(), axis = 0)
    matrix_f = np.append(matrix_f, net1(test_class_8[0].reshape(1,21)).numpy(), axis = 0)
    matrix_f = np.append(matrix_f, net1(test_class_9[0].reshape(1,21)).numpy(), axis = 0) #9*10
    for i in range(num_class):
        for j in range(1, 45):
            temp_sample_feature = net1(globals()['test_class_%d' % (i + 1)][j].reshape(1,21)).numpy()
            matrix_class = np.matmul(matrix_f, temp_sample_feature.transpose()).reshape(9)
            if np.argmax(matrix_class) == i:
                correct_count += 1
    accuracy_record.append(correct_count / 396) #396 = 9 * 44

#save feature extraction network
torch.save(net1, "YOUR PATH AND MODEL.pth")

#save loss & accuracy record
data_loss = np.array(loss_record)
data_accuracy = np.array(accuracy_record)
io.savemat('total_loss.mat', {'array': data_loss})
io.savemat('test_accuracy.mat', {'array': data_accuracy})

#Figure-loss
plt.figure(1)
plt.plot(data_loss)
plt.xlabel('Epoch')
plt.ylabel('Total loss')
plt.title('Stage 1')
plt.show()

#support set
x_support = np.array(pd.read_excel("YOUR PATH")) #21*21 electrodes on surface * electrodes on surface
x_support = MinMaxScaler(x_support)
zl_support = np.array(pd.read_excel("YOUR PATH")) #8*21 electrodes in well * electrodes on surface
zl_support = MinMaxScaler(zl_support)
zr_support = np.array(pd.read_excel("YOUR PATH")) #8*21
zr_support = MinMaxScaler(zr_support)

#query
sp = np.array(pd.read_excel("YOUR PATH")) #1*21
sp = MinMaxScaler(sp)

#result after pre-training
with torch.no_grad():
    print("Result:")
    sp = torch.tensor(torch.from_numpy(sp), dtype=torch.float32) #1*21
    sp = net1(sp) #1*10
    sp = torch.transpose(sp, 0, 1) #10*1

    #X-direction
    x_support = torch.tensor(torch.from_numpy(x_support), dtype=torch.float32) #21*21
    x_temp = net1(x_support) #21*10
    x_temp = torch.matmul(x_temp, sp)  #21*10 * 10*1 = 21*1
    result_x = torch.softmax(torch.transpose(x_temp, 0, 1), dim=1)
    result_x = result_x.reshape(21).numpy()
    print("Probability distribution of X-direction：")
    print(result_x)

    #Figure-X
    plt.figure(2)
    plt.plot(result_x)
    plt.xlabel('Position')
    plt.ylabel('Probability')
    plt.title('First Pred Result: X direction')
    plt.show()

    #Z-direction
    zl_support = torch.tensor(torch.from_numpy(zl_support), dtype=torch.float32) #8*21
    zl_temp = net1(zl_support) #8*10
    zr_support = torch.tensor(torch.from_numpy(zr_support), dtype=torch.float32)
    zr_temp = net1(zr_support)
    z_temp = (zl_temp + zr_temp) / 2 #average
    z_temp = torch.matmul(z_temp, sp)
    result_z = torch.softmax(torch.transpose(z_temp, 0, 1), dim=1)
    result_z = result_z.reshape(8).numpy()
    print("Probability distribution of Z-direction：")
    print(result_z)

    #Figure-Z
    plt.figure(3)
    plt.plot(result_z)
    plt.xlabel('Position')
    plt.ylabel('Probability')
    plt.title('First Pred Result: Z direction')
    plt.show()

    #save result
    io.savemat('result-x.mat', {'array': result_x})
    io.savemat('result-z.mat', {'array': result_z})