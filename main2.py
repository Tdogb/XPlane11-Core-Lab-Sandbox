import enum
from torch._C import dtype
from torch.utils.data.dataloader import DataLoader
import xpc
import sys, threading, time, math, random
from numpy.lib.function_base import append
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.tensor import Tensor
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv

SET_NUM_DATAPOINTS = 100

dS = []
controls = []

posi_raw = []
controls_raw = []

'''
lat, lon, elev, pitch, roll, yaw, angp, angr, angy, localvx, localvy, localvz
ele, ail, rud, throttle
'''
def monitor():
    with xpc.XPlaneConnect() as client:
        posi = client.getPOSI()
        posi_temp = list(posi)
        posi_temp[2] += 500
        client.sendPOSI(posi_temp[0:len(posi_temp)-6])
        client.sendPOSI(posi_temp[0:len(posi_temp)-6], 1)
        time.sleep(0.02)
        while len(posi_raw) < SET_NUM_DATAPOINTS:
            if posi != []:
                posi_raw.append(list(client.getPOSI()))
                controls_raw.append(list(client.getCTRL()))
            time.sleep(0.02)
            
def processData():
    for i in range(1,len(posi_raw)):
        # print(posi_raw[i][len(posi_raw[0])-1])
        posi_prev_temp = posi_raw[i-1][2:12]
        posi_temp = posi_raw[i][2:12]
        control_temp = controls_raw[i][0:4]
        dS_line = []
        for b in range(0, len(posi_temp)):
            control_temp[b] = float(control_temp[b])
            dS_line.append(float(posi_temp[b]) - float(posi_prev_temp[b]))
        dS.append(dS_line)
        control_temp.extend(posi_temp)
        controls.append(control_temp)

def writecsv():
    with open('Nominal-Simulation-Training.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar="|", quoting=csv.QUOTE_MINIMAL)
        for i in range(0, len(dS)):
            temp = controls[i]
            temp.extend(dS[i])
            writer.writerow(temp)
def readcsv():
    with open('/Users/tdogb/Robotics/Core Lab/Plane Project/XPlane-Sim/Nominal-Simulation-Training-new1.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            controls.append(list(map(float,row[0:14])))
            dS.append(list(map(float,row[14:24])))

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.relu0 = nn.ReLU()
        self.l0 = nn.Linear(14,32)
        # self.l1 = nn.Linear(100,30)
        self.l2 = nn.Linear(32,10)

    def forward(self, x):
        A = self.relu0(self.l0(x))
        # B = self.relu0(self.l1(A))
        C = self.l2(A)
        return C

net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.MSELoss()

def nn():
    # print(states[59])
    # print("controls: ", controls[59])
    losses, x_test, y_test = train()
    fig, axs = plt.subplots(3)
    axs[0].plot(range(0, len(losses)), losses)
    # axs[0].set_title("Training loss")
    new = losses[100:len(losses)-1]
    axs[1].plot(range(0, len(new)), new)
    # axs[1].set_title("zoomed in training loss")
    losses_test = test(x_test, y_test)
    axs[2].plot(range(0, len(losses_test)), losses_test)
    # axs[2].set_title("Testing Loss")
    plt.show()

def train():
    net.train()
    losses = []
    x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(controls, dS, train_size=0.8)
    print(x_train_t[0])
    dataset = TensorDataset(torch.FloatTensor(x_train_t), torch.FloatTensor(y_train_t))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    for epoch in range(1,2000):
        for idx, (x,y) in enumerate(dataloader):
            x_train = Variable(x).float()
            y_train = Variable(y).float()
            optimizer.zero_grad()
            y_pred = net(x_train)
            loss = criterion(y_pred, y_train)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        if epoch % 10:
            print(loss.item())
    torch.save(net.state_dict(), "nn_output")
    return losses, x_test_t, y_test_t

def test(x_test, y_test):
    print("----------TESTING-----------")
    losses_test = []
    dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for idx, (x,y) in enumerate(dataloader):
        x_test_i = Variable(x).float()
        y_test_i = Variable(y).float()
        y_pred = net(x_test_i)
        loss = criterion(y_test_i, y_pred)
        losses_test.append(loss.item())
        # print("i:", idx, " real: ", y_test_i, " pred: ", y_pred)
        print(loss.item())
    return losses_test

if __name__ == "__main__":
    # monitor()
    # processData()
    # writecsv()
    readcsv()
    nn()