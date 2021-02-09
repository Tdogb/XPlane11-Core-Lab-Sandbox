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

states = []
controls = []

SET_NUM_DATAPOINTS = 1000
previousPosi = []
def monitor():
    with xpc.XPlaneConnect() as client:
        posi = client.getPOSI()
        posi_temp = list(posi)
        posi_temp[2] += 500
        client.sendPOSI(posi_temp[0:len(posi_temp)-3])
        client.sendPOSI(posi_temp[0:len(posi_temp)-3], 1)
        time.sleep(0.02)
        posi_trash = [0,0,0,0,0,0,0]
        while len(states) < SET_NUM_DATAPOINTS:
            posi = client.getPOSI();
            ctrl = client.getCTRL();
            if(posi != []):
                posi_temp = list(posi)[2:9]
                cntrl_temp = list(ctrl)[0:4]
                if len(states) != 0:
                    cntrl_temp.extend(states[-1])
                    controls.append(cntrl_temp)
                else:
                    cntrl_temp.append(range(1,len(posi_temp)))
                    controls.append(cntrl_temp)
                states.append(posi_temp)
            #Height, airspeed, roll, pitch, yaw, aroll, apitch, ayaw
            #Throttle, roll, pitch, yaw
            # print "Loc: (%4f, %4f, %4f) Aileron:%2f Elevator:%2f Rudder:%2f\n"\
            #    % (posi[0], posi[1], posi[2], ctrl[1], ctrl[0], ctrl[2])
            time.sleep(0.02)
        states.pop(0)
        controls.pop(0)

def writeCSV():
    states_temp = states
    controls_temp = controls
    with open('Nominal-Simulation-Training.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar="|", quoting=csv.QUOTE_MINIMAL)
        for i in range(1, len(states)):
            controls_temp[i].extend(states_temp[i])
            data = controls_temp[i]
            writer.writerow(data)
def readCSV():
    with open('/Users/tdogb/Robotics/Core Lab/Plane Project/XPlane-Sim/Nominal-Simulation-Training-2.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            controls.append(row[0:11])
            states.append(row[11:19])
        for i in range(0,len(controls)):
            for b in range(0,len(controls[0])):
                controls[i][b] = float(controls[i][b])
        for i in range(0,len(states)):
            for b in range(0,len(states[0])):
                states[i][b] = float(states[i][b]) - float(controls[i][b+4])
        controls.pop(0)
        states.pop(0)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.relu0 = nn.ReLU()
        self.l0 = nn.Linear(11,100)
        self.l1 = nn.Linear(100,100)
        self.l2 = nn.Linear(100,7)

    def forward(self, x):
        A = self.relu0(self.l0(x))
        B = self.relu0(self.l1(A))
        C = self.l2(B)
        return C

net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.005)
criterion = nn.MSELoss()

def nn():
    print(states[0])
    print("controls: ", controls[0])
    losses, x_test, y_test = train()
    fig, axs = plt.subplots(4)
    axs[0].plot(range(0, len(losses)), losses)
    new = losses[100:len(losses)-1]
    axs[1].plot(range(0, len(new)), new)
    losses_test = test(x_test, y_test)
    axs[2].plot(range(0, len(losses_test)), losses_test)
    plt.show()

def train():
    net.train()
    losses = []
    x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(controls, states, train_size=0.8)
    dataset = TensorDataset(torch.Tensor(x_train_t), torch.Tensor(y_train_t))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in range(1,200):
        for idx, (x,y) in enumerate(dataloader):
            x_train = Variable(x).float()
            y_train = Variable(y).float()
            y_pred = net(x_train)
            loss = criterion(y_pred, y_train)
            losses.append(loss.item())
            optimizer.zero_grad()
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
    # writeCSV()
    readCSV()
    nn()