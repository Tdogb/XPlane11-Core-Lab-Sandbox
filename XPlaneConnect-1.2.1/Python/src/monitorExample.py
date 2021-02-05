import sys
import threading
import time
import xpc
from numpy.lib.function_base import append
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import math,random
from torch.tensor import Tensor

states = [[]]
controls = [[]]

def monitor():
    with xpc.XPlaneConnect() as client:
        while True:
            posi = client.getPOSI();
            ctrl = client.getCTRL();
            states.append(list(posi))
            controls.append(list(ctrl))
            print(states)
            #Height, airspeed, roll, pitch, yaw, aroll, apitch, ayaw
            #Throttle, roll, pitch, yaw
            # print "Loc: (%4f, %4f, %4f) Aileron:%2f Elevator:%2f Rudder:%2f\n"\
            #    % (posi[0], posi[1], posi[2], ctrl[1], ctrl[0], ctrl[2])
            time.sleep(0.01)

def nn():
    model = nn.Sequential(
    nn.Linear(4,100),
    nn.ReLU(),
    nn.Linear(100,100),
    nn.ReLU(),
    nn.Linear(100,8)
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
)
    pass
    # while True:
    #     print(states)

if __name__ == "__main__":
    monitor_thread = threading.Thread(target=monitor)
    nn_thread = threading.Thread(target=nn)
    monitor_thread.start()
    nn_thread.start()