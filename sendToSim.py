from torch._C import dtype
from torch.utils.data.dataloader import DataLoader
import xpc
import sys, threading, time, math, random, socket, json
from numpy.lib.function_base import append
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.tensor import Tensor
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib

def Tcp_connect( HostIp, Port ):
    global s
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.connect((HostIp, Port))
    s.settimeout(0.01)
    return
    
def Tcp_server_wait( numofclientwait, port ):
    global s2
    s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    s2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s2.bind(('',port))
    s2.listen(numofclientwait) 

def Tcp_server_next( ):
		global s
		s = s2.accept()[0]
   
def Tcp_Write(D):
   s.send(str.encode(D + '\r'))
   return
   
# def Tcp_Read():
#     a = ' '
#     b = b''
#     while a != b'\r':
#         a = s.recv(1)
#         if not a: break
#         if a != b'\r':
#             b = b + a
#     return b
def Tcp_Read(): 
    a = ' '
    b = b''
    a = s.recv(1)
    while a != b'\r':
        b = b + a
        a = s.recv(1)
    return b

def Tcp_Close():
   s.close()
   return 

# Tcp_Write(b'hi')
# print(Tcp_Read())
# Tcp_Write(b'hi')

SET_NUM_DATAPOINTS = 10000

posi_raw = []
controls_raw = []

names_from_xplane = {
    "timestep": 0,
    "elev": 0.,
    "pitch": 0.,
    "roll": 0.,
    "yaw": 0.,
    "angPitch": 0.,
    "angRoll": 0.,
    "angYaw": 0.,
    "localVx": 0.,
    "localVy": 0.,
    "localVz": 0.
}
names_from_mpc = {
    "timestep": 0,
    "ele": 0.,
    "ail": 0.,
    "rud": 0.,
    "throttle": 0.
}

nextTimestep = True
timestep = 0
timestepWhichHasFinishedSendingToXPlane = -1

'''
lat, lon, elev, pitch, roll, yaw, angp, angr, angy, localvx, localvy, localvz
ele, ail, rud, throttle
'''
def monitor():
    global nextTimestep,timestep,posi_raw,controls_raw,names_from_mpc,names_from_xplane,timestepWhichHasFinishedSendingToXPlane
    with xpc.XPlaneConnect() as client:
        posi = client.getPOSI()
        posi_temp = list(posi)
        posi_temp[2] += 500
        client.sendPOSI(posi_temp[0:len(posi_temp)-6])
        client.sendPOSI(posi_temp[0:len(posi_temp)-6], 1)
        time.sleep(0.02)
        while True:
            if timestep > timestepWhichHasFinishedSendingToXPlane:
                posi_temp = list(client.getPOSI())
                if posi_temp != []:
                    posi_raw = list(posi_temp)
                    controls_raw = list(client.getCTRL())
                    control_list = [names_from_mpc["ele"], names_from_mpc["ail"], names_from_mpc["rud"], names_from_mpc["throttle"]]
                    # print("control_list", control_list)
                    client.sendCTRL(control_list)
                    # timestep = timestep + 1
                    timestepWhichHasFinishedSendingToXPlane = timestep
                    # print("Run")
                    time.sleep(0.02)
offset = 2
def sendToMPC():
    global nextTimestep,timestep,posi_raw,controls_raw,names_from_mpc,names_from_xplane
    while True:
        while posi_raw == []:
            # print("posi_raw == []")
            pass
        
        names_from_xplane = {
            "timestep": timestep,
            "elev": posi_raw[offset],
            "pitch": posi_raw[offset+1],
            "roll": posi_raw[offset+2],
            "yaw": posi_raw[offset+3],
            "angPitch": posi_raw[offset+4],
            "angRoll": posi_raw[offset+5],
            "angYaw": posi_raw[offset+6],
            "localVx": posi_raw[offset+7],
            "localVy": posi_raw[offset+8],
            "localVz": posi_raw[offset+9]
        }
        print("before sending: ", timestep)
        Tcp_Write(json.dumps(names_from_xplane))
        # print("attempting to load. Timestep = ", names_from_mpc["timestep"], "curr ", timestep)
        # while names_from_mpc["timestep"] == timestep:
        print("in")
        read = Tcp_Read()
        print(read)
        if read != b'':
            print(read.decode('utf-8'))
            names_from_mpc = json.loads(read.decode('utf-8'))
            print("names", names_from_mpc["timestep"])
            timestep = timestep + 1

if __name__ == "__main__":
    Tcp_server_wait (5, 17098)
    Tcp_server_next()
    XPlane = threading.Thread(target=monitor)
    MPC = threading.Thread(target=sendToMPC)
    XPlane.start()
    MPC.start()