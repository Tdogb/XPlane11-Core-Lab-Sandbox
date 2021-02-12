#!/usr/bin/env python
import socket, time, json
global s
def Tcp_connect( HostIp, Port ):
    global s
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.connect((HostIp, Port))
    return
   
def Tcp_Write(D):
   s.send(str.encode(D + '\r'))
   return
   
# def Tcp_Read():
#     a = ' '
#     b = b''
#     while a != b'\r':
#         a = s.recv(1)
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

def Tcp_Close( ):
   s.close()
   return 
   
# Tcp_connect( '127.0.0.1', 17098)
# # Tcp_Write(b'hi')
# print(Tcp_Read())
# # Tcp_Write(b'hi')
# print(Tcp_Read())
# Tcp_Close()

timestep = 0

def calculate():
    time.sleep(1)
    return 1

def main():
    global timestep
    Tcp_connect( '127.0.0.1', 17098)
    while True:
        read = Tcp_Read()
        print(read)
        from_xplane = json.loads(read.decode('utf-8'))
        print("xp:", from_xplane["timestep"], " here: ", timestep)
        if from_xplane["timestep"] >= timestep:
            timestep = from_xplane["timestep"] + 1
            print(timestep)
            # calculate()
            names_from_mpc = {
                "timestep": timestep,
                "ele": 0,
                "ail": 0.,
                "rud": 0.,
                "throttle": 0.
            }
            Tcp_Write(json.dumps(names_from_mpc))
            print("sent")

if __name__ == "__main__":
    main()