import socket, time, json

'''
state_from_socket = {
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
'''
def send_control(state_from_socket):
    # elevator, aileron, rudder, throttle
    return [0,0,0,0]



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

timestep = 0

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
            cntrl = send_control(from_xplane)
            names_from_mpc = {
                "timestep": timestep,
                "ele": cntrl[0],
                "ail": cntrl[1],
                "rud": cntrl[2],
                "throttle": cntrl[3]
            }
            Tcp_Write(json.dumps(names_from_mpc))
            print("sent")

if __name__ == "__main__":
    main()