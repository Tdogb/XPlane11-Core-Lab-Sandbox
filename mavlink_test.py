import pymavlink
from pymavlink import mavutil
import time

# master = mavutil.mavlink_connection('/dev/cu.usbmodem2040366246531')
baud=115200
source_system=255
source_component=0
planner_format=None
write=False
append=False
robust_parsing=True
notimestamps=False
input=True,
dialect=None
autoreconnect=False
zero_time_base=False
retries=3
use_native=False,
force_connected=False
progress_callback=None
# fakePort = mavutil.MavlinkSerialPort("/dev/tty.bruh", 115200)
# master = mavutil.mavserial("/dev/tty.bruh")

master = mavutil.mavserial("/dev/cu.usbmodem01", baud=57600)
print("past")

def handle_heartbeat(msg):
	mode = mavutil.mode_string_v10(msg)
	is_armed = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
	is_enabled = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_GUIDED_ENABLED

def handle_imu(msg):
    imu_data = [msg.xacc, msg.yacc, msg.zacc, msg.xgyro, 
        msg.ygyro, msg.zgyro]
    print("imu_data", imu_data)

def handle_attitude(msg):
    attitude_data = [msg.roll, msg.pitch, msg.yaw, msg.rollspeed, 
        msg.pitchspeed, msg.yawspeed]
    print("attitude_data", attitude_data)

def read_mavlink_data():
    msg = master.recv_match(blocking=False)
    while not msg:
        msg = master.recv_match(blocking=False)
    msg_type = msg.get_type()
    if msg_type == "BAD_DATA":
        print("!!!!!!!!!!!!!!!BAD DATA!!!!!!!!!!!!!!!!!")
    elif msg_type == "HEARTBEAT":
        handle_heartbeat(msg)
    elif msg_type == "SCALED_IMU":
        handle_imu(msg)
    elif msg_type == "ATTITUDE":
        handle_attitude(msg)

def set_rc_channel_pwm(channel_id, pwm=1500):
    """ Set RC channel pwm value
    Args:
        channel_id (TYPE): Channel ID
        pwm (int, optional): Channel pwm value 1100-1900
    """
    if channel_id < 1 or channel_id > 18:
        print("Channel does not exist.")
        return

    # Mavlink 2 supports up to 18 channels:
    # https://mavlink.io/en/messages/common.html#RC_CHANNELS_OVERRIDE
    rc_channel_values = [65535 for _ in range(18)]
    rc_channel_values[channel_id - 1] = pwm
    master.mav.rc_channels_override_send(
        master.target_system,                # target_system
        master.target_component,             # target_component
        *rc_channel_values)                  # RC channel list, in microseconds.

if __name__ == "__main__":
    master.wait_heartbeat()
    print("past heartbeat")
    master.mav.request_data_stream_send(master.target_system, master.target_component, 
        mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1)

    while True:
        read_mavlink_data()
        time.sleep(0.02)
        