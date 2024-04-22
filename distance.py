import serial

# Change the serial port to /dev/serial0
ser = serial.Serial("/dev/serial0", 115200)

def getTFminiData():
    while True:
        count = ser.in_waiting
        if count > 8:
            recv = ser.read(9)
            ser.reset_input_buffer()
            if recv[0] == 89 and recv[1] == 89:  # Use decimal values instead of hex
                low = int(recv[2])
                high = int(recv[3])
                distance = low + high * 256
                print(distance)

if __name__ == '__main__':
    try:
        if not ser.is_open:
            ser.open()
        getTFminiData()
    except KeyboardInterrupt:   # Ctrl+C
        if ser is not None:
            ser.close()