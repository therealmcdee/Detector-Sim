import time 
from serial import Serial 
import sys

arduino = Serial(port = '/dev/ttyACM0', baudrate=9600, timeout = 0.5)

x_loc = []
y_loc = []

def update_loc(line):
    print(line)
    x_loc.append(line[0:2])
    y_loc.append(line[3:5])


time.sleep(5)

for i in range(4):
    arduino.write('pos{}\n'.format(i))
    time.sleep(1)
    if (arduino.readline()!='')==True:
        inpt = arduino.readline()
        update_loc(inpt)
    time.sleep(10)

#arduino.write("pos \n")
#time.sleep(10)

#arduino.write("pos2\n")
#time.sleep(10)

#arduino.write("pos3\n")
#time.sleep(10)

#arduino.write("pos4\n")
#time.sleep(10)

arduino.write("rtrn\n")

#update_loc()
arduino.close()
#print(x_loc)
#print(y_loc)

#new = open("prev_run.txt",'w')
#for i in len(x_loc):
    #new.write('{}, {}'.format(x_loc[i],y_loc[i]))
#new.close()
