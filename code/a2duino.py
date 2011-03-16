#! /usr/bin/env python                                                                          
import pyduino
import sys
import time

def get_anolog(pin=5):
    arduino = pyduino.Arduino('/dev/tty.usbserial-A600eu7L')
    arduino.analog[pin].set_active(5)
    last_value = 0
    while 1:
        try:
            arduino.iterate()
            value = arduino.analog[pin].read()
#            if value != last_value:
            last_value = value
            print "Analog pin value is %f"% value
        except KeyboardInterrupt:
            arduino.exit()
            break


