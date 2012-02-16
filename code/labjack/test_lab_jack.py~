import u6
from time import sleep

d = u6.U6()

#FI02_STATE_REGISTER = 6002
#d.writeRegister(FI02_STATE_REGISTER, 1)

d.writeRegister(50501, 1) #enable one timer

#if you wish to move the output...
d.writeRegister(50500, 2) #to change the offset to 2 pins

d.writeRegister(7100, 0) #for 16bit PWM output
d.writeRegister(7000, 4) #for 4 MHz/divisor base clock
d.writeRegister(7200, 65535) #set the duty cycle to about 0%

TIME_CLOCK_BASE = 4000000 # in Hz
FREQUENCY_CONVERSION_CONSTANT = 65536


freqs = [0.5, 1, 2, 3, 5, 10, 20, 30]

for f in freqs:
    desired_freq = 1 #in Hz
    divisor = int(TIME_CLOCK_BASE/(FREQUENCY_CONVERSION_CONSTANT*desired_freq))
    d.writeRegister(7002, divisor) #set the divisor
    d.writeRegister(7200, 32768) #set the duty cycle to about 50%
    sleep(4)



#d.writeRegister(7200, 65535) #set the duty cycle to about 0%
