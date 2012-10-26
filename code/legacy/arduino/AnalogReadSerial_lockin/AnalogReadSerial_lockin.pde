/*
  AnalogReadSerial_lockin
 Reads an analog input on pin 0 from a lock-in amplifier, smooths and prints the result to the serial monitor 
 */

int reading   = 0;                // the readings from the analog input
int zeros     = 0;                // counter for dropped readings
int count     = 0;                // the index of the current reading
int total     = 0;                // the running total
int average   = 0;                // the average

long blocktime = 10;              // block time in ms 
long starttime = millis();        // timer

int inputPin = A0;

void setup()
{
  // initialize serial communication with computer:
  Serial.begin(115200);                   
}

void loop() 
{
  // get start time:
  starttime = millis();  
  while( millis() - starttime < blocktime ) 
  {
    // read from pin
    reading = analogRead(inputPin);
    // add to block total
    total = total + reading;    
    // augment count
    count += 1;
    // keep track of zeros
    if (reading == 0) 
    {
     zeros += 1;
    }                       
    // calculate the average:
    if (zeros < count) 
    {
        average = total / ( count - zeros );    
    } else 
    {
        average = 0;
    }
  // send it to the computer (as ASCII digits) 
  Serial.println(average, DEC);
  // reset variables
  zeros = 0;
  count = 0;
  total = 0;  
  }
}
