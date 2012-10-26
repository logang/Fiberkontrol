/*
  AnalogReadSerial_lockin
 Reads an analog input on pin 0 from a lock-in amplifier, smooths and prints the result to the serial monitor 
 */

int reading   = 0;                // the readings from the analog input
int zeros     = 0;                // counter for dropped readings
int count     = 0;                // the index of the current reading
int total     = 0;                // the running total
int average   = 0;                // the average


long blocktime = 1;              // block time in ms 
long starttime = millis();        // timer


long stimTime = 10*1000;
long firePeriod = 1000;
long bwStimTime = 0*1000;
long period = stimTime + bwStimTime;
long startFireTime = millis();
long firePulseLength = 5; //for making every pulse 2 ms, and just changing the gap size bw pulses

int inputPin = A0;
int shutterPin = 13;
String msg = "000";
String prevMsg = msg;
char nextChar = ' ';

void setup()
{
  // initialize serial communication with computer:
  Serial.begin(115200);
  pinMode(shutterPin, OUTPUT);  
}

void loop() 
{
  
   prevMsg = msg;
   while(Serial.available()>0){
      nextChar = Serial.read();
      msg = msg + nextChar;
    }
    
    if(msg.length() >=4) {
      msg = msg.substring(msg.length() - 4);
    }
  
  
  
  long currentTime = millis();
  
  if(msg != prevMsg) {
     startFireTime = currentTime;
  }
  
  if (msg != "0000") {

      if (msg == "xxx") {
         digitalWrite(shutterPin, HIGH);     
      } else { 
        char d1 = msg[0] - '0';
        char d2 = msg[1] - '0';
        char d3 = msg[2] - '0';
        char d4 = msg[3] - '0';
         
        firePeriod = int(d1)*1000 + int(d2)*100 + int(d3)*10 + int(d4);
      } 

        
        
      if ( (currentTime - startFireTime)%period <= stimTime )  {
         
        
          if ( ((currentTime - startFireTime)%period)%(2*firePeriod) <= firePeriod) {
          //if ( ((currentTime - startFireTime)%period)%(2*firePeriod) <= firePulseLength) {
             digitalWrite(shutterPin, HIGH);
          } else {
             digitalWrite(shutterPin, LOW);
          }
        } else {
           digitalWrite(shutterPin, LOW);
        }
  } else {
       digitalWrite(shutterPin, LOW);  
  }

  
  
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
