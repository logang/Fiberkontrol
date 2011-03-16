/*
  AnalogReadSerial
 Reads an analog input on pin 0, prints the result to the serial monitor 
 
 This example code is in the public domain.
 */

int sensorMin = 0;
int sensorMax = 1023;

void setup() {
  Serial.begin(115200);
}

void loop() {
  int sensorValue = analogRead(A0);
  sensorValue = map(sensorValue, sensorMin, sensorMax, 0, 255);
  Serial.println(sensorValue, DEC);
}


