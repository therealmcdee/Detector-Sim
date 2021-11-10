#include <AFMotor.h>
//dark-box is roughly 2000x2400 steps
AF_Stepper y_motor(200,1);
AF_Stepper x_motor(200,2);

int y_lim = A1;
int y_lim_val = 800;
int x_lim = A0;
int x_lim_val = 800;
int x_location = 0;
int y_location = 0;
int steps;
String (command);


void setup() {
  Serial.begin(9600);
  y_motor.setSpeed(50);
  x_motor.setSpeed(50);
  pinMode(y_lim, INPUT);
  pinMode(x_lim, INPUT);
  initialize(y_lim, y_motor);
  initialize(x_lim, x_motor);
  Serial.println("Initialization Complete");
}

void initialize (int lim, AF_Stepper motor) {
  int lim_val;
  do {
    lim_val = analogRead(lim);
    motor.step(1,FORWARD,SINGLE);
  } while (lim_val > 800);
}


void loop() { 
  command = Serial.readStringUntil("\n");
  Serial.setTimeout(5000);
  if (command != -1){
    Serial.println(command);
  }
  if (command == "pos0\n") {
    steps = 400;
    x_location = x_location + steps;
    y_location = y_location + steps;
    Serial.print(x_location);
    Serial.println(y_location);
    x_motor.step(steps,BACKWARD,SINGLE);
    y_motor.step(steps,BACKWARD,SINGLE);
    
  }
  else if (command == "pos1\n") {
    steps = 200;
    x_location = x_location - steps;
    y_location = y_location - steps;
    Serial.print(x_location);
    Serial.println(y_location);
    x_motor.step(steps,FORWARD,SINGLE);
    y_motor.step(steps,FORWARD,SINGLE);
    
  }
  else if (command == "pos2\n") {
    steps = 250;
    x_location = x_location + steps;
    y_location = y_location + steps;
    Serial.print(x_location);
    Serial.println(y_location);
    x_motor.step(steps,BACKWARD,SINGLE);
    y_motor.step(steps,BACKWARD,SINGLE);
    
  }
  else if (command == "pos3\n") {
    steps = 50;
    x_location = x_location - steps;
    y_location = y_location - steps;
    Serial.print(x_location);
    Serial.println(y_location);
    x_motor.step(steps,FORWARD, SINGLE);
    y_motor.step(steps,FORWARD, SINGLE);
   
  }
  else if (command == "rtrn\n") {
    initialize(y_lim, y_motor);
    initialize(x_lim, x_motor);
  }
}
