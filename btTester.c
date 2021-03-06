#pragma config(Hubs,  S1, HTServo,  none,     none,     none)
#pragma config(Sensor, S1,     ,               sensorI2CMuxController)
#pragma config(Sensor, S2,     touchSensor,    sensorTouch)
#pragma config(Servo,  srvo_S1_C1_1,    servo1,               tServoStandard)
#pragma config(Servo,  srvo_S1_C1_2,    servo2,               tServoNone)
#pragma config(Servo,  srvo_S1_C1_3,    servo3,               tServoNone)
#pragma config(Servo,  srvo_S1_C1_4,    servo4,               tServoNone)
#pragma config(Servo,  srvo_S1_C1_5,    servo5,               tServoNone)
#pragma config(Servo,  srvo_S1_C1_6,    servo6,               tServoStandard)
//*!!Code automatically generated by 'ROBOTC' configuration wizard               !!*//

float imWidth = 480;
float imHeight = 640;

float width = 18;
float height = 24;

float length = 19;

int downAngle = 40;
int upAngle = 0;
int defaultAngle = 130;

float encToCm = 4.1/360;

float deltaX = -3;
float deltaY = -3.2;

int udServo = servo6;
int lrServo = servo1;



task support() {
	while(true) {
		motor[motorC] = 100;
		wait1Msec(400);
		motor[motorC] = 0;
		wait1Msec(100);
	}

}

void turnOnAngle(int angle, int id) {

	int s = servo[id];

	if (s > angle) {
		for (int i = s; i > angle; i --) {
				servo[id] = i;
				wait10Msec(3);
		}
	}

	else {
		for (int i = s; i < angle; i ++) {
				servo[id] = i;
				wait10Msec(3);
		}
	}
}
task normalAnlge() {
	turnOnAngle(upAngle, udServo);
	turnOnAngle(defaultAngle, lrServo);
}

void deliver(){

	int target = -9400;

	turnOnAngle(upAngle, udServo);
	turnOnAngle(defaultAngle, lrServo);

	motor[motorA] = -60;

	while(nMotorEncoder[motorA] > target) {
		wait1Msec(5);
	}

	motor[motorA] = 0;

	turnOnAngle(downAngle - 20, udServo);

	stopTask(support);

	motor[motorC] = -100;

	wait10Msec(100);

	turnOnAngle(upAngle, udServo);

	motor[motorC] = 0;
}

void pick() {
	motor[motorC] = -100;
	wait1Msec(1000);

	turnOnAngle(downAngle, udServo);

	motor[motorC] = 100;

	wait1Msec(1000);

	startTask(support);

	wait1Msec(2000);

	turnOnAngle(upAngle, udServo);
}


int getRotAngle(float dy) {

	float sinA = dy/length;
	return (int)(256*acos(sinA)/PI);

}

void nullifyDelta(float dx, float dy) {
	float r = sqrt(length*length - dy*dy);

	nxtDisplayCenteredTextLine(2, "%d", dx);
	if (r > dx) {
		float ddelta = r - dx;
		motor[motorA] = 20;
		int st = nMotorEncoder[motorA];
		while(nMotorEncoder[motorA] - st < (int)(ddelta/encToCm)) wait1Msec(5);


	}
	else if (r < dx) {
		float ddelta = dx - r;
		motor[motorA] = -20;
		int st = nMotorEncoder[motorA];

		while(st - nMotorEncoder[motorA] < (int)(ddelta/encToCm)) wait1Msec(5);

	}

	motor[motorA] = 0;
}

void collect(int x, int y, int pos) {
	turnOnAngle(defaultAngle, lrServo);
	int delta = nMotorEncoder[motorA];
	delta-=pos;

	displayCenteredTextLine(0, "%d", pos);
  displayCenteredTextLine(2, "%d", nMotorEncoder[motorA]);

	if (delta > 0) {
		motor[motorA] = -80;
		while(nMotorEncoder[motorA] > pos)
		{
			displayCenteredTextLine(0, "%d", pos);
  		displayCenteredTextLine(2, "%d", nMotorEncoder[motorA]);
			wait1Msec(3);
		}
	} else {
		motor[motorA] = 80;
		while(nMotorEncoder[motorA] < pos)
		{
			wait1Msec(3);
		}
	}

	motor[motorA] = 0;

	float dx = (imWidth - x)/imWidth*width + deltaX;
	float dy = y/imHeight*height + deltaY;

	nullifyDelta(dx, dy);

	int angle = getRotAngle(dy);

	turnOnAngle(angle, lrServo);


	wait10Msec(200);

	pick();



	deliver();

	playSound(soundBlip);
	sendMessageWithParm(100, 1, 1);

}

void waitForCmd() {

		long x, y, z;
  	string displayx, displayy, displayz;

		x = messageParm[0];
    y = messageParm[1];
    z = messageParm[2];

    //Formats the variables into a 'Value x: ' format and displays each on a seperate line
    stringFormat(displayx, "Value x: %d", x);
    stringFormat(displayy, "Value y: %d", y);
    stringFormat(displayz, "Value z: %d", z);

    displayCenteredTextLine(0, displayx);
    displayCenteredTextLine(2, displayy);
    displayCenteredTextLine(4, displayz);

		while(x == 0) {

			x = messageParm[0];
	    y = messageParm[1];
	    z = messageParm[2];
	    if (x > 0) playSound(soundBeepBeep);

	    //Formats the variables into a 'Value x: ' format and displays each on a seperate line
	    stringFormat(displayx, "Value x: %d", x);
	    stringFormat(displayy, "Value y: %d", y);
	    stringFormat(displayz, "Value z: %d", z);

	    displayCenteredTextLine(0, displayx);
	    displayCenteredTextLine(2, displayy);
	    displayCenteredTextLine(4, displayz);

	    wait10Msec(1);
	    ClearMessage();

		}

		collect(x,y,-z);
		ClearMessage();
		ClearMessage();
		waitForCmd();

}

task main()
{

startTask(normalAnlge);

	while(SensorValue(touchSensor) == 0)
	{
		motor[motorA] = 80;
	}

	motor[motorA] = 0;

	nMotorEncoder[motorA] = 0;
  long x, y, z;
  string displayx, displayy, displayz;

while(y < 2)
  {
  	//Receive variable values x, y, and z. Note that the messageParm array starts at 0 and not 1
    x = messageParm[0];
    y = messageParm[1];
    z = messageParm[2];

    //Formats the variables into a 'Value x: ' format and displays each on a seperate line
    stringFormat(displayx, "Value x: %d", x);
    stringFormat(displayy, "Value y: %d", y);
    stringFormat(displayz, "Value z: %d", z);

    displayCenteredTextLine(0, displayx);
    displayCenteredTextLine(2, displayy);
    displayCenteredTextLine(4, displayz);

    if (y==1) sendMessageWithParm(313, 1, 1);

    //Waits 300 milliseconds, clears the screen, and loops again
    wait1Msec(100);
    ClearMessage();
  }

  playSound(soundBeepBeep);

  ClearMessage();
  nMotorEncoder[motorA] = 0;
  motor[motorA] = -40;
  int a = -nMotorEncoder[motorA];
  turnOnAngle(upAngle, udServo);
	turnOnAngle(defaultAngle, lrServo);
  bool tr = true;
  while(tr)
  {
  	//Receive variable values x, y, and z. Note that the messageParm array starts at 0 and not 1
    x = messageParm[0];
    y = messageParm[1];
    z = messageParm[2];


		a = -nMotorEncoder[motorA];
    sendMessageWithParm(a, x, 1);
    if (a > 8900) {
    	motor[motorA] = 0;
    	tr = false;
    	}

    //Formats the variables into a 'Value x: ' format and displays each on a seperate line
    stringFormat(displayx, "Value x: %d", a);
    stringFormat(displayy, "Value y: %d", y);
    stringFormat(displayz, "Value z: %d", z);

    displayCenteredTextLine(0, displayx);
    displayCenteredTextLine(2, displayy);
    displayCenteredTextLine(4, displayz);

    //Waits 300 milliseconds, clears the screen, and loops again
    wait1Msec(10);
    ClearMessage();
  }

  waitForCmd();
}
