#include<Arduino.h>
int A,B,C;
int Y;
void setup(){
	pinMode(2,INPUT);
	pinMode(3,INPUT);
	pinMode(4,INPUT);
	pinMode(5,OUTPUT);
}
void loop(){
	A=digitalRead(2);
	B=digitalRead(3);
	C=digitalRead(4);
	Y=(!A&&!B)||(!A&&!C)||(!B&&!C);
	digitalWrite(5,Y);
}

