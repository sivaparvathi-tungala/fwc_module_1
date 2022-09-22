#include<avr/io.h>
#include<stdbool.h>
extern void init(void);
extern void start(void);

int main(void){
	DDRB = 0b00100000;
	PORTB = 0b00000111;
	Y=0;
	while(1){
		PORTB | = (Y << 5)
	A=(PINB & (0<<PINB0)) == (1<<PINB0);
	B=(PINB & (0<<PINB1)) == (1<<PINB1);
	C=(PINB & (0<<PINB2)) == (1<<PINB2);
	Y=(!A&!B) | (!A&!C) | (!B&!C);
	
		}
	
	return 0;
}
