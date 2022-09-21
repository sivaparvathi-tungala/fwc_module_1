.include "/home/user/fwc/asm/m328Pdef.inc"

ldi r16,0b11111000  ;2 pin as output
out DDRB,r16
start:
in r17,PINB

ldi r18,0b00000001 ; value
ldi r19,0b00000010 ; value
ldi r20,0b00000100 ; value

and r18,r17
and r19,r17
lsr r19
and r20,r17
lsr r20
lsr r20

com r18
com r19
com r20

mov r21,r18
mov r22,r19
mov r23,r20

and r21,r22
and r22,r23
and r18,r23

or r21,r22
or r18,r21


lsl r18
lsl r18
lsl r18
lsl r18
lsl r18

out PORTB,r18             ;F output

rjmp start
