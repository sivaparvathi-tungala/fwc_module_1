#!/bin/bash


#Download python and latex templates

#svn co https://github.com/gadepall/training/trunk/math  /s>

#Test Latex Installation
#Uncomment only the following lines and comment the above >
cd  /sdcard/Download/line
python3 line.py
texfot pdflatex line.tex
termux-open line.pdf
#cd /sdcard/Download/math
#texfot pdflatex gvv_math_eg.tex
#termux-open gvv_math_eg.tex


#Test Python Installation
#Uncomment only the following line
#python3 /data/data/com.termux/files/home/storage/shared/t>
