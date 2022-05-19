#import RPi.GPIO as GPIO
from subprocess import call
import time
import os
import glob
import smtplib
import base64
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.MIMEText import MIMEText
import subprocess
import sys

gmail_user = "dineshclash5418@gmail.com"
gmail_pwd = "dineshraj11"
FROM = 'dineshclash5418@gmail.com'
TO = ['gamesahamed354@gmail.com'] #must be a list

#IMAGE

##subprocess.Popen("sudo fswebcam input.jpg",shell=True).communicate()      
##time.sleep(1)

msg = MIMEMultipart()
time.sleep(1)
msg['Subject'] ="Test message"

#BODY with 2 argument

body="helloo"          
msg.attach(MIMEText(body,'plain'))
time.sleep(1)


#IMAGE
fp = open("/home/pi/main/input.jpg", 'rb')   		
time.sleep(1)
img = MIMEImage(fp.read())
time.sleep(1)
fp.close()
time.sleep(1)
msg.attach(img)
time.sleep(1)

try:
        server = smtplib.SMTP("smtp.gmail.com", 587) #or port 465 doesn't seem to work!
        print ("smtp.gmail")
        server.ehlo()
        print "ehlo"
        server.starttls()
        print "starttls"
        server.login(gmail_user, gmail_pwd)
        print "reading mail & password"
        server.sendmail(FROM, TO, msg.as_string())
        print "from"
        server.close()
        print 'successfully sent the mail'
except:
        print "failed to send mail"


