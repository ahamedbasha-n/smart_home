import time
import paho.mqtt.client as mqtt
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(21,GPIO.OUT)
GPIO.setup(20,GPIO.OUT)

def on_connect(client, userdata, flags, rc): 
   print("Connected with result code " + str(rc))
   

   client.subscribe("ack") 

def on_message(client, userdata, msg):
   h = str( msg.payload)
   print(h)
   if("door open" in h):
      print("Opening Door")
      GPIO.output(21,True)
##      GPIO.output(20,False)
      time.sleep(4)
##      GPIO.output(20,True)
      GPIO.output(21,False)
##      time.sleep(2)
##      GPIO.output(20,False)
##      GPIO.output(21,False)
client = mqtt.Client() 
client.on_connect = on_connect 
client.on_message = on_message 
client.connect('broker.mqtt-dashboard.com', 1883) 

client.loop_start() 
