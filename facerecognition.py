# USAGE
# python facerecognition.py --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle

from imutils.video import FPS, VideoStream
from datetime import datetime
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import json
import sys
import signal
import os
import numpy as np
import subprocess
import aws_s3
from datetime import datetime


haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'  #All the faces data will be present this folder

face_cascade = cv2.CascadeClassifier(haar_file)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def printjson(type, message):
	print(json.dumps({type: message}))
	sys.stdout.flush()

def signalHandler(signal, frame):
	global closeSafe
	closeSafe = True

signal.signal(signal.SIGINT, signalHandler)
closeSafe = False

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", type=str, required=False, default="haarcascade_frontalface_default.xml",
	help = "path to where the face cascade resides")
ap.add_argument("-e", "--encodings", type=str, required=False, default="encodings.pickle",
	help="path to serialized db of facial encodings")
ap.add_argument("-p", "--usePiCamera", type=int, required=False, default=1,
	help="Is using picamera or builtin/usb cam")
ap.add_argument("-s", "--source", required=False, default=0,
	help="Use 0 for /dev/video0 or 'http://link.to/stream'")
ap.add_argument("-r", "--rotateCamera", type=int, required=False, default=0,
	help="rotate camera")
ap.add_argument("-m", "--method", type=str, required=False, default="dnn",
	help="method to detect faces (dnn, haar)")
ap.add_argument("-d", "--detectionMethod", type=str, required=False, default="hog",
	help="face detection model to use: either `hog` or `cnn`")
ap.add_argument("-i", "--interval", type=int, required=False, default=2000,
	help="interval between recognitions")
ap.add_argument("-o", "--output", type=int, required=False, default=1,
	help="Show output")
ap.add_argument("-eds", "--extendDataset", type=str2bool, required=False, default=False,
	help="Extend Dataset with unknown pictures")
ap.add_argument("-ds", "--dataset", required=False, default="../dataset/",
	help="path to input directory of faces + images")
ap.add_argument("-t", "--tolerance", type=float, required=False, default=0.60,
	help="How much distance between faces to consider it a match. Lower is more strict.")
args = vars(ap.parse_args())


printjson("status", "loading encodings + face detector...")
data = pickle.loads(open(args["encodings"], "rb").read())
detector = cv2.CascadeClassifier(args["cascade"])

printjson("status", "starting video stream...")

vs = VideoStream(src=0).start()
time.sleep(2.0)

def create():
    global vs
    count = 1
    sub_data = input("enter your name:")
    path = os.path.join(datasets, sub_data)
    if not os.path.isdir(path):
        os.mkdir(path)
    (width, height) = (130, 100)
    while count < 50:
        print(count)
        im = vs.read()
        cv2.imwrite('%s/%s.png' % (path,count), im)
        count += 1
##        cv2.imshow('OpenCV', im)
##        cv2.waitKey(1)
##    vs.stop()


prevNames = []

if args["extendDataset"] is True:
	unknownPath = os.path.dirname(args["dataset"] + "unknown/")
	try:
			os.stat(unknownPath)
	except:
			os.mkdir(unknownPath)

tolerance = float(args["tolerance"])


fps = FPS().start()

while True:
        

	originalFrame = vs.read()
	outfile = '%s.jpg' % (str(datetime.now()))
	cv2.imwrite("/home/pi/dev/sopare/images/"+outfile,originalFrame)
	cv2.imwrite("/home/pi/main/input.jpg",originalFrame)
	frame = imutils.resize(originalFrame, width=500)

	if args["method"] == "dnn":

		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		boxes = face_recognition.face_locations(rgb,
			model=args["detectionMethod"])
	elif args["method"] == "haar":

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


		rects = detector.detectMultiScale(gray, scaleFactor=1.1,
			minNeighbors=5, minSize=(30, 30),
			flags=cv2.CASCADE_SCALE_IMAGE)


		boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]


	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []


	for encoding in encodings:

		distances = face_recognition.face_distance(data["encodings"], encoding)

		minDistance = 1.0
		if len(distances) > 0:

			minDistance = min(distances)

		if minDistance < tolerance:
			idx = np.where(distances == minDistance)[0][0]
			name = data["names"][idx]
			now = str(datetime.now())

			data1 = "name: "+name+"  date time"+now
			outfile = '%s.txt' % (str(datetime.now()))
			f=open("/home/pi/dev/sopare/images/"+outfile,'w')
			f.write(data1)
			f.close()
			aws_s3.upload()			
			subprocess.Popen("sudo python /home/pi/main/mail.py",shell=True)
			print(name)
		else:
                        name = "unknown"
                        print(name)
                        subprocess.Popen("sudo python /home/pi/main/mail.py",shell=True)
                        f=open("log.txt",'w')
                        f.write("unknown")
                        f.close()
                        create()
                        subprocess.Popen("sudo python3 /home/pi/main/encode.py --dataset /home/pi/main/datasets --encodings /home/pi/main/encodings.pickle",shell=True)
                        while(1):
                                f = open("log.txt", "r")
                                rcv=f.read()
                                f.close()
                                if(rcv=="done"):
                                        f = open("log.txt", "w")
                                        rcv=f.write(" ")
                                        f.close()
                                        break
                        data = pickle.loads(open(args["encodings"], "rb").read())
##                        vs.start()

                            

		names.append(name)

	for ((top, right, bottom, left), name) in zip(boxes, names):
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		txt = name + " (" + "{:.2f}".format(minDistance) + ")"
		cv2.putText(frame, txt, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)


	if (args["output"] == 1):
		cv2.imshow("Frame", frame)


	fps.update()


	prevNames = names

	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q") or closeSafe == True:
		break

	time.sleep(args["interval"] / 1000)

fps.stop()
printjson("status", "elasped time: {:.2f}".format(fps.elapsed()))
printjson("status", "approx. FPS: {:.2f}".format(fps.fps()))


cv2.destroyAllWindows()
vs.stop()
