# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
from send_mail import send_mail


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=10000, help="minimum area size")
ap.add_argument("-m","--is_mail",type=int, default=0,help="is send mail out")
args = vars(ap.parse_args())


# initialize the first frame in the video stream
firstFrame = None
switch_period = 5			# change first frame every 5 mins
start_time = datetime.datetime.now() # start time
# output for saving frames
# width = vs.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)   # float
# height = vs.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) # float
out = None
CONST_TIME = 1000
width = 640
height = 480
save_frame_num = CONST_TIME
save_flag = False


# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	vs = VideoStream(src=0,resolution=(width,height)).start()
	time.sleep(2.0)

# otherwise, we are reading from a video file
else:
	vs = cv2.VideoCapture(args["video"])


# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	frame = vs.read()
	frame = frame if args.get("video", None) is None else frame[1]
	text = "Unoccupied"

	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if frame is None:
		break

	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=width,height=height)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue

	# compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < args["min_area"]:
			continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "Occupied"

	# draw the text and timestamp on the frame
	cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

	# show the frame and record if the user presses a key
	cv2.imshow("Security Feed", frame)
	cv2.imshow("Thresh", thresh)
	cv2.imshow("Frame Delta", frameDelta)
	key = cv2.waitKey(1) & 0xFF

	# save to disk
	if text == 'Occupied' and save_flag == False:
		record_time = datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")
		ref_frame = frame
		if out is None:
			out = cv2.VideoWriter('../save_'+ record_time +'.mp4',\
				cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (int(width),int(height)))
		save_flag = True
	if save_flag == True:
		if save_frame_num>=0:
			out.write(frame)
			save_frame_num = save_frame_num - 1
		else:
			save_flag = False
			save_frame_num = CONST_TIME
			out.release()
			out = None
			if args['is_mail'] == 1:
				send_mail('warning','there is a moving detected at '+ record_time,cv2.imencode('.jpg',ref_frame)[1].tostring())
	
	# switch reference frame
	if ((datetime.datetime.now() - start_time).seconds / 60) > switch_period\
		and text == "Unoccupied":
		firstFrame = gray
		start_time = datetime.datetime.now() # reset start time

	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()