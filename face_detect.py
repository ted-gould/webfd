#!/usr/bin/python2
import sys, cv2
sys.path.append('/usr/lib/python2.7/dist-packages')

# Get user supplied values
cascPath = sys.argv[1]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
video_capture = cv2.VideoCapture(0)
ret, image = video_capture.read()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

#print "Found {0} faces!".format(len(faces))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# TODO:
# Buscar la llamada para en lugar de hacer un imshow, simplemente sobre-escriba el archivo
#cv2.imshow("Faces found", image)
cv2.imwrite("processed_shot.jpg", image)

cv2.waitKey(0)
