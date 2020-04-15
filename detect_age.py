import numpy as np
import argparse
import imutils
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input image")
ap.add_argument("-f", "--face", required=True,
                help="Path to face detector model")
ap.add_argument("-a", "--age", required=True,
                help="Path to age detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="Min probability to filter weak predictions")
args = vars(ap.parse_args())

AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)",
               "(25-32)", "(38-43)", "(48-53)", "(60-100)"]

print("[INFO] Loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join(
    [args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] Loading Age detector model...")
prototxtPath = os.path.sep.join([args["age"], "age_deploy.prototxt"])
weightsPath = os.path.sep.join([args["age"], "age_net.caffemodel"])

ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)

image = cv2.imread(args["image"])
image = imutils.resize(image,width=400)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

print("[INFO] Computing Face detections...")
faceNet.setInput(blob)
detections = faceNet.forward()

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > args["confidence"]:
        box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
        startX, startY, endX, endY = box.astype("int")
        face = image[startY:endY, startX:endX]
        faceBlob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        ageNet.setInput(faceBlob)
        preds = ageNet.forward()
        i = preds[0].argmax()
        age = AGE_BUCKETS[i]
        ageConfidence = preds[0][i]

        text = "{}: {:.2f}%".format(age, ageConfidence * 100)
        print("[INFO] {}".format(text))

        y = startY - 10 if startY - 10 > 0 else startY+10
        cv2.rectangle(image,(startX,startY),(endX,endY),(0,0,255),2)
        cv2.putText(image,text,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)

cv2.imshow("Image",image)
cv2.waitKey(0)