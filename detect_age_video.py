from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_age(frame,faceNet,age_net,minConf = 0.5):

    AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)",
               "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
    results = []

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > minConf:
            box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")
            face = frame[startY:endY, startX:endX]
            if face.shape[0]<20 or face.shape[1]<20:
                continue
            faceBlob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            ageNet.setInput(faceBlob)
            preds = ageNet.forward()
            i = preds[0].argmax()
            age = AGE_BUCKETS[i]
            ageConfidence = preds[0][i]

            d={
                "loc":(startX,startY,endX,endY),
                "age":(age,ageConfidence)
            }
            results.append(d)
    return results


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True,
                help="Path to face detector model")
ap.add_argument("-a", "--age", required=True,
                help="Path to age detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="Min probability to filter weak predictions")
args = vars(ap.parse_args())


print("[INFO] Loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join(
    [args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] Loading Age detector model...")
prototxtPath = os.path.sep.join([args["age"], "age_deploy.prototxt"])
weightsPath = os.path.sep.join([args["age"], "age_net.caffemodel"])

ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] Starting videostream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame,width=800)
    
    results = detect_and_predict_age(frame,faceNet,ageNet,args["confidence"])

    for r in results:
        text = "{}: {:.2f}%".format(r["age"][0], r["age"][1] * 100)
        (startX,startY,endX,endY) = r["loc"]
        y = startY - 10 if startY - 10 > 0 else startY+10
        cv2.rectangle(frame,(startX,startY),(endX,endY),(0,0,255),2)
        cv2.putText(frame,text,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)
    cv2.imshow("VideoStream",frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
vs.stop()