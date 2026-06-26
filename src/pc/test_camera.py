import cv2 as cv

for index in range(5):
    camera = cv.VideoCapture(index)

    if camera.isOpened():
        print("Camera found at index:", index)

        res, frame = camera.read()
        if res:
            cv.imshow("camera {}".format(index), frame)
            cv.waitKey(1000)

    camera.release()

cv.destroyAllWindows()