import cv2
import numpy as np
import os
import glob

cameraLeft = cv2.VideoCapture(0)
cameraRight = cv2.VideoCapture(1)

# cameraLeft.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cameraLeft.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cameraRight.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cameraRight.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
scale = 40

CHECKERBOARD = (9, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objectpoints = []
imgpointsLeft = []
imgpointsRight = []

image_count = 0

# def change_scale(frame):
#     height, width, channels = frame.shape
#
#     # prepare the crop
#     centerX, centerY = int(height / 2), int(width / 2)
#     radiusX, radiusY = int(scale * height / 100), int(scale * width / 100)
#
#     minX, maxX = centerX - radiusX, centerX + radiusX
#     minY, maxY = centerY - radiusY, centerY + radiusY
#
#     cropped = frame[minX:maxX, minY:maxY]
#     resized_cropped = cv2.resize(cropped, (width, height))
#
#     return resized_cropped

while True:
    retL, frameL = cameraLeft.read()
    retR, frameR = cameraRight.read()

    if retL and retR:
        cv2.imshow('Left', frameL)
        cv2.imshow('Right', frameR)
    else:
        break

    key = cv2.waitKey(40)

    if key == 27:
        break
    elif key == 13:
        image_count += 1
        cv2.imwrite("./images/left_camera/image{0}.png".format(image_count), frameL)
        cv2.imwrite("./images/right_camera/image{0}.png".format(image_count), frameR)
        print("images saved!!!")

cameraLeft.release()
cameraRight.release()
cv2.destroyAllWindows()

leftImages = glob.glob("./images/left_camera/*.png")
rightImages = glob.glob("./images/right_camera/*.png")

for filenameL, filenameR in zip(leftImages, rightImages):
    imageL = cv2.imread(filenameL)
    imageR = cv2.imread(filenameR)
    grayL = cv2.cvtColor(imageL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imageR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if retL and retR:
        objectpoints.append(objp)

        cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        imgpointsLeft.append(cornersL)
        imgpointsRight.append(cornersR)

        cv2.drawChessboardCorners(imageL, CHECKERBOARD, cornersL, retL)
        cv2.drawChessboardCorners(imageR, CHECKERBOARD, cornersR, retR)

        cv2.imshow('corners left', imageL)
        cv2.imshow('corners right', imageR)

        cv2.waitKey(1000)


cv2.destroyAllWindows()


retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objectpoints, imgpointsLeft, grayL.shape[::-1], None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objectpoints, imgpointsRight, grayL.shape[::-1], None, None)

print(mtxL)
print(mtxR)
height, weight = imageL.shape[:2]
OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (height, weight), 1, (height, weight))

height, weight = imageR.shape[:2]
OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (height, weight), 1, (height, weight))

flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objectpoints, imgpointsLeft, imgpointsRight, mtxL, distL, mtxR, distR, grayL.shape[::-1], criteria, flags)

RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS, grayL.shape[::-1], R, T, 0, (0, 0))

Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL, grayL.shape[::-1], cv2.CV_16SC2)
Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, grayL.shape[::-1], cv2.CV_16SC2)

cv_file = cv2.FileStorage("improved_params.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])

cv_file_Q = cv2.FileStorage("Q.xml", cv2.FILE_STORAGE_WRITE)
cv_file_Q.write("Q", Q)

cv_file.release()
cv_file_Q.release()

print('Parameters saved!!!')

