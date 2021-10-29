import cv2
import numpy as np

cameraLeft = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cameraRight = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# cameraLeft.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cameraLeft.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cameraRight.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cameraRight.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
scale = 40
B = 5.5
f = 803
t = 4

minDisparity = 2
numDisparities = 48
stereo = cv2.StereoBM_create(numDisparities, blockSize=21)
cv_file = cv2.FileStorage("improved_params.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()

cv_file_Q = cv2.FileStorage("Q.xml", cv2.FILE_STORAGE_READ)
Q = cv_file_Q.getNode("Q").mat()

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

value_pairs = []
max_dist = 170 # max distance to keep the target object (in cm)
min_dist = 20 # Minimum distance the stereo setup can measure (in cm)
sample_delta = 10
Z = max_dist


cv_file = cv2.FileStorage("M.xml", cv2.FILE_STORAGE_READ)
M = cv_file.getNode("M").real()
cv_file.release()

def onMouse(event, x, y, flag, disparity_normalized):
    global Z
    if event == cv2.EVENT_LBUTTONDOWN:
        distance = disparity_normalized[y][x]
        value_pairs.append([Z, distance])
        print("Distance: %r cm  | Disparity: %r" % (Z, distance))
        Z -= sample_delta
        return distance


def onMouseGetDistance(event, x, y, flag,  depth_map):
    if event == cv2.EVENT_LBUTTONDOWN:
        distance = depth_map[y][x]
        print("Distance = %r cm" % (distance))

cv2.namedWindow("DepthMap", cv2.WINDOW_NORMAL)
cv2.resizeWindow("DepthMap", 640, 480)

while True:
    retL, frameL = cameraLeft.read()
    retR, frameR = cameraRight.read()

    if retL and retR:
        leftRemap = cv2.remap(frameL, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        rightRemap = cv2.remap(frameR, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        grayL = cv2.cvtColor(leftRemap, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rightRemap, cv2.COLOR_BGR2GRAY)

        disparity = stereo.compute(grayL, grayR)
        disparity_normalized = cv2.normalize(disparity, None, min_dist, max_dist, cv2.NORM_MINMAX)
        disparity = disparity.astype(np.float32)

        # Scaling down the disparity values and normalizing them
        disparity = (disparity /16.0 - minDisparity)/numDisparities

        depth_map = M / (disparity)
        # image_3d_reprojection = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
        image = np.array(depth_map, dtype=np.uint8)
        disparity_color = cv2.applyColorMap(image, cv2.COLORMAP_JET)

        # cv2.setMouseCallback("DepthMap", onMouse, disparity)
        cv2.setMouseCallback("DepthMap", onMouseGetDistance, depth_map)

        cv2.imshow('Left', leftRemap)
        cv2.imshow('Right', rightRemap)
        left_stacked = cv2.addWeighted(leftRemap, 0.5, disparity_color, 0.5, 0.0)
        cv2.imshow("DepthMap", left_stacked)

        # if Z < min_dist:
        #     break

    if cv2.waitKey(1) == 27:
        break

val_par = np.array(value_pairs)
disparity = val_par[:, 1]
z = val_par[:, 0]

disp_inv = 1 / disparity
coeff = np.vstack([disp_inv, np.ones(len(disp_inv))]).T
ret, sol = cv2.solve(coeff, z, flags=cv2.DECOMP_QR)
M = sol[0, 0]
print(M)

# cv_file = cv2.FileStorage("M.xml", cv2.FILE_STORAGE_WRITE)
# cv_file.write("M", M)
# cv_file.release()

cameraLeft.release()
cameraRight.release()
cv2.destroyAllWindows()
