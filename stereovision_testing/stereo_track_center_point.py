import cv2
import numpy as np

def coords_mouse_disp(x, y, disp):
    global Distance
    average=0
    for u in range(-1, 2):
        for v in range(-1, 2):
            average += disp[y+u, x+v]
    average = average / 9
    Distance = -593.97 * average ** 3 + 1506.8 * average ** 2 - 1373.1 * average + 522.06 #polynomial regression =n of order 3
    Distance = np.around(Distance * 0.01, decimals=2)       #rounding of array to 2 decimal places
    print('Distance: ' + str(Distance)+' m')


cv_file = cv2.FileStorage("improved_params.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()
# Create StereoSGBM and prepare all parameters
window_size = 3
min_disp = 2
num_disp = 130 - min_disp
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=window_size,
                               uniquenessRatio=10,
                               speckleWindowSize=100,
                               speckleRange=32,
                               disp12MaxDiff=5,
                               P1=8 * 3 * window_size ** 2,
                               P2=32 * 3 * window_size ** 2)

# Used for the filtered image
stereoR = cv2.ximgproc.createRightMatcher(stereo)  # Create another stereo for right this time

# WLS FILTER Parameters
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

# Filtering
kernel = np.ones((3, 3), np.uint8)

# *************************************
# ***** Starting the StereoVision *****
# *************************************
# Call the two cameras
CamR = cv2.VideoCapture(0)  # When 0 then Right Cam and when 2 Left Cam
CamL = cv2.VideoCapture(1)


while True:
    retR, frameR = CamR.read()
    retL, frameL = CamL.read()

    center_X = int(frameL.shape[1] / 2)
    center_Y = int(frameL.shape[0] / 2)

    # Rectify the images on rotation and alignement
    Left_nice = cv2.remap(frameL,Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)  # Rectify the image using the calibration parameters found during the initialisation
    Right_nice = cv2.remap(frameR,Right_Stereo_Map_x ,Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    grayR = cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)

    # Compute the 2 images for the Depth_image
    disp = stereo.compute(grayL, grayR)#.astype(np.float32) / 16

    disp = ((disp.astype(np.float32) / 16) - min_disp)/num_disp # Calculation allowing us to have 0 for the most distant object able to detect


    coords_mouse_disp(center_X, center_Y, disp)
    cv2.circle(frameL, (center_X, center_Y), 10, (0, 255, 255))
    cv2.imshow("Disparity", frameL)

    if cv2.waitKey(1) == 27:
        break

# Release the Cameras
CamR.release()
CamL.release()
cv2.destroyAllWindows()