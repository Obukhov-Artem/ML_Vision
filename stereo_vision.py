import cv2
import numpy as np

def coords_mouse_disp(event,x,y,flags,param):
    global Distance
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(x, y, disp[y,x], filteredImg[y,x])
        average=0
        for u in range(-1, 2):
            for v in range(-1, 2):
                average += disp[y+u,x+v]
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
kernel= np.ones((3,3),np.uint8)

# *************************************
# ***** Starting the StereoVision *****
# *************************************
# Call the two cameras
CamR = cv2.VideoCapture(0)  # When 0 then Right Cam and when 2 Left Cam
CamL = cv2.VideoCapture(1)

while True:
    retR, frameR = CamR.read()
    retL, frameL = CamL.read()

    # Rectify the images on rotation and alignement
    Left_nice = cv2.remap(frameL,Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)  # Rectify the image using the calibration parameters found during the initialisation
    Right_nice = cv2.remap(frameR,Right_Stereo_Map_x ,Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    grayR = cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)

    # Compute the 2 images for the Depth_image
    disp = stereo.compute(grayL, grayR)#.astype(np.float32) / 16
    dispL = disp
    dispR = stereoR.compute(grayR,grayL)
    dispL = np.int16(dispL)
    dispR = np.int16(dispR)

    # Using the WLS filter
    filteredImg = wls_filter.filter(dispL, grayL, None, dispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)
    cv2.imshow('Disparity Map', filteredImg)

    disp = ((disp.astype(np.float32) / 16) - min_disp)/num_disp # Calculation allowing us to have 0 for the most distant object able to detect

    # Filtering the Results with a closing filter
    closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise)

    # Colors map
    dispc = (closing - closing.min()) * 255
    dispC = dispc.astype(np.uint8)                                   # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
    disp_Color = cv2.applyColorMap(dispC,cv2.COLORMAP_OCEAN)         # Change the Color of the Picture into an Ocean Color_Map
    filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_JET)
    # Show the result for the Depth_image
    # cv2.imshow('Disparity', disp)
    #cv2.imshow('Closing', closing)
    cv2.imshow('Color Depth', disp_Color)
    cv2.imshow('Filtered Color Depth', filt_Color)

    cv2.setMouseCallback("Filtered Color Depth", coords_mouse_disp, dispC)

    # Keyevent press for motion detection
    #    if keyboard.is_pressed('m'):
    #        motion()
    # End the Programme
    if cv2.waitKey(1) == 27:
        break

# Release the Cameras
CamR.release()
CamL.release()
cv2.destroyAllWindows()