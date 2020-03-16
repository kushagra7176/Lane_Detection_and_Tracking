import cv2
import time
import glob
import numpy as np
from skimage.exposure import rescale_intensity
from scipy import ndimage
import matplotlib.pyplot as plt

left_point1, left_point2, left_point3 = [], [], []
right_point1, right_point2, right_point3 = [], [], []


##########################################################################################################################


##########################################################################################################################

def lane_detection(img, num_windows=20, margin=23, minpix=1, draw_windows=True):
    global left_point1, left_point2, left_point3, right_point1, right_point2, right_point3
    left_fit_ = np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img)) * 255

    # Fine histogram of the input image
    histogram = hist = np.sum(img[img.shape[0]//255:,:], axis=0)

    # Divide warped image into left and right halves and find respective histogram peaks
    image_midpoint = int(histogram.shape[0] / 2)
    leftx_half_peak = np.argmax(histogram[:image_midpoint])
    rightx_half_peak = np.argmax(histogram[image_midpoint:]) +350#+ midpoint


    # Set height of window
    window_height = np.int(img.shape[0] / num_windows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero_pixels = img.nonzero()
    nonzero_pixels_y = np.array(nonzero_pixels[0])
    nonzero_pixels_x = np.array(nonzero_pixels[1])

    # Current positions to be updated for each window
    leftx_current_pixel = leftx_half_peak
    rightx_current_pixel = rightx_half_peak

    # Created empty lists for left and right lane pixel indices
    left_lane_indices = []
    right_lane_indices = []

    # Step through the windows one by one
    for window in range(num_windows):

        # Identify window boundaries for right and left lanes.
        win_y_bottom = img.shape[0] - (window + 1) * window_height
        win_y_top = img.shape[0] - window * window_height

        win_x_leftLane_bottom = leftx_current_pixel - margin
        win_x_leftLane_top = leftx_current_pixel + margin

        win_x_right_bottom = rightx_current_pixel - margin
        win_x_right_top = rightx_current_pixel + margin

        # Draw the windows on the visualization image. (Uncomment the following block to draw the sliding window boxes.)
        # if draw_windows == True:
        #     cv2.rectangle(out_img, (win_x_leftLane_bottom, win_y_bottom), (win_x_leftLane_top, win_y_top),
        #                   (100, 255, 255), 3)
        #     cv2.rectangle(out_img, (win_x_right_bottom, win_y_bottom), (win_x_right_top, win_y_top),
        #                   (100, 255, 255), 3)

        # Identify the nonzero pixels in x and y within the window
        optimal_leftLane_indices = ((nonzero_pixels_y >= win_y_bottom) & (nonzero_pixels_y < win_y_top) &
                                    (nonzero_pixels_x >= win_x_leftLane_bottom) & (nonzero_pixels_x < win_x_leftLane_top)).nonzero()[0]

        optimal_rightLane_indices = ((nonzero_pixels_y >= win_y_bottom) & (nonzero_pixels_y < win_y_top) &
                                    (nonzero_pixels_x >= win_x_right_bottom) & (nonzero_pixels_x < win_x_right_top)).nonzero()[0]

        # Append these indices to the lists
        left_lane_indices.append(optimal_leftLane_indices)
        right_lane_indices.append(optimal_rightLane_indices)

        # Recenter next window on the mean optimal lane pixels.
        if len(optimal_leftLane_indices) > minpix:
            leftx_current_pixel = np.int(np.mean(nonzero_pixels_x[optimal_leftLane_indices]))
        if len(optimal_rightLane_indices) > minpix:
            rightx_current_pixel = np.int(np.mean(nonzero_pixels_x[optimal_rightLane_indices]))

    # Combine the arrays of indices into one final lane indices list
    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)

    # Extract left and right line pixel positions
    leftLane_x_loc = nonzero_pixels_x[left_lane_indices]
    leftLane_y_loc = nonzero_pixels_y[left_lane_indices]
    rightLane_x_loc = nonzero_pixels_x[right_lane_indices]
    rightLane_y_loc = nonzero_pixels_y[right_lane_indices]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(leftLane_y_loc, leftLane_x_loc, 2)
    right_fit = np.polyfit(rightLane_y_loc, rightLane_x_loc, 2)

    left_point1.append(left_fit[0])
    left_point2.append(left_fit[1])
    left_point3.append(left_fit[2])

    right_point1.append(right_fit[0])
    right_point2.append(right_fit[1])
    right_point3.append(right_fit[2])

    a = -10
    left_fit_[0] = np.mean(left_point1[a:])
    left_fit_[1] = np.mean(left_point2[a:])
    left_fit_[2] = np.mean(left_point3[a:])

    right_fit_[0] = np.mean(right_point1[a:])
    right_fit_[1] = np.mean(right_point2[a:])
    right_fit_[2] = np.mean(right_point3[a:])

    # Generate x and y values for plotting
    plot_y = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit_[0] * plot_y ** 2 + left_fit_[1] * plot_y + left_fit_[2]
    right_fitx = right_fit_[0] * plot_y ** 2 + right_fit_[1] * plot_y + right_fit_[2]
    m_left = (plot_y[0] - plot_y[int(len(plot_y) / 2)]) / (left_fitx[0] - left_fitx[int(len(left_fitx) / 2)])  # .........

    #Uncomment the following lines to colour the detected pixels.
    # out_img[nonzero_pixels_y[left_lane_indices], nonzero_pixels_x[left_lane_indices]] = [255, 0, 100]
    # out_img[nonzero_pixels_y[right_lane_indices], nonzero_pixels_x[right_lane_indices]] = [0, 100, 255]

    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), plot_y, m_left


def get_curve(img, leftLane_x_loc, rightLane_x_loc):
    plot_y = np.linspace(0, img.shape[0] - 1, img.shape[0])
    y_eval = np.max(plot_y)
    ym_per_pix = 30.5 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 720  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(plot_y * ym_per_pix, leftLane_x_loc * xm_per_pix, 2)
    right_fit_cr = np.polyfit(plot_y * ym_per_pix, rightLane_x_loc * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curve_radius = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                         2 * left_fit_cr[0])
    right_curve_radius = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                         2 * right_fit_cr[0])

    current_pos = img.shape[1] / 2
    left_x_pos = left_fit_cr[0] * img.shape[0] ** 2 + left_fit_cr[1] * img.shape[0] + left_fit_cr[2]
    right_x_pos = right_fit_cr[0] * img.shape[0] ** 2 + right_fit_cr[1] * img.shape[0] + right_fit_cr[2]
    lane_center_position = (right_x_pos + left_x_pos) / 2
    center_position = (current_pos - lane_center_position) * xm_per_pix / 10

    return (left_curve_radius, right_curve_radius, center_position)



def draw_lanes(img, left_fit, right_fit):

    plot_y = np.linspace(0, img.shape[0] - 1, img.shape[0])
    lane = np.zeros_like(img)
    left = np.array([np.transpose(np.vstack([left_fit, plot_y]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, plot_y])))])
    points = np.hstack((left, right))
    cv2.fillPoly(lane, np.int_(points), (0, 175, 255))

    return lane

def draw_data(m_left):  # .........................
    if (0 < m_left < 25):
        text = "Left Curve"
    elif (-25 < m_left < 0):
        text = "Right Curve"
    else:
        text = "Straight Road"

    return text

##########################################################################################################################


#########################################################################################################################

fourcc= cv2.VideoWriter_fourcc(*'XVID')
out= cv2.VideoWriter('Question2_Data_2.avi', fourcc, 20.0, (1280, 720))
img_count = -1

ploty_list = []
im_fill_list = []
curves_list = []
m_left_list = []
# curves_1_list = []

cap = cv2.VideoCapture('Problem 2\data_2\challenge_video.mp4')
if (cap.isOpened()== False):
    print("Error opening video stream or file")

while cap.isOpened():
    ret,img = cap.read()

    img_count =img_count+1
    print("image number:",img_count)
    img = cv2.resize(img, (1280, 720))

    dimensions = img.shape
    # cv2.circle(img, (193, 500), 5, (0, 0, 255), -1)#red
    # cv2.circle(img, (720, 470), 5, (0, 255, 255), -1)#yellow
    # cv2.circle(img, (583, 280), 5, (255, 0, 255), -1)#pink
    # cv2.circle(img, (723, 280), 5, (255, 0, 0), -1)#blue

    # print("dim",dimensions)

    src = np.float32([[550,500 ],[810,500],[240, 690],[1260, 690]])
    dst = np.float32([[0,0],[700, 0],[0, 720],[700, 720]])
    matrix = cv2.getPerspectiveTransform(src,dst)
    warped = cv2.warpPerspective(img,matrix,(700,720))
    # warped_gray = cv2.cvtColor(warped, cv2.Co)

    warped_b,warped_g,warped_r = cv2.split(warped)
    # #
    warped_LUV = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)
    # warped_LUV = cv2.resize(warped_LUV, (300, 300))
    warped_L_luv, warped_U_luv, warped_V_luv = cv2.split(warped_LUV)

    warped_HLS = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)
    # warped_HLS = cv2.resize(warped_HLS, (300, 300))
    warped_H, warped_L, warped_S = cv2.split(warped_HLS)

    warped_HSV = cv2.cvtColor(warped,cv2.COLOR_BGR2HSV)
    # warped_HSV = cv2.resize(warped_HSV,(300,300))
    warped_H_hsv, warped_S_hsv, warped_V_hsv= cv2.split(warped_HSV)
    # # warped_La, warped_A, warped_B \
    warped_LAB= cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    # warped_LAB = cv2.resize(warped_LAB, (300, 300))
    warped_La, warped_A, warped_B = cv2.split(warped_LAB)
    warp_r = cv2.resize(warped_r,(300,300))
    # cv2.imshow("Images", np.vstack([np.hstack([warped_S, warped_S_hsv, warped_V_hsv]), np.hstack([warped_L_luv, warped_U_luv, warped_V_luv])]))
    # cv2.imshow("warped: S",warped_HLS[:,:,2])

    clahe1 = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(16, 16))

    # adjusted1 = adjust_gamma(warped_r, gamma=1.0)
    # cv2.imshow("gamma:", adjusted1)
    # cl1 = clahe1.apply(adjusted1)
    # cv2.imshow("cl1:", cl1)


    # cv2.imshow("warped: G", warped_g)
    # cv2.imshow("warped: R", warped_r)         #-------------------------------------
    # ret, warped_gray_thresh = cv2.threshold(warped_V_hsv, 175, 255, cv2.THRESH_BINARY)
    ret,warped_gray_thresh_right = cv2.threshold(warped_V_hsv, 195,255,cv2.THRESH_BINARY)
    ret, warped_gray_thresh_left = cv2.threshold(warped_S_hsv, 100, 255, cv2.THRESH_BINARY)
    # warped_gray_thresh_left = cv2.bitwise_not(warped_gray_thresh_left)
    # canny_edge = cv2.Canny(warped_gray_thresh, 100, 300 ,13, L2gradient=True)

    # laplacian = cv2.Laplacian(warped,cv2.CV_64F)

    # sobelx = cv2.Sobel(warped,cv2.CV_64F,1,0,ksize=5)
    # sobely = cv2.Sobel(warped,cv2.CV_64F,0,1,ksize=5)
    # sobel = sobelx+sobely

    warped_gray_thresh = cv2.add(warped_gray_thresh_right,warped_gray_thresh_left)

    # cv2.imshow("warp_fin:  img no:%d" %(img_count), warped_gray_thresh)    #-------------------------------------
    # cv2.imshow("warp_right:  img no:%d" % (img_count), warped_gray_thresh_right)  # -------------------------------------
    # cv2.imshow("warp_left:  img no:%d" % (img_count), warped_gray_thresh_left)  # -------------------------------------

    #####################################################################################################################

    # out_img, curves, lanes, ploty = lane_detection(warped_gray_thresh)
    try:
        out_img, curves, lanes, ploty,m_left = lane_detection(warped_gray_thresh)
        ploty_list.append(ploty)
        img_fill = draw_lanes(img, curves[0], curves[1])
        im_fill_list.append(img_fill)
        curves_list.append(curves)
        m_left_list.append(m_left)
    except:
        curves = curves_list[2]
        ploty = ploty_list[2]
        m_left = m_left_list[2]
        img_fill = draw_lanes(img, curves[0], curves[1])
        print("poor Quality image")

    # print("len curve",len(curves),img_count, curves[0])
    print(" CCurves:", curves[0][0], curves[1][0])
    if curves[0][0]> 900 or curves[1][0]<400:
        # print("wrong curves:",curves[0][0],curves[1][0])
        ind = len(curves_list)
        if ind >2:
            curves = curves_list[2]
            img_fill = draw_lanes(img, curves[0], curves[1])
            # print("Updated curves",curves)
    # cv2.imshow("out img",out_img)
    # #
    # #
    # # # plt.imshow(out_img)
    # # # plt.plot(curves[0], ploty, color='yellow', linewidth=1)
    # # # plt.plot(curves[1], ploty, color='yellow', linewidth=1)
    # # # print(np.asarray(curves).shape)
    curve_rad = get_curve(img, curves[0], curves[1])  # .....................
    # print("curverad:",curverad)
    lane_curve = np.mean([curve_rad[0], curve_rad[1]])  # ..................
    # print("curve0",ploty)
    # print("len curve:",len(curves))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_fill = draw_lanes(img, curves[0], curves[1])
    for i in range(len(curves[0])):
        cv2.circle(out_img, (int(curves[0][i]), int(ploty[i])), 5, (255, 100, 255), -1)
        cv2.circle(out_img, (int(curves[1][i]), int(ploty[i])), 5, (0, 0, 255), -1)
        # print("curve 0:",curves[0][i])# yellow
        # print("curve 1:", int(curves[1][i]))
    # # cv2.imshow("hsv",img)
    #
    matrix_inv_img_fill = cv2.getPerspectiveTransform(dst, src)
    warped_inv_img_fill = cv2.warpPerspective(img_fill, matrix_inv_img_fill, (1280, 720))

    # cv2.imshow("sliding img no:%d"%(img_count),out_img)    #-------------------------------------

    matrix_inv = cv2.getPerspectiveTransform(dst, src)
    warped_inv = cv2.warpPerspective(out_img, matrix_inv, (1280, 720))
    # # cv2.imshow("inv-warp",warped_inv)
    warped_inv_bin = cv2.cvtColor(warped_inv, cv2.COLOR_BGR2GRAY)
    ret, warped_inv_bin_thresh = cv2.threshold(warped_inv_bin, 50, 255, cv2.THRESH_BINARY)
    # # cv2.imshow("inv-warp", warped_inv_bin_thresh)
    # # trial_img = cv2.subtract(gray, warped_inv_bin_thresh)
    # # trial_imgi = cv2.cvtColor(trial_img, cv2.COLOR_GRAY2BGR)
    test = cv2.bitwise_and(img,img,mask = cv2.bitwise_not(warped_inv_bin_thresh))
    # # cv2.imshow("trial", test)

    # cv2.imshow("finalimg no:%d", cv2.add(test, warped_inv))
    output_image = (cv2.add(cv2.add(test, warped_inv), warped_inv_img_fill))#......................
    text = draw_data(m_left)  # ......................
    # cv2.imshow("final",cv2.add(cv2.add(test,warped_inv),warped_inv_img_fill))
    cv2.putText(output_image, 'Lane Curvature: {:.0f} m'.format(lane_curve), (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 3)
    cv2.putText(output_image, 'Vehicle offset: {:.4f} m'.format(curve_rad[2]), (100, 200), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 3)
    cv2.putText(output_image, text, (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imshow("final", output_image)
    out.write(output_image)
    # cv2.imshow("final",cv2.add(cv2.add(test,warped_inv),warped_inv_img_fill))
    ######################################################################################################################


    # cv2.imshow("frame",img)
    if cv2.waitKey(1) == 27:
        # cv2.destroyAllWindows()
        break
cv2.destroyAllWindows()
