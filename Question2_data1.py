import cv2
import time
import glob
import numpy as np
from skimage.exposure import rescale_intensity
from scipy import ndimage
import matplotlib.pyplot as plt


left_point1, left_point2, left_point3 = [], [], []
right_point1, right_point2, right_point3 = [], [], []

def load_Images():
    original=[]
    title=[]
    for f in glob.iglob("Problem 2\data_1\data\*"):
            img = cv2.imread(f)
            # print("Title:",f)
            # image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            original.append(img)
            title.append(f)
    time.sleep(1)
    length=len(original)
    print('Updated Images')
    return original,title,length



##########################################################################################################################


##########################################################################################################################

def lane_Detection(img, num_windows=20, margin=23, minpix=1, draw_windows=True):
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

def draw_data(m_left):

    # print("mleft",m_left)
    if (0 < m_left < 15):
        text = "Left Curve"
    elif (-15 < m_left < 0):
        text = "Right Curve"
    else:
        text = "Straight Road"

    return text


##########################################################################################################################
#----------------------------------------------     MAIN     --------------------------------------------------------------
#########################################################################################################################

fourcc= cv2.VideoWriter_fourcc(*'XVID')
out= cv2.VideoWriter('Question2_Data_1.avi', fourcc, 20.0, (1392, 512))

smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
img_count = 0
original,title,length =load_Images()
for img in original:
    img_count =img_count+1
    print("image number:",img_count)

    dimensions = img.shape
    # cv2.circle(img, (193, 500), 5, (0, 0, 255), -1)#red
    # cv2.circle(img, (903, 500), 5, (0, 255, 255), -1)#yellow
    # cv2.circle(img, (583, 280), 5, (255, 0, 255), -1)#pink
    # cv2.circle(img, (723, 280), 5, (255, 0, 0), -1)#blue

    # img = cv2.imread(r"Problem 2\data_1\data\0000000112.png")

    src = np.float32([[480,320 ],[780,320],[120, 505],[1000, 505]])
    dst = np.float32([[0,0],[700, 0],[0, 512],[700, 512]])
    matrix = cv2.getPerspectiveTransform(src,dst)
    warped = cv2.warpPerspective(img,matrix,(700,512))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cv2.imshow("warp", warped_gray)

    # cv2.imshow("warped:",warped_gray)

    ret,warped_gray_thresh = cv2.threshold(warped_gray, 200,255,cv2.THRESH_BINARY)
    canny_edge = cv2.Canny(warped_gray_thresh, 100, 300 ,13, L2gradient=True)

    # cv2.imshow("warp:  img no:%d"%(img_count), warped_gray_thresh)

    #####################################################################################################################

    out_img, curves, lanes, plot_y, m_left = lane_Detection(warped_gray_thresh)

    curve_rad = get_curve(img, curves[0], curves[1])
    lane_curve = np.mean([curve_rad[0], curve_rad[1]])  # ..................

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_fill = draw_lanes(img, curves[0], curves[1])

    # Plot the Lane lines
    for i in range(len(curves[0])):
        cv2.circle(out_img, (int(curves[0][i]), int(plot_y[i])), 5, (255, 100, 255), -1) # Left Lane  Pink
        cv2.circle(out_img, (int(curves[1][i]), int(plot_y[i])), 5, (0, 0, 255), -1) # Right Lane Red

    # Find inverse warp perspective for lane plane
    matrix_inv_img_fill = cv2.getPerspectiveTransform(dst, src)
    warped_inv_img_fill = cv2.warpPerspective(img_fill, matrix_inv_img_fill, (1392, 512))

    # cv2.imshow("sliding img no:%d"%(img_count),out_img)

    # Find inverse warp perspective for lane lines
    matrix_inv = cv2.getPerspectiveTransform(dst, src)
    warped_inv = cv2.warpPerspective(out_img, matrix_inv, (1392, 512))

    # cv2.imshow("inv-warp",warped_inv)
    # Following block adds lines to original image.
    warped_inv_bin = cv2.cvtColor(warped_inv, cv2.COLOR_BGR2GRAY)
    ret, warped_inv_bin_thresh = cv2.threshold(warped_inv_bin, 50, 255, cv2.THRESH_BINARY)
    test = cv2.bitwise_and(img,img,mask = cv2.bitwise_not(warped_inv_bin_thresh))
    # cv2.imshow("trial", test)
    output_image = (cv2.add(cv2.add(test, warped_inv), warped_inv_img_fill))

    text = draw_data(m_left)

    cv2.putText(output_image, 'Lane Curvature: {:.0f} m'.format(lane_curve), (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 3)
    cv2.putText(output_image, 'Vehicle offset: {:.4f} m'.format(curve_rad[2]), (100, 200), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 3)
    cv2.putText(output_image, text, (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imshow("final", output_image)
    out.write(output_image)
    ######################################################################################################################


    if cv2.waitKey(1) == 27:
        # cv2.destroyAllWindows()
        break
cv2.destroyAllWindows()