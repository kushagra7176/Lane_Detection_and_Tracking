# Importing the required libraries

import numpy as np
import cv2

outputPath = 'C:/Users/nikvi/Downloads'

# Method to perform gamma correction on each frame

def adjust_gamma(image, gamma=1.0):
    ''' building a lookup table mapping the pixel values [0, 255] to
     their adjusted gamma values '''
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# Importing the video file

cap = cv2.VideoCapture('Night Drive - 2689.mp4')
if (cap.isOpened() == False):
    print("Error opening video stream or file")

output = cv2.VideoWriter('firstOutput.avi', cv2.VideoWriter_fourcc(*'XVID'), 20, (1280, 720))

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.resize(frame, (700, 500))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Converting to grayscale

        gray = cv2.resize(gray, (500, 300))

        # Instantiating the CLAHE object to apply to the gamma corrected frame

        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(32, 32))

        # Gamma value set to 1.9

        gamma1 = 1.9

        # Gama corrected image

        adjusted_img = adjust_gamma(gray, gamma=gamma1)

        # Performing CLAHE

        clahe_img = clahe.apply(adjusted_img)

        # 3x3 Kernel
        kernel1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img_cl12 = cv2.filter2D(adjusted_img, -1, kernel1)

        # Original Video
        cv2.putText(gray, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 3)

        # Original Video with Gamma Correction
        cv2.putText(adjusted_img, "g = 1.9 Output 1", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 3)

        # Gamma corrected video after CLAHE
        cv2.putText(clahe_img, "Output 2", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 3)

        # Final Video Output after applying 2D filter to the gamma corrected video
        cv2.putText(img_cl12, "Output 3", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 3)
        cv2.imshow("Images", np.vstack([np.hstack([gray, adjusted_img]), np.hstack([clahe_img, img_cl12])]))

        outframe = np.vstack([np.hstack([gray, adjusted_img]), np.hstack([clahe_img, img_cl12])])
        outframe = cv2.cvtColor(outframe,cv2.COLOR_GRAY2BGR)
        outframe = cv2.resize(outframe,(1280,720))

        output.write(outframe)

        if cv2.waitKey(1) == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
