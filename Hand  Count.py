import cv2 as cv
import numpy as np
import math


def nothing(x):
    pass


cap = cv.VideoCapture(0, cv.CAP_DSHOW)

cv.namedWindow("Color Adjustments")
cv.resizeWindow("Color Adjustments", (300, 300))

cv.createTrackbar("Thresh", "Color Adjustments", 0, 255, nothing)

cv.createTrackbar("Lower_H:", "Color Adjustments", 0, 255, nothing)
cv.createTrackbar("Lower_S:", "Color Adjustments", 0, 255, nothing)
cv.createTrackbar("Lower_V:", "Color Adjustments", 0, 255, nothing)

cv.createTrackbar("Upper_H:", "Color Adjustments", 255, 255, nothing)
cv.createTrackbar("Upper_S:", "Color Adjustments", 255, 255, nothing)
cv.createTrackbar("Upper_V:", "Color Adjustments", 255, 255, nothing)

while True:
    bol, image = cap.read()
    if bol == True:
        image = cv.resize(image, (400, 600))
        # image = cv.medianBlur(image, 3)
        crop_image = image[1:600, 0:400]

        # Convert BGR to HSV
        hsv = cv.cvtColor(crop_image, cv.COLOR_BGR2HSV)

        l_h = cv.getTrackbarPos("Lower_H:", "Color Adjustments")
        l_s = cv.getTrackbarPos("Lower_S:", "Color Adjustments")
        l_v = cv.getTrackbarPos("Lower_V:", "Color Adjustments")

        u_h = cv.getTrackbarPos("Upper_H:", "Color Adjustments")
        u_s = cv.getTrackbarPos("Upper_S:", "Color Adjustments")
        u_v = cv.getTrackbarPos("Upper_V:", "Color Adjustments")

        lower_b = np.array([l_h, l_s, l_v])
        upper_b = np.array([u_h, u_s, u_v])

        mask = cv.inRange(hsv, lower_b, upper_b)
        filtr = cv.bitwise_and(crop_image, crop_image, mask=mask)
        mask = cv.medianBlur(mask, 3)

        mask1 = cv.bitwise_not(mask)
        m_g = cv.getTrackbarPos("Thresh", "Color Adjustments")
        ret, thrash = cv.threshold(mask1, m_g, 255, cv.THRESH_BINARY)
        dilate = cv.dilate(thrash, (3, 3), iterations=6)

        # Finding contuers...
        cnts, hier = cv.findContours(thrash, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2:]

        try:
            cm = max(cnts, key=lambda x: cv.contourArea(x))
            epsilon = 0.0005 * cv.arcLength(cm, True)
            data = cv.approxPolyDP(cm, epsilon, True)
            hull = cv.convexHull(cm)

            cv.drawContours(crop_image, [cm], -1, (0, 0, 0), 2)
            cv.drawContours(crop_image, [hull], -1, (0, 255, 0), 2)

            # Find convexity defects...
            hull = cv.convexHull(cm, returnPoints=False)
            defects = cv.convexityDefects(cm, hull)
            count_defects = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(cm[s][0])
                end = tuple(cm[e][0])
                far = tuple(cm[f][0])

                # Cosin Rule
                a = math.sqrt((end[0] - start[0]) * 2 + (end[1] - start[1]) * 2)
                b = math.sqrt((far[0] - start[0]) * 2 + (far[1] - start[1]) * 2)
                c = math.sqrt((end[0] - far[0]) * 2 + (end[1] - far[1]) * 2)
                angle = (math.acos((b * 2 + c * 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

                if angle <= 60:
                    count_defects += 1
                    cv.circle(crop_image, far, 5, (255, 255, 255), -1)

            print("Count== ", count_defects)

            ### Printing number of fingers...
            if count_defects == 0:
                cv.putText(crop_image, "1", (50, 50), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 2)
            elif count_defects == 1:
                cv.putText(crop_image, "2", (50, 50), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 2)
            elif count_defects == 2:
                cv.putText(crop_image, "3", (50, 50), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 2)
            elif count_defects == 3:
                cv.putText(crop_image, "4", (50, 50), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 2)
            elif count_defects == 4:
                cv.putText(crop_image, "5", (50, 50), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 2)
            else:
                pass
        except:
            pass

        image[1:600, 0:400] = crop_image
        # cv.imshow("Thresh", thrash)
        cv.imshow("mask==", mask)
        cv.imshow("filter==", filtr)
        cv.imshow("Result", image)

        key = cv.waitKey(25) & 0xFF
        if key == 27:
            break
    else:
        break

cap.release()
cv.destroyAllWindows()