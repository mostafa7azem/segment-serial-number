from imutils import contours
import numpy as np
import imutils
import cv2

image = cv2.imread("10banknote.jpg")
image = imutils.resize(image, width=700)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
locs = []

for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    if x and y and w and h:
        if 3.7 < ar < 9:
            if 90 < w < 150 and 10 < h < 50:
                locs.append((x, y, w, h))

# sort the digit locations from left-to-right, then initialize the
# list of classified digits
locs = sorted(locs, key=lambda x: x[0])

for (i, (gX, gY, gW, gH)) in enumerate(locs):

    # extract the group ROI of 4 digits from the grayscale image,
    # then apply thresholding to segment the digits from the
    # background of the credit card
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # detect the contours of each individual digit in the group,
    # then sort the digit contours from left to right
    digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = digitCnts[0] if imutils.is_cv2() else digitCnts[1]
    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

    for c in digitCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = cv2.resize(group[y:y + h, x:x + w], (57, 88))

    #cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
    #serial postion
    x=(gX - 5, gY - 5)
    y=(gX + gW + 5, gY + gH + 5)
    print(x[0],x[1],y[0],y[1])
crop_serial= image[x[1]:y[1],x[0]+40:y[0]]
crop_preserial =image[x[1]:y[1],x[0]:y[0]-90]
crop_no=[]
for i in range(9):
        z=x[0] + 40 +i*10
        v=y[0] - 80+i*10
        crop_no.append(image[x[1]+5:y[1]-5, z:v])

del crop_no[2],crop_no[5]
cv2.imshow("Image", image)
cv2.imshow("Image2", crop_serial)
cv2.imshow("Image3", crop_preserial)
k=4
for no_image in crop_no :
    cv2.imshow("Image"+str(k), no_image)
    k=k+1
cv2.waitKey()
cv2.destroyAllWindows()
