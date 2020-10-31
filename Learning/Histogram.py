import cv2

img = cv2.imread('../src/desktop.jpg')

cv2.imshow('', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
