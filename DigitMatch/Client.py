import cv2
"""
数字匹配 效果不好
"""


def get_target_cnt():
    file_path = '../DigitMatch/test/3-2.png'
    img = cv2.imread(file_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, 1, (255, 0, 0), 3)
    # cv2.imshow('', img)
    return contours[1]


def digit_cnt_set():
    cnt_src = []
    for i in range(10):
        file_path = '../DigitMatch/src/' + str(i) + '.png'
        img = cv2.imread(file_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img, contours[1], -1, (0, 255, 0), 3)
        # cv2.imshow(str(i), img)
        cnt_src.append(contours[1])
    return cnt_src


def figure_match(cnt_target, cnt_src):
    min_val = 100
    min_index = -1
    for i in range(10):
        match = cv2.matchShapes(cnt_target, cnt_src[i], cv2.CONTOURS_MATCH_I1, 0.0)
        # print(str(match))
        min_val = min_val if match > min_val else match
        min_index = min_index if match > min_val else i
    return min_index


target = get_target_cnt()
cnt_set = digit_cnt_set()
# print(str(cv2.matchShapes(target, target, cv2.CONTOURS_MATCH_I1, 0.0)))
print(str(figure_match(target, cnt_set)))
cv2.waitKey(0)
cv2.destroyAllWindows()
