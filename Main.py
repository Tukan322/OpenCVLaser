import cv2
import numpy as np
import math


def loading_colour(num):
    return cv2.imread("lines/w/" + str(num) + ".jfif")


def split_boxes(boxxx, w):
    ret_box = []
    n_w = 15
    blank = 7
    y_u = boxxx[0][1]
    y_d = boxxx[2][1]
    l_up = boxxx[0][0]
    l_down = boxxx[3][0]
    r_up = boxxx[0][0] + n_w
    r_down = boxxx[3][0] + n_w
    if w < 15:
        ret_box.append([[boxxx[0][0], boxxx[0][1]],
                        [boxxx[1][0], boxxx[1][1]],
                        [boxxx[2][0], boxxx[2][1]],
                        [boxxx[3][0], boxxx[3][1]]])
        w = 0
    while w > 0:
        ret_box.append([[l_up, y_u],
                        [r_up, y_u],
                        [r_down, y_d],
                        [l_down, y_d]])
        w -= 25
        l_up += n_w + blank
        r_up += n_w + blank
        r_down += n_w + blank
        l_down += n_w + blank

    return ret_box


def loading_grey(num):
    return cv2.imread("lines/" + str(num) + ".jfif")


def loading_sample(num):
    return cv2.imread("lines/sample/grade" + str(num) + ".jpeg")


def display(img):
    cv2.imshow("line", img)
    cv2.waitKey(0)


def display2(img1, img2):
    cv2.imshow("line1", img1)
    cv2.imshow("line2", img2)
    cv2.waitKey(0)


def box_equals(a, b):
    if a[0][0] == b[0][0]:
        return True
    if a[0][1] == b[0][1]:
        return True
    if a[1][0] == b[1][0]:
        return True
    if a[1][1] == b[1][1]:
        return True
    if a[2][0] == b[2][0]:
        return True
    if a[2][1] == b[2][1]:
        return True
    if a[3][0] == b[3][0]:
        return True
    if a[3][1] == b[3][1]:
        return True
    return False


def box_normalize(bx):
    bx.sort(key=lambda x: x[0])
    s0 = bx[0].copy()
    s1 = bx[1].copy()
    s2 = bx[2].copy()
    s3 = bx[3].copy()
    boxx = []
    # len s0, s1 > len s0, s2
    if (math.sqrt(pow(s0[0] - s1[0], 2) + pow(s0[1]-s1[1],2))
            > math.sqrt(pow(s0[0] - s2[0],2) + pow(s0[1] - s2[1], 2))):
        if s0[1] > s1[1]:
            boxx.append(s1)
            boxx.append(s3)
            boxx.append(s2)
            boxx.append(s0)
        if s0[1] <= s1[1]:
            boxx.append(s0)
            boxx.append(s2)
            boxx.append(s3)
            boxx.append(s1)
    else:
        if s0[1] > s3[1]:
            boxx.append(s2)
            boxx.append(s3)
            boxx.append(s1)
            boxx.append(s0)
        else:
            boxx.append(s0)
            boxx.append(s1)
            boxx.append(s3)
            boxx.append(s2)
    return boxx


def rotate(im, angle):
    (h, w) = im.shape[:2]
    center = (int(w/2), int(h/2))
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(im, rot_mat, (w, h))
    return rotated


def angles_calc(bx):
    if abs(bx[0][0] - bx[1][0]) > abs(bx[0][1] - bx[3][1]):
        tga = (bx[1][0] - bx[0][0])/(bx[1][1] - bx[0][1] + 0.000000000001)
    else:
        tga = (bx[1][1] - bx[0][1])/(bx[1][0]-bx[0][0] + 0.000000000001)
    a = math.atan(tga)/math.pi * 180
    if abs(a) > 45:
        return 90
    else:
        return 0

def kp_inside_box(keypoint, boxx, flag):
    if flag:
        if boxx[1][0] >= boxx[0][0]:
            if boxx[0][0] <= keypoint[0] <= boxx[2][0]:
                return True
        if boxx[1][0] < boxx[0][0]:
            if boxx[1][0] <= keypoint[0] <= boxx[3][0]:
                return True
    if not flag:
        if boxx[3][0] >= boxx[0][0]:
            if boxx[1][0] <= keypoint[0] <= boxx[3][0]:
                return True
        if boxx[3][0] >= boxx[0][0]:
            if boxx[3][0] <= keypoint[0] <= boxx[1][0]:
                return True

    return False


# loading image

main_image = loading_colour(27)

# preprocess
# blue, green, red to gray
r_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
# blurring and thresholding image
r_median_blur = cv2.medianBlur(r_image, 7)
r_t, r_blur = cv2.threshold(r_median_blur, 65, 255, cv2.THRESH_BINARY_INV)
# finding canny edges and contours
r_canny = cv2.Canny(r_blur, 125, 175)
r_cont, r_hie = cv2.findContours(r_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# rotation calcs
not_rot_img = main_image.copy()
angles = []
for cnt in r_cont:
    rect = cv2.minAreaRect(cnt)
    area = int(rect[1][0] * rect[1][1])
    nor_box = cv2.boxPoints(rect)
    nor_box = np.intp(nor_box)
    box = []
    for bx in nor_box:
        box.append(bx)
    box = box_normalize(box)
    box = np.intp(box)

    if area > 3000:
        angles.append(angles_calc(box))
sum_a = 0
len_a = 0
for an in angles:
    len_a += 1
    sum_a += an
alpha = sum_a / len_a
if alpha > 45:
    alpha = 90
else:
    alpha = 0

main_image = rotate(main_image, alpha)
# no more prep
cv2.rectangle(main_image, (0, 0), (640, 0), (255, 255, 255), 30)
cv2.rectangle(main_image, (0, 0), (640, 480), (255, 255, 255), 30)
cv2.rectangle(not_rot_img, (0, 0), (640, 0), (255, 255, 255), 30)
cv2.rectangle(not_rot_img, (0, 0), (640, 480), (255, 255, 255), 30)

image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
main_image_with_many_contours = main_image.copy()

main_image_with_one_contour = main_image.copy()
# blurring image
median_blur = cv2.medianBlur(image, 7)
# thresholding image
t, blur = cv2.threshold(median_blur, 75, 255, cv2.THRESH_BINARY_INV)
# finding canny edges and contours
canny = cv2.Canny(blur, 85, 175)
cont, hie = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# detect features
orb = cv2.ORB.create()
kps, des = orb.detectAndCompute(median_blur, None)
image_with_kp = cv2.drawKeypoints(image, kps, None)
# var
max_y = 0
min_y = 1000
boxes = []

# finding max_y, max_y
for cnt in cont:
    flag = False
    rect = cv2.minAreaRect(cnt)
    nor_box = cv2.boxPoints(rect)
    nor_box = np.intp(nor_box)
    area = int(rect[1][0] * rect[1][1])
    box = []
    for bx in nor_box:
        box.append(bx)
    box = box_normalize(box)
    box = np.intp(box)
    y_up = box[2][1]
    y_up2 = box[3][1]
    y_down = box[1][1]
    y_down2 = box[0][1]
    for bx in box:
        if bx[0] < 5 or bx[0] > 635:
            flag = True
    # find max and min
    if area > 3000 and not flag:
        if y_up > max_y:
            max_y = y_up
        if y_up2 > max_y:
            max_y = y_up
        if y_down < min_y:
            min_y = y_down
        if y_down2 < min_y:
            min_y = y_down2


for cnt in cont:
    flag = False
    rect = cv2.minAreaRect(cnt)
    nor_box = cv2.boxPoints(rect)
    nor_box = np.intp(nor_box)
    area = int(rect[1][0] * rect[1][1])

    box_ns = []
    for bx in nor_box:
        box_ns.append(bx)
    box_ns = box_normalize(box_ns)
    box_ns = np.intp(box_ns)
    box_ns[3][1] = max_y
    box_ns[2][1] = max_y
    box_ns[1][1] = min_y
    box_ns[0][1] = min_y
    width1 = math.sqrt(pow(box_ns[0][0] - box_ns[1][0], 2)
                       + pow(box_ns[0][1] - box_ns[1][1], 2))
    width2 = math.sqrt(pow(box_ns[2][0] - box_ns[3][0], 2)
                       + pow(box_ns[2][1] - box_ns[3][1], 2))
    width = (width2 + width1) / 2
    box_s = []
    if box_ns[0][0] > 35 and box_ns[1][0] < 605:
        box_s = split_boxes(box_ns, width)
    for box in box_s:
        box = np.intp(box)
        if width > 10:

            # intersection detect
            # check pending box to be in existing box
            # upper
            for b in boxes:
                if b[1][0] - b[0][0] >= abs((b[0][0] + b[1][0])/2 - box[0][0])*2:
                    if not box[3][0] - b[2][0] > 10:
                        flag = True
            # lower
                if b[2][0] - b[3][0] >= abs((b[3][0] + b[2][0])/2 - box[3][0])*2:
                    if not box[0][0] - b[1][0] > 10:
                        flag = True

            # check pending box to contain existing box
            for b in boxes:
                # upper
                if box[1][0] - box[0][0] >= abs((box[0][0] + box[1][0])/2 - b[0][0])*2:
                    if not b[3][0] - box[2][0] > 10:
                        flag = True
                # lower
                if box[2][0] - box[3][0] >= abs((box[3][0] + box[2][0])/2 - b[3][0])*2:
                    if not b[0][0] - box[1][0] > 10:
                        flag = True
            # not on the edge
            for bx in box:
                if bx[0] < 31 or bx[0] > 640 - 31:
                    flag = True

            if not flag:
                boxes.append(box)

boxes.sort(key=lambda x: x[0][0])
for bx in boxes:
    cv2.drawContours(main_image_with_many_contours,
                     [bx], 0, (255, 0, 0), 2)

# matching features in contours
feature_number = []


for bx in boxes:
    fl = True
    if bx[0][1] > bx[1][1]:
        fl = False
    feature_counter = 0
    for kp in kps:

        if kp_inside_box(kp.pt, bx, fl):
            feature_counter += 1
    feature_number.append((bx, feature_counter))


feature_number.sort(key=lambda x: x[0][0][0])
feature_len = 0
for ftr in feature_number:
    feature_len += 1


final_box = []
if feature_len % 2 == 0:
    if feature_number[0][1] > feature_number[-1][1]:
        final_box = feature_number[int(feature_len/2)-1]
    else:
        final_box = feature_number[int(feature_len/2)]
else:
    final_box = feature_number[int(feature_len/2)]

main_image_with_one_contour = cv2.drawContours(main_image_with_one_contour,
                                               final_box, 0,
                                               (0,255,0), 2)
main_image_with_kp = cv2.drawKeypoints(main_image_with_many_contours, kps, None)
display2(not_rot_img, main_image_with_one_contour)
final_x_coords = (final_box[0][0][0], final_box[0][2][0])
print("focusing line lies between " + str(final_x_coords))
