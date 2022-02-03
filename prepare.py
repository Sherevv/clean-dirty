import glob
import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm

RADIUS_MIN = 10
RADIUS_MAX = 140


def crop_image(img, coords):
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    rect = img[y:y + h, x:x + w]
    return rect

def crop_circle(img, radius):
    a=int(radius*np.sqrt(2))
    h, w = img.shape[:2]
    x=(h-a)//2
    rect = img[x:x + a, x:x + a]
    return rect

# def crop_rect(img):
#     a=radius*np.sqrt(2)
#     h, w = img.shape[:2]
#     x=(h-a)//2
#     rect = img[x:x + a, x:x + a]
#     return rect

def filter_contours(img):
    #work_img = img.copy()
    work_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # work_img = cv2.medianBlur(gray, 5)
    canvas = np.zeros(img.shape, np.uint8)

    # threshold = cv2.inRange(work_img, np.array((0, 100, 0)), np.array((255, 250, 255)))

    # _, threshold = cv2.threshold(work_img, 100, 255, 0)
    _, threshold = cv2.threshold(work_img, 10, 255, cv2.THRESH_BINARY)  # cv2.THRESH_BINARY)
    # Find edges
    # edge_detected_image = cv2.Canny(threshold, 75, 200)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_contour = []
    k = 0
    max_c = 0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        # print(approx)

        cnt_area = cv2.contourArea(cnt)

        if len(approx) > 3 and cnt_area > max_area:
            max_area = cnt_area
            max_contour = approx
            max_c = k
            # cv2.waitKey(0)
            #
            # cv2.drawContours(canvas, contours, k, (255, 0, 0), 2, cv2.LINE_AA, hierarchy, 0)
            # cv2.imshow('cont', canvas)

        k += 1

    work_img = cv2.cvtColor(work_img, cv2.COLOR_GRAY2RGB)
    mask = np.zeros(img.shape[:2], np.uint8)
    # mask[:] = 0
    # cv2.fillPoly(mask, contours[max_c], (255,255,255))
    cv2.drawContours(mask, contours, max_c, (255, 255, 255), -1)  # , cv2.LINE_AA, hierarchy, 0)
    # work_img = cv2.bitwise_and(work_img, mask)
    masked = cv2.bitwise_and(img, img, mask=mask)
    # masked[mask == 0] = (255, 255, 255)
    cropped = crop_image(masked, contours[max_c])
    # cv2.drawContours(work_img, contours, max_c, (255, 255, 255), -1, cv2.LINE_AA, hierarchy, 0)
    # cv2.drawContours(work_img, [max_contour], 0, (255, 0, 0), 4)
    # cv2.imshow('cont2', cropped)

    return cropped


def grub_cut(path):
    img = cv2.imread(path)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    img_height, img_width = img.shape[:2]
    rect = (20, 20, img_height-10, img_width-10)
    blug = cv2.blur(img, (7, 7))
    cv2.grabCut(blug, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    # img[mask2 == 0] = (255, 255, 255)
    # img[mask2 == 0] = (0, 0, 0)
    # cv2.imshow('cont23', img)
    img = filter_contours(img)
    # cropped = crop_image(img)

    cv2.imwrite(os.path.join(os.path.dirname(path), os.path.basename(path)), img)


def find_circle(path):
    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows,
                               param1=100, param2=10,
                               minRadius=RADIUS_MIN, maxRadius=RADIUS_MAX)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            img_height, img_width = img.shape[:2]
            tr = 10
            if not (int(center[0]) + radius > img_width + tr
                    or int(center[0]) - radius < -tr
                    or int(center[1]) + radius > img_height + tr
                    or int(center[1]) - radius < -tr):
                #print(1,path)
                #return False
                mask = np.zeros(img.shape[:2], dtype="uint8")
                cv2.circle(mask, center, radius, (255, 255, 255), -1)

                # cv2.imshow('mask', mask)
                masked = cv2.bitwise_and(img, img, mask=mask)
                #masked[mask == 0] = (255, 255, 255)
                coords = cv2.findNonZero(mask)



                cropped = crop_image(masked, coords)
                #cropped = crop_circle(cropped,radius)
                # cv2.imshow('mask', cropped)
                # print('yes')
                # print(os.path.join(os.path.dirname(path),'transform',os.path.basename(path)))
                cv2.imwrite(os.path.join(os.path.dirname(path), os.path.basename(path)), cropped)
                return True

    #print(path)
    return False

    # cv2.imshow('thresh', masked)

    # escape condition
    # cv2.waitKey(0)

    # clean up windows
    # cv2.destroyAllWindows()


def prepare_image(dirs, data_root):
    #dirs = ['train/cleaned', 'train/dirty', 'test']
    for dirr in dirs:
        print(dirr)

        os.makedirs(os.path.join(data_root, dirr), exist_ok=True)
        # files = glob.glob(os.path.join(data_root, dirr, '*'))
        # for f in files:
        #     os.remove(f)

        for filename in os.listdir(os.path.join(data_root, dirr)):
            if filename.endswith(".jpg"):
                if not find_circle(os.path.join(data_root, dirr, filename)):
                    grub_cut(os.path.join(data_root, dirr, filename))

def make_train_val_test(data_root):

    train_dir = 'train'
    val_dir = 'val'
    test_dir = 'test'
    class_names = ['cleaned', 'dirty']

    for dir_name in [train_dir, val_dir]:
        for class_name in class_names:
            os.makedirs(os.path.join(data_root, dir_name, class_name), exist_ok=True)

    for class_name in class_names:
        source_dir = os.path.join(data_root, 'train', class_name)
        for i, file_name in enumerate(tqdm(os.listdir(source_dir))):
            #             if i % 5 != 0:
            #                 dest_dir = os.path.join(train_dir, class_name)
            #             else:
            #                 dest_dir = os.path.join(val_dir, class_name)
            shutil.copy(os.path.join(source_dir, file_name), os.path.join(data_root, train_dir, class_name, file_name))

            shutil.copy(os.path.join(source_dir, file_name), os.path.join(data_root, val_dir, class_name, file_name))
    if not os.path.isdir(os.path.join(data_root, test_dir, 'unknown')):
        shutil.copytree(os.path.join(data_root, 'test'), os.path.join(data_root, test_dir,  'unknown'))
