import math
import time
import cv2
import numpy as np
from copy import deepcopy
from model.yolo_model import YOLO

class Point2D:
    def __init__(self, width, height):
        self.width = width
        self.height = height

def process_image(img):
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image


def get_classes(file):
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names


if __name__ == '__main__':
    # 파일 열기
    camera = cv2.VideoCapture("input/example.mp4")
    if camera.isOpened() == False:
        print("Can't open Video.")

    # Yolo 학습
    yolo = YOLO(0.6, 0.5)
    file = 'data/coco_classes.txt'
    all_classes = get_classes(file)

    # 10 카운트 할 때마다 frame 얻어서 파일로 저장
    success, image = camera.read()
    count = 0
    frameCount = 0
    start_time = time.time()
    while success:
        frameCount += 1
        if frameCount % 3 == 0:
            cv2.imwrite("mid/frame%d.png" % count, image)     # save frame as JPEG file
            count += 1
        success, image = camera.read()

    camera.release()

    # 각 프레임 별로 Image Detection 후 프레임 번호, 객체 이름(name)과 객체의 크기(size), 객체가 얼마나 가운데 있는지(coordinatevalue) 저장
    detectionInfo = []
    for i in range(count):
        #filename = "mid/frame"+str(i)+"."
        image = cv2.imread("mid/frame%d.png" % i)
        pimage = process_image(image)
        boxes, classes, scores = yolo.predict(pimage, image.shape)
        for box, score, cl in zip(boxes, scores, classes):
            x, y, w, h = box
            name = all_classes[cl]
            size = int(w*h)

            if size <= 4000: # 사이즈가 너무 작아 썸네일로 적합하지 않은 경우
                continue
            if x <= 0 or x+w >= image.shape[1] or y <= 0 or y+h >= image.shape[0]: # 검출된 객체가 프레임 밖으로 나간 경우
                continue
            # 얼마나 가운데인지 확인하는 알고리즘
            object = Point2D(width= x + w/2, height= y + h/2)
            a = image.shape[1]/2 - object.width
            b = image.shape[0]/2 - object.height
            coordinatevalue = int(math.sqrt((a*a)+(b*b)))

            top = max(0, np.floor(x + 0.5).astype(int))
            left = max(0, np.floor(y + 0.5).astype(int))
            right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
            bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

            # 객체 정보 및 계산 값 저장
            detectionInfo.append([i, name, size, coordinatevalue, top, left, right, bottom])

    # 검출 리스트 txt파일로 내보내기
    f = open("detectionInfo.txt", 'w')
    for i in range(len(detectionInfo)):
        data = str(detectionInfo[i][0]) +", " + detectionInfo[i][1] + ", " + str(detectionInfo[i][2]) + ", " + str(detectionInfo[i][3]) + ", " + str(detectionInfo[i][4]) + ", " + str(detectionInfo[i][5]) + ", " + str(detectionInfo[i][6]) + ", " + str(detectionInfo[i][7]) + "\n"
        f.write(data)
    f.close()

    # 크롭할 이미지의 이름을 저장하는 딕셔너리
    cropDict = {}

    # 검출된 물체 리스트(중복 없이)
    namelist = {}

    for i in range(len(detectionInfo)):
        if not detectionInfo[i][1] in namelist:
            namelist[detectionInfo[i][1]] = []

    # 크기
    for objectName in namelist.keys():
        maxindex = 0
        maxvalue = 0
        for j in range(len(detectionInfo)):
            if detectionInfo[j][1] == objectName:
                if detectionInfo[j][2] > maxvalue:
                    maxvalue = detectionInfo[j][2]
                    maxindex = detectionInfo[j][0]
        namelist[objectName].append(maxindex)

    cropDict[1] = []
    for objectname, framelist in namelist.items():
        image = cv2.imread("mid/frame%d.png" % framelist[0])
        output1 = cv2.GaussianBlur(image, (5,5), 0)
        cv2.imwrite("output1/%s.png"% (objectname), output1)
        cropDict[1].append((objectname, detectionInfo[framelist[0]][4],detectionInfo[framelist[0]][5],
                            detectionInfo[framelist[0]][6], detectionInfo[framelist[0]][7]))

    # 가운데 위치
    for objectName in namelist.keys():
        namelist[objectName] = []

    for objectName in namelist.keys():
        minindex = 0
        minvalue = 999999
        for j in range(len(detectionInfo)):
            if detectionInfo[j][1] == objectName:
                if detectionInfo[j][3] < minvalue:
                    minvalue = detectionInfo[j][3]
                    minindex = detectionInfo[j][0]
        namelist[objectName].append(minindex)

    cropDict[2] = []
    for objectname, framelist in namelist.items():
        image = cv2.imread("mid/frame%d.png" % framelist[0])
        output2 = cv2.GaussianBlur(image, (5,5), 0)
        cv2.imwrite("output2/%s.png"% (objectname), output2)
        cropDict[2].append((objectname, detectionInfo[framelist[0]][4], detectionInfo[framelist[0]][5],
                            detectionInfo[framelist[0]][6], detectionInfo[framelist[0]][7]))

    # 계획2 : 프레임별로 나온 객체 겹치는 부분 제외하고 넓이 구해 큰거 Indexlist에 넣기
    # 모든 프레임에 적용하지 않고 여러 객체가 나온 프레임 선정, 프레임 인덱스 저장하는 best 딕셔너리
    best = list(list(zip(*detectionInfo))[0])
    bestList= {}
    for i in range(len(detectionInfo)):
        if best.count(detectionInfo[i][0]) == 2:
            if detectionInfo[i][0] in bestList:
                bestList[detectionInfo[i][0]].append(detectionInfo[i][1:]) # 이름, 사이즈, 중심과의 거리, x, y, w, h 저장
            else:
                bestList[detectionInfo[i][0]] = [detectionInfo[i][1:]]
        elif best.count(best[i]) > 2:
            if best[i] in bestList:
                bestList[best[i]].append([i, detectionInfo[i][3]])
            else:
                bestList[best[i]] = [[i, detectionInfo[i][3]]]

    # 프레임에 등장하는 객체가 2개 이상인 경우, 가장 적합한 객체 두 개 선정
    for key, value in bestList.items():
        if len(value[0]) == 2:
            tmpValue = deepcopy(value)
            first, second = 0, 0
            indexList = list(list(zip(*tmpValue))[0])
            coordiList = list(list(zip(*tmpValue))[1])
            minCordi = coordiList.index(min(coordiList))
            first = indexList[minCordi]
            coordiList[minCordi] = 99999
            minCordi = coordiList.index(min(coordiList))
            second = indexList[minCordi]
            bestList[key] = [detectionInfo[first][1:], detectionInfo[second][1:]]

    # beOverlap 에 선정된 두 객체의 top, left, right, bottom 값 비교하여 두 객체의 합산 size 계산, 이를 overlapped에 저장
    # 이 후로, value는 항상 두 가지 값만을 가짐
    deleteList = []
    for key, value in bestList.items():
        a_top, a_left, a_right, a_bottom = value[0][3:]
        b_top, b_left, b_right, b_bottom = value[1][3:]
        o_top, o_left, o_right, o_bottom = 0, 0, 0, 0

        # 두 객체가 겹쳤는지 좌표 비교를 통해 확인, 겹쳤다면 겹친 부분의 좌표 값 o_xxx 에 저장
        isOverlapped = False
        if a_top < b_top < a_bottom or b_top < a_top < b_bottom \
                or a_left < b_left < a_right or b_left < a_left < b_right:
            isOverlapped = True
            o_top = max(a_top, b_top)
            o_bottom = min(a_bottom, b_bottom)
            o_left = max(a_left, b_left)
            o_right = min(a_right, b_right)

        # 겹친 부분의 넓이 구하기
        o_size = (o_bottom-o_top)*(o_right-o_left)
        # 겹친 객체를 하나로 보고, 두 객체의 중심점을 찾아 프레임의 중앙과의 거리 구하기
        if isOverlapped == True:
            a_object = Point2D(width= (a_top + a_bottom)//2, height=(a_left + a_right)//2)
            b_object = Point2D(width= (b_top + b_bottom)//2, height=(b_left + b_right)//2)
            o_object = Point2D(width=(a_object.width+b_object.width)//2 , \
                               height = (a_object.height + b_object.height)//2)
            toTheOrigin_w = image.shape[1] / 2 - o_object.width
            toTheOrigin_h = image.shape[0] / 2 - o_object.height
            coordiValue_object = int(math.sqrt((toTheOrigin_w ** 2) + (toTheOrigin_h ** 2)))
            bestList[key] = [value[0][0] + ", "+ value[1][0], value[0][1]+value[1][1]-o_size, \
                             coordiValue_object, min(a_top, b_top), min(a_left, b_left), \
                             max(a_right, b_right), max(a_bottom, b_bottom)]
        else:
            # 겹치지 않았다면 목록에서 삭제
            deleteList.append(key)
    for i in range(len(deleteList)-1, -1, -1):
        del bestList[deleteList[i]]

    namelist.clear()

    # 검출된 리스트 중 limitSize를 넘으며, 가장 중앙에 가까운 객체 선출
    limitSize = 100000
    for key, value in bestList.items():
        if value[1] > limitSize:
            if value[0] in namelist:
                if value[2] < namelist[value[0]][2]:
                    namelist[value[0]] = [key] + value[1:]
            else:
                namelist[value[0]] = [key] + value[1:]


    # output3에 출력
    cropDict[3] = []
    for objectname, framelist in namelist.items():
        image = cv2.imread("mid/frame%d.png" % framelist[0])
        output3 = cv2.GaussianBlur(image, (5, 5), 0)
        cv2.imwrite("output3/%s.png" % (objectname), output3)
        cropDict[3].append((objectname, detectionInfo[framelist[0]][4],detectionInfo[framelist[0]][5],
                            detectionInfo[framelist[0]][6], detectionInfo[framelist[0]][7]))

    # 계획3 : 객체가 특정 위치에 있는 프레임 뽑기
    # output1~3의 결과들을 가지고 특정 위치에 있게 이미지 크롭, 결과를 output1_1, output2_1, output3_1 에 저장
    for outputNum, frameList in cropDict.items():
        for i in range(len(frameList)):
            image = cv2.imread("output%d/%s.png" % (outputNum, frameList[i][0]))
            crop_top = 0
            crop_bottom = image.shape[0]
            crop_left = 0
            crop_right = image.shape[1]
            output = image[crop_top:crop_bottom, crop_left:crop_right].copy()
            # 읽어온 이미지의 비가 거의 16:9이면, 입력받은 이미지 그대로 추천
            if image.shape[0]//9 * 15 < image.shape[1] <= image.shape[0]//9 * 16:
                cropDict[outputNum][i] = (frameList[i][0], 0, image.shape[0], 0, image.shape[1])
            else:
                # 읽어온 이미지의 비가 16:9가 아니라면, 검출 객체의 높이를 가져와서 이에 맞는 16비의 너비 구함
                # frameList = (프레임이름, top, left, right, bottom)
                crop_top = int(frameList[i][1] + (frameList[i][4] - frameList[i][1]) * 0.15)
                crop_bottom = frameList[i][4]
                height = crop_bottom - crop_top
                width = height // 9 * 16
                crop_left = frameList[i][2]
                crop_right = frameList[i][3]

                # 원하는 가로길이가 원본프레임의 가로길이보다 큰 경우
                if width > image.shape[1]:
                    padNum = width - image.shape[1]
                    output = image[crop_top:crop_bottom, 0:image.shape[1]].copy()
                    ratio = int((image.shape[1]+padNum)/image.shape[1])
                    output = cv2.resize(output, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
                # 물체가 프레임의 중앙에 있을 경우
                elif image.shape[1]//2 - image.shape[1]//20 < (crop_left + crop_right)// 2 \
                        <= image.shape[1]//2 + image.shape[1]//20:
                    crop_left = (crop_left + crop_right)//2 - width//2
                    crop_right = (crop_left + crop_right)//2 + width//2
                    output = image[crop_top:crop_bottom, crop_left:crop_right].copy()
                # 물체가 프레임의 왼쪽에 있을 경우
                elif (crop_left + crop_right)//2 < image.shape[1]//2:
                    crop_left = 0
                    crop_right = width
                    output = image[crop_top:crop_bottom, crop_left:crop_right].copy()
                elif (crop_left + crop_right)//2 > image.shape[1]//2:
                    crop_left = image.shape[1]-width
                    crop_right = image.shape[1]
                    output = image[crop_top:crop_bottom, crop_left:crop_right].copy()

            output = cv2.GaussianBlur(output, (7, 7), 0)
            kernel_sharpen = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
            output = cv2.filter2D(output,-1, kernel_sharpen)
            cv2.imwrite("output%d_crop/%s.png" % (outputNum, frameList[i][0]), output)