import cv2
import dlib
import numpy as np

def get_parts(image, landmarks):
    # 根据特征点坐标获取眼睛、鼻子和嘴巴的区域
    left_eye = image[landmarks.part(37).y:landmarks.part(41).y, landmarks.part(36).x:landmarks.part(39).x]
    right_eye = image[landmarks.part(44).y:landmarks.part(46).y, landmarks.part(42).x:landmarks.part(45).x]
    nose = image[landmarks.part(28).y:landmarks.part(33).y, landmarks.part(31).x:landmarks.part(35).x]
    mouth = image[landmarks.part(51).y:landmarks.part(57).y, landmarks.part(48).x:landmarks.part(54).x]
    
    return left_eye, right_eye, nose, mouth

# 加载Dlib的人脸检测器和人脸特征点检测器
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
capture = cv2.VideoCapture(0)
# 读取图像
if capture.isOpened()is False:
    print("Error opening the camera")
while capture.isOpened():
    ret, frame = capture.read()
    if ret is True:
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray_image)
        for rect in faces:
            landmarks = landmark_predictor(gray_image, rect)
    
            # 获取人脸部位
            left_eye, right_eye, nose, mouth = get_parts(frame, landmarks)

            # 显示人脸部位
            cv2.imshow("Left Eye", left_eye)
            cv2.imshow("Right Eye", right_eye)
            cv2.imshow("Nose", nose)
            cv2.imshow("Mouth", mouth)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()