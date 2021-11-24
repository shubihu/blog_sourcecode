---
title: Python 监控
date: 2021-09-02 13:55:36
index_img: /img/article/monitor.png
categories:
    - Python
tags:
    - 玩
comment: 'valine'
---
## 使用python监控电脑键盘、鼠标并拍照录像
<!-- more -->
```
import keyboard
from cv2 import cv2
# from pynput.mouse import Listener
import pyautogui as pag    #监听鼠标
# from pynput.keyboard import Key, Listener
from threading import Thread

x1, y1 = pag.position()
# print(x1, y1)

def camera():
    '''
    拍照
    '''
    cap = cv2.VideoCapture(0)
    ret,frame = cap.read() #读取摄像头内容
    cv2.imwrite("./test.jpg",frame) #保存到磁盘
    #释放摄像头
    cap.release()

def record_video():
    '''
    录制视频
    '''
    cap = cv2.VideoCapture(0)
    fps = 30
    size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter=cv2.VideoWriter('./test.avi',cv2.VideoWriter_fourcc('X','V','I','D'),fps,size)
    success,frame = cap.read()
    numFrameRemaining = 5 * fps    #摄像头捕获持续时间
    while success and numFrameRemaining > 0:
        videoWriter.write(frame)
        success,frame = cap.read()
        numFrameRemaining -= 1

    cap.release()

def display_video():
    '''
    实时窗口
    '''
    face_locations = []
    cap = cv2.VideoCapture(0)

    while True:
        # Grab a single frame of video
        ret, frame = cap.read()

        # Convert the image from BGR color (whichOpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces in the current frameof video
        face_locations = face_recognition.face_locations(rgb_frame)

        # Display the results
        for top, right, bottom, left in face_locations:
            # Draw a box around the face
            cv2.rectangle(frame, (left, top),(right, bottom), (0, 0, 255), 2)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle tothe webcam
    cap.release()
    cv2.destroyAllWindows()

def display_video2():
    '''
    实时检测
    '''
    #存储知道人名列表
    known_names=['yahaha1', 'yahaha2'] 
    #存储知道的特征值
    known_faces=[]

    image1 =face_recognition.load_image_file("yahaha2.jpg")
    face_encoding1 =face_recognition.face_encodings(image1)

    image2 =face_recognition.load_image_file("yahaha1.jpg")
    face_encoding2 =face_recognition.face_encodings(image1)

    if face_encoding1 and face_encoding2:
        face_encoding1 = face_encoding1[0]
        face_encoding2 = face_encoding2[0]
    else:
        sys.exit()

    known_faces = [face_encoding1, face_encoding2]

    cap = cv2.VideoCapture(0)

    while True:
        # Grab a single frame of video
        ret, frame = cap.read()
        # Convert the image from BGR color (whichOpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings inthe current frame of video
        face_locations =face_recognition.face_locations(rgb_frame)  # 如有gpu可添加参数model='cnn'提升精度
        face_encodings =face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for theknown face(s)
            matches =face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.60)

            name = None
            # if match[0]:
            #     name = "Yahaha"
            print(matches)
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]
            else:
                name = 'Unkonwn'

            face_names.append(name)

        # Label the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if not name:
                continue

            # Draw a box around the face
            cv2.rectangle(frame, (left, top),(right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below theface
            cv2.rectangle(frame, (left, bottom -25), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6,bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # All done!
    cap.release()
    cv2.destroyAllWindows()

def proof(x):
    # print(x)
    # record_video()
    camera()

def monitor_keyboard():
    keyboard.hook(proof)
    #按下任何按键时，都会调用proof，其中一定会传一个值，就是键盘事件
    keyboard.wait()


def monitor_mouse():
    x2, y2 = pag.position()
    while x1 == x2:
        x2, y2 = pag.position()
    else:
        # record_video()
        camera()

if __name__ == '__main__':
    k = Thread(target=monitor_keyboard, args=())
    m = Thread(target=monitor_mouse, args=())
    k.start()
    m.start()
    k.join()
    m.join()

```

