import sys
import cv2
import argparse
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import time

from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow

from yolo import YOLO
import pyrealsense2 as rs
# modbus
import modbus_tk.modbus_tcp as mt
import modbus_tk.defines as md

class Ui_MainWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.timer_video = QtCore.QTimer()
        self.setupUi(self)
        self.init_logo()
        self.init_slots()
        self.cap = cv2.VideoCapture()
        self.out = None

    def init_slots(self):
        self.pushButton_img.clicked.connect(self.button_image_open)
        self.pushButton_video.clicked.connect(self.button_video_open)
        self.pushButton_camera.clicked.connect(self.button_camera_open)
        self.timer_video.timeout.connect(self.show_video_frame)

    def init_logo(self):
        pix = QtGui.QPixmap('背景.png')
        self.label.setScaledContents(True)
        self.label.setPixmap(pix)

    # .ui文件自动生成的
    def setupUi(self, MainWindow):
        MainWindow.setWindowIcon(QtGui.QIcon("./logo.png"))
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(-1, -1, 0, -1)
        self.verticalLayout.setSpacing(80)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton_img = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_img.sizePolicy().hasHeightForWidth())
        self.pushButton_img.setSizePolicy(sizePolicy)
        self.pushButton_img.setMinimumSize(QtCore.QSize(150, 100))
        self.pushButton_img.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.pushButton_img.setFont(font)
        self.pushButton_img.setObjectName("pushButton_img")
        self.verticalLayout.addWidget(self.pushButton_img, 0, QtCore.Qt.AlignHCenter)
        self.pushButton_camera = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_camera.sizePolicy().hasHeightForWidth())
        self.pushButton_camera.setSizePolicy(sizePolicy)
        self.pushButton_camera.setMinimumSize(QtCore.QSize(150, 100))
        self.pushButton_camera.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.pushButton_camera.setFont(font)
        self.pushButton_camera.setObjectName("pushButton_camera")
        self.verticalLayout.addWidget(self.pushButton_camera, 0, QtCore.Qt.AlignHCenter)
        self.pushButton_video = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_video.sizePolicy().hasHeightForWidth())
        self.pushButton_video.setSizePolicy(sizePolicy)
        self.pushButton_video.setMinimumSize(QtCore.QSize(150, 100))
        self.pushButton_video.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.pushButton_video.setFont(font)
        self.pushButton_video.setObjectName("pushButton_video")
        self.verticalLayout.addWidget(self.pushButton_video, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout.setStretch(2, 1)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 3)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "建筑废弃物目标检测"))
        self.pushButton_img.setText(_translate("MainWindow", "图片检测"))
        self.pushButton_camera.setText(_translate("MainWindow", "摄像头检测"))
        self.pushButton_video.setText(_translate("MainWindow", "视频检测"))
        self.label.setText(_translate("MainWindow", "TextLabel"))

    # 定义图片检测的插槽函数
    def button_image_open(self):
        yolo = YOLO()
        print('button_image_open')

        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        if not img_name:
            return

        img = Image.open(img_name)
        # img = cv2.imread(img_name)
        print(img_name)

        # 是否对目标进行裁剪
        crop = False
        # 是否计数
        count = False

        # 图片检测
        showimg = yolo.detect_image(img, crop = crop, count=count)

        # PIL转换为cv2显示
        showimg = np.array(showimg)
        showimg=cv2.cvtColor(showimg,cv2.COLOR_RGB2BGR)

        cv2.imwrite('prediction.jpg', showimg)
        self.result = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
        # self.result = cv2.resize(self.result, (1280, 720), interpolation=cv2.INTER_AREA)
        self.QtImg = QtGui.QImage(
            self.result.data, self.result.shape[1], self.result.shape[0], QtGui.QImage.Format_RGB32)
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
        # showimg.show()

    # 定义视频检测的插槽函数
    def button_video_open(self):
        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开视频", "", "*.mp4;;*.avi;;All Files(*)")

        if not video_name:
            return

        flag = self.cap.open(video_name)
        if flag == False:
            QtWidgets.QMessageBox.warning(
                self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            # self.cap.get(3) == cv2.VideoCapture.get(3) 表示在视频流的帧的宽度
            # self.cap.get(4) == cv2.VideoCapture.get(4) 在视频流的帧的高度
            self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(
                *'MJPG'), 30, (int(self.cap.get(3)), int(self.cap.get(4))))

            # 当计时器结束，执行show_video_frame函数
            self.timer_video.start(0) 
            self.pushButton_video.setDisabled(True)
            self.pushButton_img.setDisabled(True)
            self.pushButton_camera.setDisabled(True)


    def show_video_frame(self):
        yolo = YOLO()
    
        flag, img = self.cap.read()
        if img is not None:
            # 是否对目标进行裁剪
            crop = False
            # 是否计数
            count = False
            #cv2显示切换为PIL显示
            img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            # 图片检测
            showimg = yolo.detect_image(img, crop = crop, count=count)

            # PIL转换为cv2显示
            showimg = np.array(showimg)
            showimg=cv2.cvtColor(showimg,cv2.COLOR_RGB2BGR)

            cv2.imwrite('prediction.jpg', showimg)
            self.result = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
            # self.result = cv2.resize(self.result, (1280, 720), interpolation=cv2.INTER_AREA)
            self.QtImg = QtGui.QImage(
                self.result.data, self.result.shape[1], self.result.shape[0], QtGui.QImage.Format_RGB32)
            self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            # showimg.show()

        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            self.pushButton_video.setDisabled(False)
            self.pushButton_img.setDisabled(False)
            self.pushButton_camera.setDisabled(False)
            self.init_logo()
    
    # 定义检测realsense的插槽函数
    def button_camera_open(self):
        # ModBus远程连接到服务器端
        master = mt.TcpMaster("192.168.1.202",502)
        master.set_timeout(5.0)

        yolo = YOLO()
        if not self.timer_video.isActive():
            # 深度相机初始化
            pipeline = rs.pipeline()
            config = rs.config()
            # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

            # config.enable_stream(rs.stream.depth, 1920, 1080, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            pipeline.start(config)
            align_to_color = rs.align(rs.stream.color)

            # 定义计数器
            count = 0
            fps = 0.0
            object_count = 1
            time_order = 1

            # modbus发送同步信号————置0
            value = master.execute(slave=1, function_code=md.WRITE_SINGLE_REGISTER, starting_address=3800, quantity_of_x=1, output_value=0)
            value = master.execute(slave=1, function_code=md.WRITE_SINGLE_REGISTER, starting_address=3801, quantity_of_x=1, output_value=0)
            
            while(True):

                t1 = time.time()
                # 读取某一帧
                frames = pipeline.wait_for_frames()
                frames = align_to_color.process(frames)
                frame = frames.get_color_frame()
                # 转变成Image
                frame = Image.fromarray(np.asanyarray(frame.get_data()))
                # 格式转变，BGRtoRGB
                frame = cv2.cvtColor(np.array(frame),cv2.COLOR_BGR2RGB)
                frame1 = Image.fromarray(np.asanyarray(frame))
                # 进行检测
                frame = np.array(yolo.detect_image(frame1))

                # 每循环十次向外传递一次信息
                if count % 2 == 0 :
                    # value = master.execute(slave=1, function_code=md.WRITE_SINGLE_REGISTER, starting_address=3800, quantity_of_x=1, output_value=0)
                    # value = master.execute(slave=1, function_code=md.WRITE_SINGLE_REGISTER, starting_address=3801, quantity_of_x=1, output_value=1)
                    #检测物体位置类别信息
                    x_axis,y_axis,z_axis,r_axis,grasp_width,label,box_width = yolo.detect_image_info(frame1)
                    
                    if label == 9 :
                        # 格式转变，RGBtoBGR,cv2图像的显示格式为BGR
                        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                        
                        fps  = ( fps + (1./(time.time()-t1)) ) / 2
                        # print("fps= %.2f"%(fps))
                        frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        cv2.imshow("video",frame)
                        count = count+1
                        cv2.waitKey(1) & 0xff 
                        continue
                    else:
                        print(f"第{object_count}个物体：\t时间序号：{time_order}; \tY轴：{y_axis};  \tX轴：{x_axis};  \tZ轴：{z_axis};  \tR轴：{r_axis};  \t槽号：{label};  \t识别框宽度：{box_width}")
                        # modbus发送同步信号————置1
                        value = master.execute(slave=1, function_code=md.WRITE_SINGLE_REGISTER, starting_address=3800, quantity_of_x=1, output_value=0)
                        value = master.execute(slave=1, function_code=md.WRITE_SINGLE_REGISTER, starting_address=3801, quantity_of_x=1, output_value=1)

                        # druring_time = time.time() - t1
                        # print(druring_time)

                        # ModBus 发送数据
                        yolo.send_data(int(time_order),float(y_axis),float(x_axis),float(z_axis),float(r_axis),float(grasp_width),int(label),float(box_width))
                        # print(f"第{object_count}个物体：\t时间序号：{time_order}; \tY轴：{y_axis};  \tX轴：{x_axis};  \tZ轴：{z_axis};  \tR轴：{r_axis};  \t电爪轴：{grasp_width};  \t槽号：{label};")
                        object_count = object_count + 1
                        time_order = time_order + 1

                        time.sleep(0.1)
                        # modbus发送同步信号————置0
                        value = master.execute(slave=1, function_code=md.WRITE_SINGLE_REGISTER, starting_address=3800, quantity_of_x=1, output_value=0)
                        value = master.execute(slave=1, function_code=md.WRITE_SINGLE_REGISTER, starting_address=3801, quantity_of_x=1, output_value=0)

                        # druring_time = time.time() - t1
                        # print(druring_time)

                # 格式转变，RGBtoBGR,cv2图像的显示格式为BGR
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                
                fps  = ( fps + (1./(time.time()-t1)) ) / 2
                # print("fps= %.2f"%(fps))
                frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                self.result = cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA)
                self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1],self.result.shape[0],QtGui.QImage.Format_RGB32)
                self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
                
                cv2.imshow("video",frame)
                count = count+1
                cv2.waitKey(1) & 0xff 

                # druring_time = time.time() - t1
                # print(druring_time)
        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            self.init_logo()
            self.pushButton_video.setDisabled(False)
            self.pushButton_img.setDisabled(False)
            self.pushButton_camera.setDisabled("摄像头检测")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())