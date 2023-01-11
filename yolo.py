import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
from utils.utils_bbox import decode_outputs, non_max_suppression

# import gol

# modbus
import modbus_tk.modbus_tcp as mt
import modbus_tk.defines as md
import struct

# ModBus远程连接到服务器端
master = mt.TcpMaster("192.168.1.202",502)
master.set_timeout(5.0)

'''
训练自己的数据集必看注释！
'''
class YOLO(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        "model_path"        : 'logs/yolox_s_cw.pth',
        "classes_path"      : 'model_data/cls_classes.txt',
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        "input_shape"       : [640, 640],
        #---------------------------------------------------------------------#
        #   所使用的YoloX的版本。nano、tiny、s、m、l、x
        #---------------------------------------------------------------------#
        "phi"               : 's',
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : True,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
            
        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()
        
        # show_config(**self._defaults)

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self, onnx=False):
        self.net    = YoloBody(self.num_classes, self.phi)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        # print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, crop = False, count = False):
        #---------------------------------------------------#
        #   获得输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return image

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        #---------------------------------------------------------#
        #   计数
        #---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        #---------------------------------------------------------#
        #   是否进行目标的裁剪
        #---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        #---------------------------------------------------------#
        #   图像绘制
        #-------------------------
        top_array = []
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            score = top_conf[i]
            box = top_boxes[i]
            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            # print(label, top, left, bottom, right)

            top_array.append(top)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image


    def detect_image_info(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        
        image       = cvtColor(image)
        
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return image

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        # 判断是否有物体被识别
        if(len(top_boxes) == 0):
            x_axis = 0
            y_axis = 0
            z_axis = 244
            r_axis = 0
            grasp_width = 25
            label = 9
            box_width = 300

            return x_axis,y_axis,z_axis,r_axis,grasp_width,label,box_width
            
        
        top_label = YOLO.switch_label(top_label)
        
        j = 0
        box_0 = top_boxes[0]
        top_0, left_0, bottom_0, right_0 = box_0
        top_bottom = bottom_0
        for i,c in list(enumerate(top_label)):
            box             = top_boxes[i]
            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            # left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            # right   = min(image.size[0], np.floor(right).astype('int32'))

            # if ((bottom <= top_bottom) ):
            #     if((len(top_boxes) == 1) & (top < 30) ) :
            #         return 0,0,0,244,25,9,300
            #     else:
            #         continue
            # else:
            #     j = i
            #     top_bottom = bottom
            if ((top > 20) and (bottom >= np.floor(top_bottom)) and (bottom < 460)):
                j = i
                top_bottom = bottom
            else :
                if(len(top_boxes) == i+1):
                    return 0,0,0,244,25,9,300
                else:
                    continue

        box_use = top_boxes[j]
        top_use, left_use, bottom_use, right_use = box_use
        label_use = top_label[j]

        x_axis_in_pixel = (left_use+right_use)/2
        y_axis_in_pixel = (top_use+bottom_use)/2
        z_axis_in_pixel = 1020

        x_axis,y_axis = YOLO.get_coordinate_in_cam(x_axis_in_pixel,y_axis_in_pixel,z_axis_in_pixel)

        # 求像素点(left_use,top_use）对应相机位置(x_left_use,y_top_use)，即识别框左上角
        x_left_use,y_top_use = YOLO.get_coordinate_in_cam(left_use,top_use,z_axis_in_pixel)
        # 求像素点(right_use,bottom_use）对应相机位置(x_right_use,y_bottom_use)，即识别框右下角
        x_right_use,y_bottom_use = YOLO.get_coordinate_in_cam(right_use,bottom_use,z_axis_in_pixel)

        # 判断抓取方向及识别框抓取宽度
        if abs(bottom_use-top_use)>abs(right_use-left_use):
            grasp_angle = 0.0000
            box_width = abs(x_right_use-x_left_use) + 70
        else:
            grasp_angle = 90.0000
            box_width = abs(y_bottom_use-y_top_use) + 70

        # 调转相机X轴
        x_axis = -x_axis
        y_axis = y_axis

        z_axis = 244
        r_axis = grasp_angle
        grasp_width = 25
        label = label_use

        # # ModBus 发送数据
        # YOLO.send_data(int(time_order),float(y_axis),float(x_axis),float(z_axis),float(r_axis),float(grasp_width),int(label))
        # print(f"第{object_count}个物体：\t时间序号：{time_order}; \tY轴：{y_axis};  \tX轴：{x_axis};  \tZ轴：{z_axis};  \tR轴：{r_axis};  \t电爪轴：{grasp_width};  \t槽号：{label};")
        # object_count = object_count + 1
        # time_order = time_order + 1
        
        return x_axis,y_axis,z_axis,r_axis,grasp_width,label,box_width

    def get_coordinate_in_cam(x_axis_in_pixel,y_axis_in_pixel,z_axis_in_pixel):

        #Linux读取内参
        intr_matrix = np.array([[607.023,0,325.536],
                                [0,606.652,230.354],
                                [0,0,1]])

        coordinate_in_pixel = np.array([[x_axis_in_pixel],[y_axis_in_pixel],[1]])
        coordinate_in_cam = np.dot(np.linalg.inv(intr_matrix),np.dot(z_axis_in_pixel,coordinate_in_pixel))
        x_axis_in_cam = coordinate_in_cam[0][0]
        y_axis_in_cam = coordinate_in_cam[1][0]
        return x_axis_in_cam,y_axis_in_cam

    def get_coordinate_in_world(x_axis_in_pixel,y_axis_in_pixel,z_axis_in_pixel):
        intr_matrix = np.array([[553.4734,0,317.0819],
             [0,553.2318,232.2433],
             [0,0,1]])

        RT = np.array([[-2.40911664e-02,9.91940099e-0,-1.24396363e-01,-1.33796737e+02],
                    [ 9.94187219e-01,3.68330168e-02,1.01168686e-01,-1.35072709e+03],
                    [ 1.04935170e-01,-1.21236003e-01,-9.87061519e-01,0.00000000e+00],
                    [ 0.00000000e+00,0.00000000e+00,0.00000000e+00,1.00000000e+00]])

        coordinate_in_pixel = np.array([[x_axis_in_pixel],[y_axis_in_pixel],[1]])
        coordinate_in_world = np.array([[],[],[],[]])

        coordinate_in_cam = np.dot(np.linalg.inv(intr_matrix),np.dot(z_axis_in_pixel,coordinate_in_pixel))
        # print("coordinate_in_cam:\n",coordinate_in_cam,"\n")
        # print(coordinate_in_cam[0][0])
        # print(coordinate_in_cam[1][0])
        # print(coordinate_in_cam[2][0])

        # append = np.array([1])
        # coordinate_in_cam_4dim = np.row_stack(coordinate_in_cam,1)
        coordinate_in_cam_4dim = np.array([[coordinate_in_cam[0][0]],[coordinate_in_cam[1][0]],[coordinate_in_cam[2][0]],[1]])
        # print("coordinate_in_cam_4dim:\n",coordinate_in_cam_4dim,"\n")
        coordinate_in_world = np.dot(np.linalg.inv(RT),coordinate_in_cam_4dim)

        x_axis_in_world = coordinate_in_world[0][0]
        y_axis_in_world = coordinate_in_world[1][0]
        z_axis_in_world = coordinate_in_world[2][0]

        return x_axis_in_world,y_axis_in_world,z_axis_in_world

    # 小端模式处理数据
    def floatToABCD(value):
	    valueByte = struct.unpack('>HH',struct.pack('>f', value))
	    return valueByte

    def send_data(self,time_order,y_axis,x_axis,z_axis,r_axis,grasp_width,label,box_width):
        # 设置数据
        y_axis_Byte = YOLO.floatToABCD(y_axis)
        x_axis_Byte = YOLO.floatToABCD(x_axis)
        z_axis_Byte = YOLO.floatToABCD(z_axis)
        r_axis_Byte = YOLO.floatToABCD(r_axis)
        grasp_width_Byte = YOLO.floatToABCD(grasp_width)
        box_width_Byte = YOLO.floatToABCD(box_width)
        
        ######################################### 发送数据 #############################################
        # 写入通信心跳
        value = master.execute(slave=1, function_code=md.WRITE_SINGLE_REGISTER, starting_address=3000, quantity_of_x=1, output_value=0)
        value = master.execute(slave=1, function_code=md.WRITE_SINGLE_REGISTER, starting_address=3001, quantity_of_x=1, output_value=1)

        # # 写入时间序号
        value = master.execute(slave=1, function_code=md.WRITE_SINGLE_REGISTER, starting_address=3002, quantity_of_x=1, output_value=0)
        value = master.execute(slave=1, function_code=md.WRITE_SINGLE_REGISTER, starting_address=3003, quantity_of_x=1, output_value=time_order)
        
        # Y轴
        value = master.execute(slave=1, function_code=md.WRITE_MULTIPLE_REGISTERS, starting_address=3004, quantity_of_x=1, output_value=[y_axis_Byte[0],y_axis_Byte[1]])
        # 写入X轴
        value = master.execute(slave=1, function_code=md.WRITE_MULTIPLE_REGISTERS, starting_address=3006, quantity_of_x=1, output_value=[x_axis_Byte[0],x_axis_Byte[1]])
        # Z轴
        value = master.execute(slave=1, function_code=md.WRITE_MULTIPLE_REGISTERS, starting_address=3008, quantity_of_x=1, output_value=[z_axis_Byte[0],z_axis_Byte[1]])
        # R轴
        value = master.execute(slave=1, function_code=md.WRITE_MULTIPLE_REGISTERS, starting_address=3010, quantity_of_x=1, output_value=[r_axis_Byte[0],r_axis_Byte[1]])
        # # 电爪轴
        # value = master.execute(slave=1, function_code=md.WRITE_MULTIPLE_REGISTERS, starting_address=3012, quantity_of_x=1, output_value=[grasp_width_Byte[0],grasp_width_Byte[1]])
        # 放料槽号2;
        value = master.execute(slave=1, function_code=md.WRITE_SINGLE_REGISTER, starting_address=3014, quantity_of_x=1, output_value=0)
        value = master.execute(slave=1, function_code=md.WRITE_SINGLE_REGISTER, starting_address=3015, quantity_of_x=1, output_value=label)
        # 识别框宽度
        value = master.execute(slave=1, function_code=md.WRITE_MULTIPLE_REGISTERS, starting_address=3037, quantity_of_x=1, output_value=[box_width_Byte[0],box_width_Byte[1]])

    def switch_label(label):
        for i in range(len(label)):
            if label[i] == 0:
                label[i] = 1
            elif label[i] == 1:
                label[i] = 4
            elif label[i] == 2:
                label[i] = 9
            elif label[i] == 3:
                label[i] = 2
            elif label[i] == 4:
                label[i] = 3
            elif label[i] == 5:
                label[i] = 9
            elif label[i] == 6:
                label[i] = 5
            elif label[i] == 7:
                label[i] = 6
            elif label[i] == 8:
                label[i] = 9
            elif label[i] == 9:
                label[i] = 7
            elif label[i] == 10:
                label[i] = 8

        return label
        
    
    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                  
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                #---------------------------------------------------------#
                outputs = self.net(images)
                outputs = decode_outputs(outputs, self.input_shape)
                #---------------------------------------------------------#
                #   将预测框进行堆叠，然后进行非极大抑制
                #---------------------------------------------------------#
                results = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                            image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, heatmap_save_path):
        import cv2
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y
        #---------------------------------------------------#
        #   获得输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            
        outputs = [output.cpu().numpy() for output in outputs]
        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask    = np.zeros((image.size[1], image.size[0]))
        for sub_output in outputs:
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(sub_output, [0, 2, 3, 1])[0]
            score      = np.max(sigmoid(sub_output[..., 5:]), -1) * sigmoid(sub_output[..., 4])
            score      = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score    = (score * 255).astype('uint8')
            mask            = np.maximum(mask, normed_score)
            
        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200)
        print("Save to the " + heatmap_save_path)
        plt.cla()

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]
        
        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                        im,
                        f               = model_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))
        
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
