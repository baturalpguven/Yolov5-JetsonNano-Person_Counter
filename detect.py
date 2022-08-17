import sys
sys.path.insert(0, './yolov5')
from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, 
check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.plots import plot_one_box
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np
import pandas as pd
from pandas import ExcelWriter
import math

##from smbus2 import SMBus
##from mlx90614 import MLX90614
### GPIO library
##import Jetson.GPIO as GPIO
## 
### Handles time
##import time 
 
import mysql.connector as mysql
mydb = mysql.connect(
 host=" localhost",
 user="root",
 password="",
 database="our_project"
)
mycursor = mydb.cursor() #-------------------------------------------> iniatates the table

mycursor.execute("DROP TABLE IF EXISTS social_distance_vialation")
sq2="CREATE TABLE social_distance_vialation (id INT AUTO_INCREMENT PRIMARY 
KEY, distance_between VARCHAR(255), violeter_id_1 VARCHAR(255), violeter_id_2 
VARCHAR(255)\
 )"
mycursor.execute(sq2)
-------------------------------------------------------------------> Lights Up and closes leds
def led_funct(count):
 # Pin Definition
 led_pin = 7
 
 # Set up the GPIO channel
 GPIO.setmode(GPIO.BOARD) 
 GPIO.setup(led_pin, GPIO.OUT, initial=GPIO.HIGH) 
 
 
 
 # Blink the LED
 if count!=0:
 GPIO.output(led_pin, GPIO.HIGH) 

 print("LED is ON")
 else:
 GPIO.output(led_pin, GPIO.LOW)
 print("LED is OFF")
---------------------------------------------------------------->
##_______________________________________>
##def thermometer_funct():
### define sensor i2c address
## thermometer_address = 0x5a
##
## 
## # create sensor object
## bus = SMBus(1)
## sensor = MLX90614(bus, address=thermometer_address)
##
## print("* MLX90614 Temperature *")
## print("Object | Ambient")
##
##
## # read sensor values
## object_temp = sensor.get_object_1() + 5 #note: method might have changed to 
get_obj_temp()
## ambient_temp = sensor.get_ambient() #note: method might have changed to 
get_amb_temp()
##

## # print readings to console
## # {} is used in conjunction with format() for substitution.
## # .2f - format to 2 decimal places.
## print("{0:>5.2f}C | {1:>5.2f}C".format(object_temp, ambient_temp), end='\r')
##
## time.sleep(1)
##
## bus.close()
 
##________________________________________>
def compute_color_for_id(label):
 """
 Simple function that adds fixed color depending on the id
 """
 palette = (2 * 11 - 1, 2 * 15 - 1, 2 ** 20 - 1)
 color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
 return tuple(color)
#----------------------------------------------------------------->
def mySQL_Connection(violater_list):
 

 
 val = (violater_list)
 sq3=" INSERT INTO social_distance_vialation 
(distance_between,violeter_id_1,violeter_id_2) VALUES (%s,%s,%s)"
 mycursor.executemany(sq3,val)
 mydb.commit()
 #-------------------------------------------------->
#-----------------------------------------------------------------> save to excel
def WriteItToExcel(*args, **kwargs):
 mydb = mysql.connect(
 host=" localhost",
 user="root",
 password="",
 database="our_project"
 )
 mycursor = mydb.cursor()
 sql = "SELECT*FROM social_distance_vialation"

 mycursor.execute(sql)
 databases=[]
 for x in mycursor:
 print(x,"\n")
 databases.append(x[1:])
 
 print((databases))
 w = ExcelWriter('Violater_people_table.xlsx')
 df_list = []
 df = pd.DataFrame(columns=['Distance between', 'Violater_id_1', 'Violater_id_2'])#-------------
------> label the coulmns
 for i in range(len(databases)):#----------------------------------------> itarate over the numpy array 
and add each row to corrosponding index
 df.loc[i] = databases[i]

 print(df)
 df_list.append(df)#------------------------------> save data to excel
 for df in (df_list):
 df.to_excel(w)
 w.save()
#-------------------------------------->
def detect(opt):
 out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate 
= \
 opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, 
opt.save_vid, \
 opt.save_txt, opt.img_size, opt.evaluate
 webcam = source == '0' or source.startswith(
 'rtsp') or source.startswith('http') or source.endswith('.txt')
 # initialize deepsort
 cfg = get_config()
 cfg.merge_from_file(opt.config_deepsort)
 attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
 deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
 max_dist=cfg.DEEPSORT.MAX_DIST, 
min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,

 nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, 
max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
 max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, 
nn_budget=cfg.DEEPSORT.NN_BUDGET,
 use_cuda=True)
 # Initialize
 device = select_device(opt.device)
 # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
 # its own .txt file. Hence, in that case, the output folder is not restored
 if not evaluate:
 if os.path.exists(out):
 pass
 shutil.rmtree(out) # delete output folder
 os.makedirs(out) # make new output folder
 half = device.type != 'cpu' # half precision only supported on CUDA
 # Load model
 model = attempt_load(yolo_weights, map_location=device) # load FP32 model
 stride = int(model.stride.max()) # model stride
 imgsz = check_img_size(imgsz, s=stride) # check img_size
 names = model.module.names if hasattr(model, 'module') else model.names # get class names
 if half:
 model.half() # to FP16
104
 # Set Dataloader
 vid_path, vid_writer = None, None
 # Check if environment supports image displays
 if show_vid:
 show_vid = check_imshow()
 if webcam:
 cudnn.benchmark = True # set True to speed up constant image size inference
 dataset = LoadStreams(source, img_size=imgsz, stride=stride)
 else:
 dataset = LoadImages(source, img_size=imgsz)
 # Get names and colors
 names = model.module.names if hasattr(model, 'module') else model.names
 # Run inference
 if device.type != 'cpu':
 model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))) # run 
once
 t0 = time.time()
 save_path = str(Path(out))
 # extract what is in between the last '/' and last '.'
 txt_file_name = source.split('/')[-1].split('.')[0]
 txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'
105
 for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
 img = torch.from_numpy(img).to(device)
 img = img.half() if half else img.float() # uint8 to fp16/32
 img /= 255.0 # 0 - 255 to 0.0 - 1.0
 if img.ndimension() == 3:
 img = img.unsqueeze(0)
 # Inference
 t1 = time_synchronized()
 pred = model(img, augment=opt.augment)[0]
 # Apply NMS
 pred = non_max_suppression(
 pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
 t2 = time_synchronized()
 # Process detections
 
 for i, det in enumerate(pred): # detections per image
 
 if webcam: # batch_size >= 1
 p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
 else:
 p, s, im0 = path, '', im0s
106
 s += '%gx%g ' % img.shape[2:] # print string
 save_path = str(Path(out) / Path(p).name)
 if det is not None and len(det):
 # Rescale boxes from img_size to im0 size
 det[:, :4] = scale_coords(
 img.shape[2:], det[:, :4], im0.shape).round()
 # Print results
 for c in det[:, -1].unique():
 n = (det[:, -1] == c).sum() # detections per class
 s += '%g %ss, ' % (n, names[int(c)]) # add to string
 xywhs = xyxy2xywh(det[:, 0:4])
 confs = det[:, 4]
 clss = det[:, 5]
 # pass detections to deepsort
 outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss, im0)
 
 # draw boxes for visualization
 if len(outputs) > 0:
 centroid_list=[]
107
 
 for j, (output, conf) in enumerate(zip(outputs, confs)):
 
 bboxes = output[0:4]
 id = output[4]
 cls = output[5]
 
 
 centroid_x=(output[0] +output[2])/2
 centroid_y=(output[1] +output[3])/2
 tuple_xy_xentroid=(centroid_x,centroid_y,id)
 
 centroid_list.append(tuple_xy_xentroid)
 x_boundry= int(im0.shape[1])#------------------------------------> sets boundry and 
rest gives in and out / x boundry
108
 y_boundry= int(im0.shape[0]/2) # / y boundry
# print(y_boundry,x_boundry)
 in_list=[]
 out_list=[]
 for i in centroid_list:
 if i[1]> y_boundry: #----------------------------> middle of the image take it from 
its sizes
 in_list.append(i)
 
 else:
 out_list.append(i)
 socail_distance_list=[] #-----------------------------------------> calculate social 
distance bewtween each image
 for i in range(0,len(in_list)):
 for j in range(i+1,len(in_list)):
 socail_distance=math.dist(in_list[i][:2],in_list[j][:2])
 
socail_distance_list.append((int(round(socail_distance)),int(in_list[i][2]),int(in_list[j][2])))
 count=0
109
 violater_list=[]
 for i in socail_distance_list: # counts social distance
 if i[0]< 100: # i[0] contains socail distance
 count+=1
 violater_list.append(i)
 mySQL_Connection(violater_list)#--------------------------------------------------> 
Database connection with mySQL
 led_funct(count)#--------------------------------------------------------------------> led 
openerer closre function 
 
 
 cv2.rectangle(im0, (25,75),(200,120),(0,0,255),-1)
 cv2.rectangle(im0, (25,640),(200,680),(0,0,255),-1)#-----------------------------------
---------------------------> writes ın out values and draw the boundry
 cv2.putText(im0, 'OUT:'+f'{len(out_list)}', 
(50,100),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
 cv2.putText(im0, 'IN:'+f'{len(in_list)}', 
(50,660),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
 cv2.rectangle(im0, (0,(y_boundry)),(x_boundry,(y_boundry+15)),(0,0,255),-1)
## 
 
 
110
## print("In_list:", len(in_list))
##
## print("tuple:", tuple_xy_xentroid)
##
## print("socail_distance_list:", socail_distance_list)
##
## print("violater_list:", violater_list)
## 
## print("out_list:", len(out_list))
##
## if count!=0:
## print("Social_distance_vialation",count+1)
## else:
## print("Social_distance_vialation",count)
 
 c = int(cls) # integer class
 label = f'{id} {names[c]} {conf:.2f} ' #--------------------------------------------> to 
understand output variable 
 color = compute_color_for_id(id)
 plot_one_box(bboxes, im0, label=label, color=color, line_thickness=2)
 if save_txt:
 # to MOT format
 bbox_top = output[0]
 bbox_left = output[1]
111
 bbox_w = output[2] - output[0]
 bbox_h = output[3] - output[1]
 # Write MOT compliant results to file
 with open(txt_path, 'a') as f:
 f.write(('%g ' * 10 + '\n') % (frame_idx, id, bbox_top,
 bbox_left, bbox_w, bbox_h, -1, -1, -1, -1)) # label format
 else:
 deepsort.increment_ages()
 # Print time (inference + NMS)
 print('%sDone. (%.3fs)' % (s, t2 - t1))
 # Stream results
 if True: #show_vid
 cv2.imshow(p, im0)
 if cv2.waitKey(1) == ord('q'): # q to quit
 WriteItToExcel()
 raise StopIteration
 # Save results (image with detections)
 if save_vid:
 if vid_path != save_path: # new video
 vid_path = save_path
 if isinstance(vid_writer, cv2.VideoWriter):
112
 vid_writer.release() # release previous video writer
 if vid_cap: # video
 fps = vid_cap.get(cv2.CAP_PROP_FPS)
 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 else: # stream
 fps, w, h = 30, im0.shape[1], im0.shape[0]
 save_path += '.mp4'
 vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, 
h))
 vid_writer.write(im0)
 if save_txt or save_vid:
 print('Results saved to %s' % os.getcwd() + os.sep + out)
 if platform == 'darwin': # MacOS
 os.system('open ' + save_path)
 print('Done. (%.3fs)' % (time.time() - t0))
if _name_ == '_main_':
 parser = argparse.ArgumentParser()
113
 parser.add_argument('--yolo_weights', type=str, default='yolov5s.pt', help='model.pt path') #--
------------------------- to track mec_plastik weights
 parser.add_argument('--deep_sort_weights', type=str, 
default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
 # file/folder, 0 for webcam
 parser.add_argument('--source', type=str, default='source/project', help='source') #---------------
--------------------- iamges converterd to video
 parser.add_argument('--output', type=str, default='inference/output', help='output folder') # 
output folder
 parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
 parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence 
threshold')
 parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
 parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify 
ffmpeg support)')
 parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
 parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
 parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
 parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to 
*.txt')
 # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
 parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 
16 17')
 parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
 parser.add_argument('--augment', action='store_true', help='augmented inference')
 parser.add_argument('--evaluate', action='store_true', help='augmented inference')
 parser.add_argument("--config_deepsort", type=str, 
default="deep_sort_pytorch/configs/deep_sort.yaml")
114
 args = parser.parse_args()
 args.img_size = check_img_size(args.img_size)
 with torch.no_grad():
 detect(args)