import exifread
import re
import os
import cv2 as cv

"""图像预处理类"""
class IMG_exif_process():
    IMGS=[]#记录图片位置
    GPS=[]#纪录图片位置信息
    file_dir=''#存储图片文件夹位置
    num=0#记录文件数目
    def __init__(self):
        file_dir='IMG'
        for parent,dirnames,flienames in os.walk(file_dir):
            for filename in flienames:
                if filename.endswith(('.JPG','.jpg')):
                    self.IMGS.append(os.path.join(parent, filename))

    def __init__(self,filedir):
        file_dir=filedir
        for parent,dirnames,flienames in os.walk(file_dir):
            for filename in flienames:
                if filename.endswith(('.JPG','.jpg')):
                    self.IMGS.append(os.path.join(parent, filename))

    """获取文件夹中的所有图片"""
    def Get_IMGS(self,zoomFlag=False):
        imgs=[]
        for i in range(len(self.IMGS)):
            img = cv.imread(self.IMGS[i])
            if zoomFlag==True:
                img=self.Zoom_IMG(img)
            imgs.append(img)
        return imgs


    """图像压缩，提高图像处理速度"""
    def Zoom_IMG(self,img):
        res = cv.resize(img, (int(img.shape[1] * 0.3), int(img.shape[0] * 0.3)), interpolation=cv.INTER_CUBIC)
        return res

    """获取无人机图像GPS"""
    def get_img_GPS(self,img):
        tags = exifread.process_file(img)
        for tag, value in tags.items():
            if re.match('GPS GPSLatitude', tag):
                try:
                    match_result = re.match('\[(\w*), (\w*), (\w.*)/(\w.*)\]', str(value)).groups()
                    Latitude = int(match_result[0]) + int(match_result[1]) / 60 + int(match_result[2]) / int(
                        match_result[3]) / 3600
                except:
                    Latitude = value
            if re.match('GPS GPSLongitude', tag):
                try:
                    match_result = re.match('\[(\w*), (\w*), (\w.*)/(\w.*)\]', str(value)).groups()
                    Longitude = int(match_result[0]) + int(match_result[1]) / 60 + int(match_result[2]) / int(
                        match_result[3]) / 3600
                except:
                    Longitude = value
            if re.match('Image DateTime', tag):
                Data = str(value)
        GPS = {'照片':img.name,'经度':Longitude, '纬度':Latitude,'时间':Data}
        return GPS


    """批量获取图片GPS"""
    def batch_generate_GPS(self):
        img=''
        self.num=len(self.IMGS)
        print("图片数量:",self.num)
        for i in range(self.num):
            img=self.IMGS[i]
            self.GPS.append(self.get_img_GPS(open(img,"rb")))
