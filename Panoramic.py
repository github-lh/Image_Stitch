import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv


"""全景图拼接类"""
class Panoramic:
    panoramic=None

    """获取匹配关键点与特征描述符"""
    def Detect_Feature_And_KeyPoints(self, image):
        detector = cv.xfeatures2d.SURF_create()
        (Keypoints, descriptors) = detector.detectAndCompute(image, None)
        Keypoints = np.float32([i.pt for i in Keypoints])
        return (Keypoints, descriptors)

    """KNN匹配特征描述符"""
    def Descriptors_Match(self,descriptors_A,descriptors_B):
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        knn_Matches = flann.knnMatch(descriptors_A, descriptors_B, 2)
        return knn_Matches


    """去除特征匹配点对列表中的非真实点对"""
    def Good_Match(self,knn_Matches,lowe_ratio=0.7):
        good = []
        for m in knn_Matches:
            if m[0].distance < lowe_ratio * m[1].distance:
                good.append((m[0].trainIdx,m[0].queryIdx))
        return good

    """提取视角变换矩阵M，与图像掩膜matchesMask"""
    def Good_Select(self,good,Keypoints_A,Keypoints_B,MIN_MATCH_COUNT=30):
        if len(good)>MIN_MATCH_COUNT:
            src_Points = np.float32([Keypoints_A[i] for (_,i) in good])
            dst_Points = np.float32([Keypoints_B[i] for (i,_) in good])
            M,mask=cv.findHomography(src_Points,dst_Points,cv.RANSAC,4.0)
            matchesMask = mask.ravel().tolist()
            return M,matchesMask
        else:
            print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
            return None,None


    """水平最优缝合线算法（基于动态规划算法的实现）"""
    def quilting(self,overlap1, overlap2, mid,RGB=True):
        print(overlap1.shape)
        print(overlap2.shape)
        E = np.full((overlap1.shape[0],overlap1.shape[1]),0)
        trace = np.zeros_like(E)
        if RGB==False:
            for i in range(E.shape[0]):
                for j in range(E.shape[1]):
                    E[i][j] = abs(int(overlap1[i][j]) - int(overlap2[i][j]))
        else:
            for i in range(E.shape[0]):
                for j in range(E.shape[1]):
                    E[i][j]=abs(int(overlap1[i][j][0])+int(overlap1[i][j][1])+int(overlap1[i][j][2])- int(overlap2[i][j][0])-int(overlap2[i][j][1])-int(overlap2[i][j][2]))
        a = np.ndarray(E.shape[1])
        for i in range(len(a)):
            a[i] = -1
        min_error = np.zeros(E.shape)
        for i in range(min_error.shape[0]):
            for j in range(min_error.shape[1]):
                min_error[i][j] = float('inf')
                trace[i][j] = -1
        min_error[mid][0] = E[mid][0]
        for j in range(min_error.shape[1] - 1):
            for i in range(min_error.shape[0]):
                if i == 0:
                    t = float('inf')
                    for k in (i, i + 1):
                        if min_error[k][j] < t:
                            t = min_error[k][j]
                            trace[i][j + 1] = k
                    min_error[i][j + 1] = t + E[i][j + 1]
                elif i == min_error.shape[0] - 1:
                    t = float('inf')
                    for k in (i - 1, i):
                        if min_error[k][j] < t:
                            t = min_error[k][j]
                            trace[i][j + 1] = k
                    min_error[i][j + 1] = t + E[i][j + 1]
                else:
                    t = float('inf')
                    for k in (i - 1, i, i + 1):
                        if min_error[k][j] < t:
                            t = min_error[k][j]
                            trace[i][j + 1] = k
                    min_error[i][j + 1] = t + E[i][j + 1]
        min = float('inf')
        position = -1
        for i in range(min_error.shape[0]):
            if min_error[i][min_error.shape[1] - 1] < min:
                min = min_error[i][min_error.shape[1] - 1]
                position = i
        a[overlap1.shape[1] - 1] = position
        for i in range(1, overlap1.shape[1]):
            a[overlap1.shape[1] - 1 - i] = trace[int(a[overlap1.shape[1] - i])][overlap1.shape[1] - i]
        return a.astype(int)

    """视角变换矩阵改良，保证图片变换结果完整性"""
    def M_fix(self,M,img1):
        x1 = np.dot(M, [0, 0, 1])
        y1 = np.dot(M, [0,img1.shape[0]-1, 1])
        x = np.dot(M, [img1.shape[1]-1,0, 1])
        y = np.dot(M, [img1.shape[1]-1, img1.shape[0]-1, 1])
        x_excursion = min(y1[0], x1[0])
        if x_excursion < 0:
            x_excursion = -x_excursion
        else:
            x_excursion = 0
        y_excursion = min(x1[1], x[1])
        if y_excursion < 0:
            y_excursion = -y_excursion
        else:
            y_excursion = 0
        move = np.float32(M)
        move[0][0] = move[0][0] + x_excursion * move[2][0]
        move[0][1] = move[0][1] + x_excursion * move[2][1]
        move[0][2] = move[0][2] + x_excursion * move[2][2]
        move[1][0] = move[1][0] + y_excursion * move[2][0]
        move[1][1] = move[1][1] + y_excursion * move[2][1]
        move[1][2] = move[1][2] + y_excursion * move[2][2]
        return move

    """以变换图为参照时,调用去图像黑边函数"""
    def reduce_black_area(self,img4,result,channel,top,left):
        for i in range(top,top+img4.shape[0]):
            for j in range(left,left+img4.shape[1]):
                for c in range(channel):
                    if(img4[i-top][j-left][c]!=0):
                        result[i][j][c]=img4[i-top][j-left][c]
        return result


    """图像拼接"""
    def getwarp_perspective(self,img1, img2, M,Anchor_Point):

        (x_anchor,y_anchor,x_anchor1,y_anchor1)=Anchor_Point
        img3 = img1.copy()
        cv.Stitcher
        img4 = img2.copy()
        Anchor=[x_anchor,y_anchor,1]
        Anchor=np.dot(M,Anchor)

        lt=[0,0,1]

        """x[0]:横坐标 x[1]:纵坐标"""
        lb=[0,img3.shape[0]-1,1]
        rt=[img3.shape[1]-1,0,1]
        rb=[img3.shape[1]-1,img3.shape[0]-1,1]

        lt = np.dot(M,lt)
        rt = np.dot(M, rt)
        lb = np.dot(M, lb)
        rb = np.dot(M, rb)
        Y_Anchor_Point=Anchor[1]/Anchor[2]
        X_Anchor_Point=Anchor[0]/Anchor[2]
        tran_y_max=max(rb[1]/rb[2],lb[1]/lb[2])
        tran_x_max=max(rb[0]/rb[2],rt[0]/rt[2])
        shape_y=max(Y_Anchor_Point,y_anchor1)+max(tran_y_max-Y_Anchor_Point,img4.shape[0]-y_anchor1)
        shape_x=max(X_Anchor_Point,x_anchor1)+max(tran_x_max-X_Anchor_Point,img4.shape[1]-x_anchor1)
        result=cv.warpPerspective(img1,M,(round(shape_x),round(shape_y)))


        """模糊图像增强method1"""
        # image_equalize = cv.cvtColor(result, cv.COLOR_BGR2YUV)
        # # 通道分离
        # channels = cv.split(image_equalize)
        # # 对图像的灰阶通道进行直方图均衡化
        # channels[0] = cv.equalizeHist(channels[0])
        # # 三通道合成彩色图片
        # image_equalize = cv.merge(channels)
        # # 将图片由YUV转为BGR
        # result = cv.cvtColor(image_equalize, cv.COLOR_YUV2BGR)

        """模糊图像增强method2"""
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 定义一个核
        # result = cv.filter2D(result, -1, kernel=kernel)


        # cv.imshow("transform picture", img3)
        # cv.imshow("reference picture", img4)
        # cv.imshow("transformed",result)
        # cv.waitKey(0)
        if X_Anchor_Point>x_anchor1:
            left=round(X_Anchor_Point-x_anchor1)
        else:
            left=0
        if Y_Anchor_Point>y_anchor1:
            top=round(Y_Anchor_Point-y_anchor1)
        else:
            top=0

        result[top:top+img4.shape[0],left:left+img4.shape[1]]=img4

        # channel = img4.shape[2]
        # result=self.reduce_black_area(img4,result,channel,top,left)


        # for i in range(top,top+img4.shape[0]):
        #     for j in range(left,left+img4.shape[1]):
        #         for c in range(channel):
        #             if(img4[i-top][j-left][c]!=0):
        #                 result[i][j][c]=img4[i-top][j-left][c]

        # cv.imshow("result",result)
        # cv.waitKey(0)
        # left=round(lt[0]/lt[2])-round(padding_left1)-1
        # right=round(lt[0]/lt[2])+img2.shape[1]-round(padding_left1)-1
        # if left<0:
        #     t=abs(left)
        #     left=0
        #     right=right+t
        # elif result.shape[1]-right<0:
        #     t=result.shape[1]-right
        #     right=right+t
        #     left=left+t
        # bottom=round(rb[1]/rb[2])+round(padding_bottom1)-1
        # top=round(rb[1]/rb[2])+round(padding_bottom1)-1-img2.shape[0]
        # if top<0:
        #     t=abs(top)
        #     top=0
        #     bottom=bottom+t
        # elif result.shape[0]-bottom<0:
        #     t=result.shape[0]-bottom
        #     bottom=bottom+t
        #     top=top+t

        # print(result.shape)
        # overleft=round(x_tl[0])
        # overright=round(x_tr[0])
        # overtop=round(x_tl[1])
        # overbottom=round(x_bl[1])
        # result[overtop:overbottom, overleft:overright]=(255,0,0)



        """使用最优缝合线需传入coordinate（调用Get_Keypoint_range函数）"""
        # overtop1=round(y_min1)-int(min(padding_top, padding_top1))
        # overbottom1=round(y_max1)+int(min(padding_bottom, padding_bottom1))
        # overleft1=round(x_min1)-int(min(padding_left1,padding_left))
        # overright1=round(x_max1)+int(min(padding_right,padding_right1))
        #
        # overleft=round(lt[0])-int(min(padding_left,padding_left1))
        # overright=overleft+overright1-overleft1
        # overtop=round(lt[1])-int(min(padding_top, padding_top1))
        # overbottom=overtop+overbottom1-overtop1
        # mid = round((overbottom - overtop) / 2)

        """最优缝合线"""
        # position=self.quilting(result[overtop:overbottom,overleft:overright],
        #                        img2[overtop1:overbottom1,overleft1:overright1],mid)


        # for j in range(len(position)):
        #     for i in range(top,top+position[j]):
        #         result[i][left+j]=img2[i-top][j]
                # result[i][left + j] =(255,0,0)
        # result[top:bottom, left:right] = img2
        # result[top:top+round(y_max1+padding_bottom1),left:right] = img2[:,:]
        return  result


    """测试用，画出匹配基准图"""
    def get_points(self,imageA,imageB):
        print(imageA.shape)
        print(imageB.shape)
        (hA, wA) = self.get_image_dimension(imageA)
        (hB, wB) = self.get_image_dimension(imageB)
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB
        return vis

    """获得图像二维长宽信息"""
    def get_image_dimension(self,image):
        (h,w) = image.shape[:2]
        return (h,w)


    """(测试用)特征匹配连线"""
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # 初始化可视化图片，将A、B图左右连接到一起
        (hA, wA) = self.get_image_dimension(imageA)
        vis = self.get_points(imageA, imageB)
        # 联合遍历，画出匹配对
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 当点对匹配成功时，画到可视化图上
            if s == 1:
                # 画出匹配对
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv.line(vis, ptA, ptB, (0, 255, 0), 1)
        # 返回可视化结果
        return vis

    """获取拼接图像特征匹配特征点矩形区域坐标"""
    """x_min.x_max,y_min,y_max为第一幅图片特征区域的横纵坐标最大，最小值"""
    def Get_Keypoint_range(self,kp1,kp2,good,matchmask):
        x_max = kp1[good[0][1]][0]
        x_min = kp1[good[0][1]][0]
        y_max = kp1[good[0][1]][1]
        y_min = kp1[good[0][1]][1]
        x_max1 = kp2[good[0][0]][0]
        x_min1 = kp2[good[0][0]][0]
        y_max1 = kp2[good[0][0]][1]
        y_min1 = kp2[good[0][0]][1]
        for i in range(len(good)):
            if matchmask[i]==1:
                if kp1[good[i][1]][0] > x_max:
                    x_max = kp1[good[i][1]][0]
                elif kp1[good[i][1]][0] < x_min:
                    x_min = kp1[good[i][1]][0]
                if kp1[good[i][1]][1] > y_max:
                    y_max = kp1[good[i][1]][1]
                elif kp1[good[i][1]][1] < y_min:
                    y_min = kp1[good[i][1]][1]
                if kp2[good[i][0]][0] > x_max1:
                    x_max1 = kp2[good[i][0]][0]
                elif kp2[good[i][0]][0] < x_min1:
                    x_min1 = kp2[good[i][0]][0]
                if kp2[good[i][0]][1] > y_max1:
                    y_max1 = kp2[good[i][0]][1]
                elif kp2[good[i][0]][1] < y_min1:
                    y_min1 = kp2[good[i][0]][1]
        return x_max,x_min,y_max,y_min,x_max1,x_min1,y_max1,y_min1


    """获取图像匹配锚点"""
    def Get_Anchor_Point(self,kp1,kp2,good,matchmask):
        for i in range(len(good)):
            if matchmask[i]==1:
                return kp1[good[i][1]][0],kp1[good[i][1]][1],kp2[good[i][0]][0],kp2[good[i][0]][1]


    """图像匹配函数"""
    def match(self,img1,img2,lowe_ratio=0.7):
        kp1, des1 = self.Detect_Feature_And_KeyPoints(img1)
        kp2, des2 = self.Detect_Feature_And_KeyPoints(img2)
        knn_matches=self.Descriptors_Match(des1,des2)
        good=self.Good_Match(knn_matches)
        M,matchesMask=self.Good_Select(good,kp1,kp2)
        if np.array(M).any():
            M_fix = self.M_fix(M, img1)
            # coordinate=self.Get_Keypoint_range(kp1,kp2,good,matchesMask)
            Anchor_Point=self.Get_Anchor_Point(kp1,kp2,good,matchesMask)
            # img3 = self.drawMatches(img1, img2, kp1, kp2, good, matchesMask)
            # cv.imshow("connect",img3)
            result=self.getwarp_perspective(img1,img2,M_fix,Anchor_Point)
            # cv.imshow("result",result)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            # plt.subplot(311)
            result1=cv.cvtColor(result,cv.COLOR_BGR2RGB)
            # plt.imshow(result1)
            self.panoramic=result
            cv.imwrite("orthographic perspective result.jpg", self.panoramic)
            # plt.subplot(312)
            # img3 =self.drawMatches(img1,img2,kp1,kp2,good,matchesMask)
            # img31 = cv.cvtColor(img3, cv.COLOR_BGR2RGB)
            # plt.imshow(img31)
            # plt.subplot(313)
            # plt.imshow(self.panoramic), plt.show()
            return 1
        else:
            print("特征不足，匹配失败")
            return 0

    """批量匹配"""
    def batch_match(self,images,lowe_ratio=0.7):
        imgs=[]
        num=len(images)
        for i in range(num):
            print("第",i,"轮拼接")
            if i==0:
                self.panoramic=images[num-1]
                flag=self.match(images[num-1],images[num-2])
                if flag==0:
                    imgs.append(images[num-2])
            elif num-i>1:
                flag=self.match(self.panoramic,images[num-i-2])
                if flag==0:
                    imgs.append(images[num-i-2])
        """待拼接图片列表=0或者无拼接匹配项时退出匹配"""
        while len(imgs)!=0 and i<(num+1)/2:
            i+=1
            img=imgs[0]
            imgs.pop(0)
            flag = self.match(self.panoramic,img)
            if flag==0:
                imgs.append(img)
        self.panoramic=cv.cvtColor(self.panoramic,cv.COLOR_BGR2RGB)
        plt.imshow(self.panoramic),plt.show()
