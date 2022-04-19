import Image_exif_process as proecess
import Panoramic as pom


"""IMG为待拼接图像所在文件夹"""
imgs=proecess.IMG_exif_process('TEST')
Panoramic=pom.Panoramic()
Panoramic.batch_match(imgs.Get_IMGS(True))
"""注意设置特征匹配成功判断阈值Good_Select的（MIN_MATCH_COUNT）参数"""
"""每次局部拼接的结果将保存至result"""


# imgs=proecess.IMG_exif_process('IMG')
# imgs.Get_IMGS(True)

"""获取无人机图像GPS"""
# imgs.batch_generate_GPS()
# for i in range(len(imgs.GPS)):
#     print(imgs.GPS[i])

# Panoramic=pom.Panoramic()
# Panoramic.batch_match(imgs.Get_IMGS(True))

# def quilting(overlap1, overlap2, mid):
#     E = np.full((overlap1.shape[0],overlap1.shape[1]),0)
#     trace = np.zeros_like(E)
#     for i in range(E.shape[0]):
#         for j in range(E.shape[1]):
#             E[i][j]=abs(overlap1[i][j]-overlap2[i][j])
#     a =np.ndarray(E.shape[1])
#     for i in range(len(a)):
#         a[i]=-1
#     min_error = np.zeros(E.shape)
#     for i in range(min_error.shape[0]):
#         for j in range(min_error.shape[1]):
#             min_error[i][j] = float('inf')
#             trace[i][j] = -1
#     min_error[mid][0] = E[mid][0]
#     for j in range(min_error.shape[1] - 1):
#         for i in range(min_error.shape[0]):
#             if i == 0:
#                 t = float('inf')
#                 for k in (i, i + 1):
#                     if min_error[k][j] < t:
#                         t = min_error[k][j]
#                         trace[i][j + 1] = k
#                 min_error[i][j + 1] = t + E[i][j + 1]
#             elif i == min_error.shape[0] - 1:
#                 t = float('inf')
#                 for k in (i-1, i):
#                     if min_error[k][j] < t:
#                         t = min_error[k][j]
#                         trace[i][j + 1] = k
#                 min_error[i][j + 1] = t + E[i][j + 1]
#             else:
#                 t = float('inf')
#                 for k in (i-1,i, i + 1):
#                     if min_error[k][j] < t:
#                         t = min_error[k][j]
#                         trace[i][j + 1] = k
#                 min_error[i][j + 1] = t + E[i][j + 1]
#     min = float('inf')
#     position = -1
#     for i in range(min_error.shape[0]):
#         if min_error[i][min_error.shape[1] - 1] < min:
#             min = min_error[i][min_error.shape[1] - 1]
#             position = i
#     a[overlap1.shape[1] - 1] = position
#     for i in range(1, overlap1.shape[1]):
#         a[overlap1.shape[1] - 1 - i] = trace[int(a[overlap1.shape[1] - i])][overlap1.shape[1] - i]
#
#     return a.astype(int)
#
# quilting(np.array([[1,2,3],[5,11,4],[14,5,9]]),np.array([[20,0,1],[13,5,10],[1,5,9]]),1)





