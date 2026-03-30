# 课程设计Part1

<div style="text-align: center;">专业：智能科学与技术 学号：2312654 姓名：李玉泽</div>

![image-20250427095650741](C:\Users\cassi\AppData\Roaming\Typora\typora-user-images\image-20250427095650741.png)

## 一、实验目的

本实验的目的是研究自行设计算法在全景图拼接中的应用，深入理解传统计算机视觉中特征匹配与图像变换的原理，通过对 Simple 和 Difficult 两组局部图像以及自行采集的第三组局部图像进行处理，实现图像拼接并完成图像融合，评判其在简单与复杂场景下的鲁棒性，为后续优化（如多频段融合、深度学习拼接）奠定基础。

学习并掌握自行设计全景图拼接算法的流程和原理。

学会处理不同类型局部图像，包括存在变形的图像，实现准确拼接。

掌握图像融合技术，有效减少拼接图像中的 “接缝”，提高拼接图像质量。

## 二、实验原理

#### 2.1 SIFT特征检测

通过尺度不变特征变换(SIFT)算法提取图像中的关键点(Keypoints)和特征描述符(Descriptors)，确保在不同视角、光照或尺度下仍能稳定匹配。

SIFT算法的具体流程如下：

##### 2.1.1 尺度空间极值检测（Scale-space Extrema Detection）

1. **构建高斯金字塔**：

   对输入图像进行不同尺度的高斯模糊（通过高斯核卷积），生成多组（Octaves）图像，每组包含多层（Layers）。每组图像尺寸逐层减半（降采样），形成金字塔结构。

   ![alt](C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\文档插图\tower.png)

2. **构建高斯差分金字塔（DoG）**：

   对相邻高斯模糊图像做差分（DoG），增强关键点检测的稳定性。

   ![alt](C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\文档插图\DOG.png)

3. **检测极值点**：

   在DoG金字塔中，将每个像素点与其相邻的26个点（同层的8邻域+上下层的各9个点）比较，寻找局部极值点。

   ![alt](C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\文档插图\check keypoint.png)

##### **2.1.2 关键点精确定位（Keypoint Localization）**

1. **去除低对比度点**：

   通过泰勒展开拟合DoG函数，计算极值点的精确位置和对比度，剔除对比度过低的点（对噪声敏感）。

2. **消除边缘响应**：

   利用Hessian矩阵计算关键点的曲率，剔除边缘上的不稳定点（保留角点等稳定特征）。

##### **2.1.3 关键点方向分配（Orientation Assignment）**

1. **计算梯度幅值和方向**：

   在关键点邻域内，计算每个像素的梯度幅值和方向。

2. **生成梯度方向直方图**：

   将360°划分为36个区间（每10°一区间），统计邻域内梯度方向的分布。

3. **确定主方向**：

   取直方图的峰值方向作为关键点的主方向，并将超过峰值方向80&的方向作为关键点辅方向。

   ![alt](C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\文档插图\m_d figure.png)

##### **2.1.4 生成特征描述符（Descriptor Generation）**

1. **划分局部区域**：

   在关键点周围16×16的区域内，划分4×4的子块（共16个子块）。

2. **计算子块梯度直方图**：

   对每个子块计算8方向的梯度直方图，共生成4×4×8=128维的特征向量。

3. **归一化处理**：

   对特征向量进行归一化（减少光照变化影响），并截断较大值（增强鲁棒性）。

   ![alt](C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\文档插图\describer.png)

#### 2.2 特征匹配

使用Brute-Force匹配器（BFMatcher）和Lowe's比率测试（距离比值<0.75）筛选优质匹配点，排除误匹配，提高匹配鲁棒性。

对于两幅图像，遍历其关键点，对于每个关键点的特征向量，采取欧氏距离比较选中的关键点的特征描述符，并运用KNN算法获取距离最小的两个点，并采用Lowe's比率测试，以最近邻距离比<0.75为条件，剔除误匹配，提高匹配精度。

#### 2.3 **图像对齐与几何变换**

基于特征匹配结果，通过计算单应性矩阵实现图像间的几何对齐，确保多幅图像能够无缝拼接成全景图。

##### 2.3.1 单应性矩阵计算（Homography Estimation）

1. **特征点匹配对筛选**：

   使用RANSAC（随机抽样一致）算法从匹配点对中筛选最优内点集，剔除误匹配点对，提高矩阵计算精度。

   RANSAC很好的保证了匹配的精度，在计算单应性矩阵时，我们并不追求匹配特征点的数量，虽然其数量越多结果越鲁棒，相反，我们需要极高的精准度，因为如果出现误匹配点，这对单应性矩阵的计算带来的结果是毁灭性的。

2. **单应性矩阵求解**：

   利用最小二乘法求解最优单应性矩阵H，矩阵H满足：X' = H·X，其中X为原图像点坐标，X'为目标图像点坐标，通过单应性矩阵对待拼接图像进行射影变换，以适应目标图像的坐标系。

   ![alt](C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\文档插图\Homography.png)

##### 2.3.2 透视变换（Perspective Transformation）

1. **边界计算**：

   计算变换后图像的四个角点坐标，确定全景图的最小外包矩形尺寸

2. **坐标平移调整**：

   构建平移矩阵确保所有像素坐标为正值，避免图像内容被裁剪，随后对图像进行拼接，将当前拼接结果与新图像的未重叠区域直接叠加。

   ![alt](C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\文档插图\figure1.jpeg)

#### 2.4 多图像拼接策略

对于给定的数据集，由于其包含多个图像，因此其拼接顺序显得尤为重要，如果拼接顺序出现错误将导致全景图失真，因此采用**匹配点数量优先策略**，根据图像间的匹配点数量动态决定拼接顺序，优先合并匹配度最高的图像对，逐步扩展全景图范围，同时在每次拼接后，更新全景图的特征点和描述符，用于后续图像的匹配与拼接。

## 三、实验步骤

#### 3.1 图像导入

在目录下，包含三组图像，其相对路径分别为"Simple"，"Difficult"以及"Mydata"，首先将他们下载后存入列表中等待下一步处理。

<div style="display: flex; justify-content: space-between;">
  <img src="C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\Simple\S1.jpg" width="14%" />
  <img src="C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\Simple\S2.jpg" width="14%" /> 
  <img src="C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\Simple\S3.jpg" width="14%" />
    <img src="C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\Simple\S4.jpg" width="14%" />
    <img src="C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\Simple\S5.jpg" width="14%" />
    <img src="C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\Simple\S6.jpg" width="14%" />
    <img src="C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\Simple\S7.jpg" width="14%" />
</div>

以上述"Simple"数据集内包含的图片为例，我们将其存入列表种等待下一步处理。

#### 3.2 SIFT特征检测与匹配

随后我们对每一张图片都进行SIFT特征检测，获取其关键点，并计算其对应的描述子，随后对每一对图像都进行特征点的匹配，挑选出匹配的特征点数量最多的一对图像，进行拼接。

#### 3.3 图像拼接

对于确定好的两张图像，我们获取其匹配的特征点序列，并基于特征点序列计算其单应性矩阵， 对需要拼接的图像进行坐标系的变换，随后将图像拼接，得出新图像，并不断重复上述过程，直至所有图像均被使用，生成一张全景图。

具体流程图如下：

![alt](C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\文档插图\流程图.png)

对于最后生成的图片，在拼接过程中接缝处可能存在不平滑，为此我们考虑对一定阈值条件下的像素邻域用中值滤波对其进行处理，以确保全景图的平滑，最后生成图片并保存。

## 四、程序代码

全景图拼接.py

```python
import os
import cv2
import numpy as np

def load_data(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        images.append(img)
    return images

folder_path = "Simple"
Simple_image_data = load_data(folder_path)

def stitch_images_with_sift(images):
    """
    使用SIFT特征进行图像拼接，基于匹配点数量决定拼接顺序,优先拼接匹配度高的图像。
    """
    n = len(images)
    # 创建SIFT特征检测器
    sift = cv2.SIFT_create()
    
    # 计算所有图像的特征点和描述符
    gray_images = []
    keypoints = []
    descriptors = []
    
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_images.append(gray)
        kp, des = sift.detectAndCompute(gray, None)
        keypoints.append(kp)
        descriptors.append(des)
    
    # 创建特征匹配器
    bf = cv2.BFMatcher()
    
    # 计算所有图像对之间的匹配点数量
    match_counts = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            matches = bf.knnMatch(descriptors[i], descriptors[j], k=2)
            good_matches = []
            for m, nn in matches:
                if m.distance < 0.75 * nn.distance:
                    good_matches.append(m)
            match_counts[i, j] = len(good_matches)
            match_counts[j, i] = len(good_matches)  # 矩阵对称
    
    # 决定拼接顺序
    used_images = [False] * n
    result_idx = np.unravel_index(np.argmax(match_counts), match_counts.shape)
    i, j = result_idx
    
    # 初始拼接结果使用匹配度最高的两张图
    used_images[i] = True
    used_images[j] = True
    
    # 拼接第一对图像
    img1, img2 = images[i], images[j]
    result = stitch_pair(img1, img2, keypoints[i], keypoints[j], descriptors[i], descriptors[j])
    
    # 保存已拼接图像的特征
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result_kp, result_des = sift.detectAndCompute(result_gray, None)
    
    # 逐步添加剩余图像
    while False in used_images:
        best_match_count = 0
        best_idx = -1
        
        # 寻找与当前拼接结果匹配度最高的未使用图像
        for i in range(n):
            if used_images[i]:
                continue
                
            matches = bf.knnMatch(result_des, descriptors[i], k=2)
            good_matches = []
            for m, nn in matches:  # 将变量n改为nn
                if m.distance < 0.75 * nn.distance:
                    good_matches.append(m)
                    
            if len(good_matches) > best_match_count:
                best_match_count = len(good_matches)
                best_idx = i
        
        if best_match_count < 4:  # 至少需要4个点来计算单应性矩阵
            print(f"无法找到更多可靠匹配的图像，已使用{sum(used_images)}/{n}张图像")
            break
            
        # 拼接匹配度最高的下一张图像
        next_img = images[best_idx]
        result = stitch_pair(result, next_img, result_kp, keypoints[best_idx], result_des, descriptors[best_idx])
        
        # 更新已使用图像和拼接结果的特征
        used_images[best_idx] = True
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result_kp, result_des = sift.detectAndCompute(result_gray, None)
    
    return result

def stitch_pair(img1, img2, kp1, kp2, des1, des2):
    """拼接两张图像"""
    # 创建特征匹配器
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # 应用Lowe's比率测试筛选匹配点
    good_matches = []
    for m, nn in matches:
        if m.distance < 0.75 * nn.distance:
            good_matches.append(m)
    
    # 至少需要4个匹配点来计算单应性矩阵
    if len(good_matches) >= 4:
        # 提取匹配点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 计算单应性矩阵
        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        # 检查H是否为None或无效
        if H is None:
            print("单应性矩阵计算失败")
            return img1
            
        # 确保H是正确的数据类型
        H = np.float32(H)
        
        # 获取图像尺寸
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 计算img2在变换后的边界坐标，方便匹配
        pts = np.float32([[0, 0], [0, h2-1], [w2-1, h2-1], [w2-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)
        
        # 计算边界点，包括两个图像的所有角点
        corners = np.concatenate([
            dst,
            np.float32([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]]).reshape(-1, 1, 2)
        ])
        
        # 计算最小和最大坐标
        [x_min, y_min] = np.int32(corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(corners.max(axis=0).ravel() + 0.5)
        
        # 创建平移矩阵，确保图像在正坐标空间
        t = [-x_min, -y_min]
        H_translation = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], dtype=np.float32)
        
        # 计算输出尺寸
        output_size = (x_max - x_min, y_max - y_min)
        
        # 使用try-except捕获可能的错误
        try:
            # 对两个图像应用透视变换
            warped1 = cv2.warpPerspective(img1, H_translation, output_size)
            H_combined = H_translation.dot(H)
            warped2 = cv2.warpPerspective(img2, H_combined, output_size)
            
            # 创建掩码以标识每个图像中的有效区域
            gray1 = cv2.cvtColor(warped1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(warped2, cv2.COLOR_BGR2GRAY)
            _, mask1 = cv2.threshold(gray1, 1, 255, cv2.THRESH_BINARY)
            _, mask2 = cv2.threshold(gray2, 1, 255, cv2.THRESH_BINARY)
            
            # 创建结果图像
            panorama = np.zeros_like(warped1)
            
            # 将两个图像融合到全景图中
            panorama = cv2.bitwise_or(
                panorama, 
                cv2.bitwise_and(warped2, warped2, mask=cv2.bitwise_not(mask1))
            )
            panorama = cv2.bitwise_or(
                panorama,
                cv2.bitwise_and(warped1, warped1)
            )
            
            return panorama
        except cv2.error as e:
            print(f"透视变换失败: {e}")
            return img1
    else:
        print(f"匹配特征点不足，只有{len(good_matches)}个")
        return img1  # 如果无法拼接，返回第一张图像

def remove_seam_lines(panorama, threshold=10, kernel_size=5):
    """
    消除全景图中的拼接黑线，使用中值滤波
    
    参数:
    - panorama: 输入的全景图
    - threshold: 黑色像素的阈值，小于此值的像素被视为黑线区域
    - kernel_size: 中值滤波核大小，必须是奇数
    
    返回:
    - 处理后的全景图
    """
    # 复制原图以保留未处理区域
    result = panorama.copy()
    
    # 转为灰度图以检测黑线
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    
    # 创建黑线掩码 - 寻找低于阈值的区域
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # 扩大黑线区域以确保完全覆盖过渡边缘
    kernel = np.ones((3,3), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    
    # 对黑线区域应用中值滤波
    filtered_panorama = cv2.medianBlur(panorama, kernel_size)
    
    # 将中值滤波结果只应用于黑线区域
    for c in range(3):  # 对BGR三个通道分别处理
        # 在黑线区域使用滤波结果，其他区域保持原图
        result[:,:,c] = np.where(dilated_mask == 255, 
                                 filtered_panorama[:,:,c], 
                                 panorama[:,:,c])
                        
    return result

# 使用SIFT进行图像拼接
panorama = stitch_images_with_sift(Simple_image_data)

# 显示拼接结果
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
panorama = remove_seam_lines(panorama, threshold=15, kernel_size=3)
cv2.imshow('Result', panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 保存拼接结果
cv2.imwrite('panorama_sift.jpg', panorama)


```

在上述代码中，只需要更改文件读取路径即可对不同数据集进行全景图拼接，例如，现代码中处理的是"Simple"，并将其存为'panorama_sift.jpg'于文件目录下。

## 五、实验结果显示

对于刚刚的"Simply"样例，在程序中的运行结果如下所示：

![](C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\实验结果\panorama_sift.jpg)

对于"Difficult"数据集，其包含以下8张图片；

<div style="display: flex; justify-content: space-between;">
  <img src="C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\Difficult\D1.jpg" width="12%" />
  <img src="C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\Difficult\D2.jpg" width="12%" /> 
  <img src="C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\Difficult\D3.jpg" width="12%" />
    <img src="C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\Difficult\D4.jpg" width="12%" />
    <img src="C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\Difficult\D5.jpg" width="12%" />
    <img src="C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\Difficult\D6.jpg" width="12%" />
    <img src="C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\Difficult\D7.jpg" width="12%" />
    <img src="C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\Difficult\D8.jpg" width="12%" />
</div>

同样在程序中运行，其全景图如下图所示：

![alt](C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\实验结果\difficult_panorama_sift.jpg)

随后我又对学校内其他地点的图像进行了收集,并将其分割为较小的图像如下所示：

<div style="display: flex; justify-content: space-between;">
  <img src="C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\Mydata\M1.jpg" width="16%" />
  <img src="C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\Mydata\M2.jpg" width="16%" /> 
  <img src="C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\Mydata\M3.jpg" width="16%" />
    <img src="C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\Mydata\M4.jpg" width="16%" />
    <img src="C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\Mydata\M5.jpg" width="16%" />
    <img src="C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\Mydata\M6.jpg" width="16%" />
</div>

对其用同样的程序进行拼接，其结果如下所示：

![alt](C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计\实验结果\mydata_panorama_sift.jpg)

可以看出算法运行效果很好，均成功实现了预期效果，同时不同的数据集也充分展示了算法的鲁棒性。