# 创建一个按钮，点击后调用convert函数
import PIL
import cv2
import numpy as np
import cv2
from skimage import io,util,filters,restoration
from scipy.fft import fft,ifft,fft2,ifft2,fftshift,ifftshift

def convert(img_path):
    # 使用OpenCV读入图像文件
    image = cv2.imread(img_path)
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 将灰度图像转换为PIL图像
    pil_image = PIL.Image.fromarray(gray)
    # 将PIL图像转换为Tkinter图像
    tk_image = PIL.ImageTk.PhotoImage(image=pil_image)
    # 使用Label组件显示图像
    # label_converted.config(image=tk_image)
    # label_converted.image = tk_image


def adjust_bright_contrast(image, b, c):
    """
    输入参数: image - 图像数组，灰度图像，数据类型uint8;
                 b - 在区间[-1, 1]内取值.b<0,降低亮度;b>0,提高亮度;
                 c - 在区间[-1, 1]内取值.c<0,降低对比度;c>0,提高对比度;
    返回参数：img_out - 灰度变换结果，数组，数据类型uint8;
    """

    k = np.tan((45 + 44 * c) * np.pi / 180)

    # 初始化查表法
    lookUpTable = np.zeros((1, 256), np.uint8)
    for i in range(256):
        s = (i - 127.5 * (1 - b)) * k + 127.5 * (1 + b)
        lookUpTable[0, i] = np.clip(s, 0, 255)

    # 查表进行灰度变换
    img_out = cv2.LUT(image, lookUpTable)
    # 返回结果
    return img_out


def convolve(image, kernel):
    """
    convolve 计算图像与滤波核的线性卷积 V1.0,2021-10
    输入参数:
        image - 二维数组，灰度图像；
       kernel - 二维数组，滤波核系数,方形，如3×3,5×5等.
    返回值：
       output - 二维数组，滤波后得到的灰度图像
    """
    # 获取输入图像核滤波核的尺寸大小（高、宽）
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    # 初始化输出图像数组变量
    output = np.zeros((iH, iW), dtype="float")
    # 计算图像边界扩展参数
    padsize = (kW - 1) // 2

    # 扩展图像边界，方式为边界复制
    image = cv2.copyMakeBorder(image, padsize, padsize, padsize, padsize, cv2.BORDER_REPLICATE)
    # 填充0扩展图像
    # image = cv.copyMakeBorder(image, padsize, padsize, padsize, padsize, cv.BORDER_CONSTANT,value=127)
    # image = cv.copyMakeBorder(image, padsize, padsize, padsize, padsize, cv.BORDER_CONSTANT, value=0)

    # 自左到右、自上而下滑动滤波核，遍历每个像素坐标(x, y)
    for x in np.arange(padsize, iH + padsize):
        for y in np.arange(padsize, iW + padsize):
            # 抽取以像素(x, y)为中心的邻域图像,图像区域尺寸与滤波核kernel大小相同
            roi = image[x - padsize:x + padsize + 1, y - padsize:y + padsize + 1]
            # 将抽取的邻域像素与对应的滤波核系数相乘,再累加,得到像素(x, y)的卷积输出
            output[x - padsize, y - padsize] = (roi.astype("float") * kernel).sum()

    # 将输出图像结果做饱和处理,数据类型转换为uint8
    output = np.clip(output, 0, 255).astype("uint8")
    # 返回输出图像
    return output


def AtmoTurbulenceSim(image, k):

    # 获取图像高/宽
    rows, cols = image.shape[0:2]

    # 采用'reflect'方式扩展图像(下面扩展rows行,右面扩张cols列)
    if image.ndim == 3:
        # 彩色图像
        imgex = np.pad(image, ((0, rows), (0, cols), (0, 0)), mode='reflect')
    elif image.ndim == 2:
        # 灰度图像
        imgex = np.pad(image, ((0, rows), (0, cols)), mode='reflect')

    # 计算扩展图像的DFT,并中心化
    img_dft = fft2(imgex, axes=(0, 1))
    img_dft = fftshift(img_dft, axes=(0, 1))

    # 生成大气湍流模糊退化函数
    # 构建频域平面坐标网格数组，坐标轴定义v列向/u行向
    v = np.arange(-cols, cols)
    u = np.arange(-rows, rows)
    Va, Ua = np.meshgrid(v, u)

    D2 = Ua ** 2 + Va ** 2
    Hatm = np.exp(-k * (D2 ** (5.0 / 6.0)))

    if image.ndim == 3:
        # 彩色图像把H串接成三维数组
        Hatm = np.dstack((Hatm, Hatm, Hatm))

        # 计算图像DFT与大气湍流模糊退化函数的点积
    Gp = img_dft * Hatm
    # 去中心化
    Gp = ifftshift(Gp, axes=(0, 1))
    # DFT反变换,并取实部
    imgp = np.real(ifft2(Gp, axes=(0, 1)))
    # 把输出图像的数据格式转换为uint8
    imgp = np.uint8(np.clip(imgp, 0, 255))
    # 截取imgp左上角与原图像大小相等的区域作为输出
    imgout = imgp[0:rows, 0:cols]

    return imgout, Hatm