import tkinter as tk
import tkinter.filedialog
import cv2
import PIL.Image, PIL.ImageTk
from skimage import exposure, feature

from tools import *


# 创建一个按钮，点击后调用convert函数
def convert():
    # 使用OpenCV读入图像文件
    image = cv2.imread(text.get())
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 将灰度图像转换为PIL图像
    pil_image = PIL.Image.fromarray(gray)
    # 将PIL图像转换为Tkinter图像
    tk_image = PIL.ImageTk.PhotoImage(image=pil_image)
    # 使用Label组件显示图像
    label_converted.config(image=tk_image)
    label_converted.image = tk_image


def show_img(image):
    pil_image = PIL.Image.fromarray(image)
    # 将PIL图像转换为Tkinter图像
    tk_image = PIL.ImageTk.PhotoImage(image=pil_image)
    # 使用Label组件显示图像
    label_converted.config(image=tk_image)
    label_converted.image = tk_image


# 创建一个按钮，点击后弹出文件选择对话框
def select_image():
    # 调用filedialog.askopenfilename函数弹出文件选择对话框
    file_path = tk.filedialog.askopenfilename()
    # 将选择的文件路径显示在文本框中
    text.set(file_path)
    # 使用OpenCV读入图像文件
    image = cv2.imread(text.get(), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 获取图像的高度和宽度
    height, width = image.shape[:2]
    # 计算调整后的图像尺寸
    # resized_width = int((480 / 300) * 200)
    resized_height = 480
    resized_width = 480
    # 调整图像的大小
    # image = cv2.resize(image, (resized_width, resized_height), fx=resized_width / width, fy=resized_height / height)

    # 将图像转换为PIL图像
    pil_image = PIL.Image.fromarray(image)
    # 将PIL图像转换为Tkinter图像
    tk_image = PIL.ImageTk.PhotoImage(image=pil_image)
    # 使用Label组件显示图像
    label_origin.config(image=tk_image)
    label_origin.image = tk_image


def read_img(is_gray=False):
    image = cv2.imread(text.get(), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if is_gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def func_3_12():
    image = read_img()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_result = adjust_bright_contrast(image, -0.35, 0.4)
    show_img(img_result)


def func_3_3():
    # 使用OpenCV读入图像文件
    image = read_img()
    img_result = adjust_bright_contrast(image, -0.35, 0.4)
    show_img(img_result)


def func_4():
    image = read_img()
    # 采用百分位数（percentile）选择rlow和rhigh
    rlow_p2, rhigh_p98 = np.percentile(image, (2, 98))
    img_rescale2 = exposure.rescale_intensity(image, in_range=(rlow_p2, rhigh_p98))
    show_img(img_rescale2)


def func_5():
    image = read_img()
    # 计算图像的直方图
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # 绘制直方图图像
    hist_image = np.zeros((300, 256, 3))
    # 将直方图归一化到[0, 300]
    hist = hist / hist.max() * 300
    # 绘制柱状图
    for i in range(256):
        cv2.line(hist_image, (i, 300), (i, 300 - int(hist[i])), (255, 255, 255))
    show_img(hist_image.astype(np.uint8))


def func_6():
    image = read_img(is_gray=True)
    # 构造一个大小为5×5的均值滤波核
    kav5 = np.ones((5, 5), np.float32) / 25
    img_blur = convolve(image, kav5)
    show_img(img_blur)


def func_7():
    gray = read_img(is_gray=True)
    # 将图像数组填充为2的整数次幂
    rows, cols = gray.shape
    nrows = cv2.getOptimalDFTSize(rows)
    ncols = cv2.getOptimalDFTSize(cols)
    nimg = np.zeros((nrows, ncols))
    nimg[:rows, :cols] = gray

    # 傅里叶变换
    fimg = np.fft.fft2(nimg)

    # 将低频部分（中心部分）取反
    fimg[int(nrows / 3):int(nrows * 2 / 3), int(ncols / 3):int(ncols * 2 / 3)] = -fimg[
                                                                                  int(nrows / 3):int(nrows * 2 / 3),
                                                                                  int(ncols / 3):int(ncols * 2 / 3)]

    # 逆傅里叶变换
    rimg = np.fft.ifft2(fimg)

    # 取实部
    rimg = np.abs(rimg)

    # 归一化
    rimg = (rimg - rimg.min()) / (rimg.max() - rimg.min()) * 255
    rimg = rimg.astype("uint8")
    show_img(rimg)


def func_8():
    image = read_img()
    # 令k=0.0025模拟大气湍流模糊退化
    img_deg, Hatm = AtmoTurbulenceSim(image, 0.0025)

    # 向退化图像中添加向图像中添加均值为0、方差为0.001的高斯噪声
    img_deg_noi = util.random_noise(img_deg, mode='gaussian', var=0.001)
    img_deg_noi = util.img_as_ubyte(img_deg_noi)

    # 获取退化图像的高/宽
    rows, cols = image.shape[0:2]

    # 采用'reflect'方式扩展图像(下面扩展rows行,右面扩张cols列)
    if image.ndim == 3:
        # 彩色图像
        imgex = np.pad(img_deg_noi, ((0, rows), (0, cols), (0, 0)), mode='reflect')
    elif image.ndim == 2:
        # 灰度图像
        imgex = np.pad(img_deg_noi, ((0, rows), (0, cols)), mode='reflect')

    # 计算扩展图像的DFT并中心化
    img_dft = fftshift(fft2(imgex, axes=(0, 1)), axes=(0, 1))

    # 计算维纳滤波复原图像的频谱
    NSR = 0.005
    Gp = img_dft * np.conj(Hatm) / (np.abs(Hatm) ** 2 + NSR + np.finfo(np.float32).eps)
    # 去中心化
    Gp = ifftshift(Gp, axes=(0, 1))
    # DFT反变换并取实部
    imgp = np.real(ifft2(Gp, axes=(0, 1)))
    # 把输出图像的数据格式转换为uint8
    imgp = np.uint8(np.clip(imgp, 0, 255))
    # 截取imgp左上角与原图像大小相等的区域作为输出
    img_res = imgp[0:rows, 0:cols]
    show_img(img_res)


def func_9_1():
    image = read_img(is_gray=True)
    # 创建3*3方形结构元素
    kernel_square = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    # kernel_square = np.ones((3,3),np.uint8)
    # 对二值图像进行腐蚀
    img_erode1 = cv2.erode(image, kernel_square, iterations=1)
    # img_erode1 = cv.morphologyEx(img, cv.MORPH_ERODE, kernel_square)
    show_img(img_erode1)


def func_9_2():
    image = read_img(is_gray=True)
    # 创建7*7椭圆形结构元素
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # 膨胀
    img_dilate2 = cv2.dilate(image, kernel_ellipse)
    show_img(img_dilate2)


def func_10():
    image = read_img(is_gray=True)
    # OpenCV: #Canny算子
    edge_canny1 = cv2.Canny(image, threshold1=50, threshold2=200)

    # Scikit-image: Canny算子,采用缺省参数
    edge_canny2 = feature.canny(image)

    # 标准差sigma=1,指定高低阈值(边缘幅值的百分位数)
    edge_canny3 = feature.canny(image, sigma=1,
                                low_threshold=0.05,
                                high_threshold=0.95,
                                use_quantiles=True)
    # 增大高斯平滑滤波器的标准差sigma=2
    edge_canny4 = feature.canny(image, sigma=2)
    show_img(edge_canny1)


# 创建一个Tkinter窗口
window = tk.Tk()
window.title('图像编辑')
window.geometry('1000x640')  # 设置窗口大小为640x480像素
# 创建一个菜单栏
menu_bar = tk.Menu(window)
# 创建一个编辑菜单
edit_menu = tk.Menu(menu_bar)
edit_menu.add_command(label='灰度化', command=convert)
edit_menu.add_command(label='调整亮度和对比度', command=func_3_12)
edit_menu.add_command(label='线性灰度变换', command=func_4)
edit_menu.add_command(label='直方图', command=func_5)
edit_menu.add_command(label='空域线性滤波与平滑', command=func_6)
edit_menu.add_command(label='频域图像锐化', command=func_7)
edit_menu.add_command(label='维纳滤波复原', command=func_8)
edit_menu.add_command(label='形态学腐蚀', command=func_9_1)
edit_menu.add_command(label='形态学膨胀', command=func_9_2)
edit_menu.add_command(label='Canny算子边缘检测', command=func_10)
menu_bar.add_cascade(label='图像处理', menu=edit_menu)

# 将菜单栏添加到窗口
window.config(menu=menu_bar)

button = tk.Button(window, text='选择图像', command=select_image)
button.pack()

# 创建一个文本框，用来显示选择的文件路径
text = tk.StringVar()
entry = tk.Entry(window, textvariable=text)
entry.pack()

# button = tk.Button(window, text='灰度化', command=convert)
# button.pack()
# button = tk.Button(window, text='转换', command=convert)
# button.pack()
# button = tk.Button(window, text='转换', command=convert)
# button.pack()

# 创建两个Label组件，分别用来显示原图像和转换后的图像
# label_origin = tk.Label(window)
# label_origin.pack(side='left')
#
# label_converted = tk.Label(window)
# label_converted.pack(side='right')

# 创建一个Frame组件
frame = tk.Frame(window)

# 创建两个Label组件
label_origin = tk.Label(frame, text='原始图像')
label_converted = tk.Label(frame, text='变换后')

# 使用pack布局管理器将两个Label组件放在左右两侧
label_origin.pack(side='left')
label_converted.pack(side='right')

# 将Frame组件放在窗口中
frame.pack()

# 进入Tkinter消息循环
window.mainloop()
