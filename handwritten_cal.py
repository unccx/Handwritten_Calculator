
#~~~3. 预测 ~~~
import numpy as np
from PIL import Image
from itertools import groupby
import  matplotlib.pyplot as plt

def predict_expression(image, model):
    # 加载图片
    # image = Image.open("testing.png").convert("L")

    t = np.array(image)

    # 调整大小至28个高度像素（模型输入要求）
    w = image.size[0]
    h = image.size[1]
    r = w / h # 宽高比
    new_w = int(r * 28)
    new_h = 28
    new_image = image.resize((new_w, new_h))

    # 转换成numpy array
    new_image_arr = np.array(new_image)

    # 反转图像，使background = 0
    new_inv_image_arr = 255 - new_image_arr

    # 将图像归一化
    final_image_arr = new_inv_image_arr / 255.0

    # 根据是否是非零列将图像数组分割成单个数字数组
    m = np.any(final_image_arr, axis=0)
    out = [final_image_arr[:,[*g]] for k, g in groupby(np.arange(len(m)), lambda x: m[x] != 0) if k]

    # 迭代数字数组以调整它们的大小，使之符合模型的输入标准 [mini_batch_size, height, width, channels] 。
    num_of_elements = len(out)
    elements_list = []

    print("len(out): " + str(num_of_elements))

    for x in range(0, num_of_elements):

        img = out[x]

        plt.imsave("symbols\img_"+str(x)+".png", img)
        
        # 添加0列作为填充物
        width = img.shape[1]
        filler = (final_image_arr.shape[0] - width) / 2
        
        if filler.is_integer() == False:    # 填充奇数个0列
            filler_l = int(filler)
            filler_r = int(filler) + 1
        else:                               # 填充偶数个0列
            filler_l = int(filler)
            filler_r = int(filler)
        
        arr_l = np.zeros((final_image_arr.shape[0], filler_l)) # 左边填充
        arr_r = np.zeros((final_image_arr.shape[0], filler_r)) # 右边填充
        
        # element_arr = arr_l + img + arr_r 填充满28的宽度
        help_ = np.concatenate((arr_l, img), axis= 1)
        element_arr = np.concatenate((help_, arr_r), axis= 1)
        
        element_arr.resize(28, 28, 1) # 将2d array调整为3d array

        # 所有元素存进一个列表中
        elements_list.append(element_arr)


    elements_array = np.array(elements_list)

    # reshaping以适应模型输入标准
    elements_array = elements_array.reshape(-1, 28, 28, 1)

    # 用模型预测
    # model = keras.models.load_model("model.h5")
    elements_pred =  model.predict(elements_array)
    elements_pred = np.argmax(elements_pred, axis = 1)

    return elements_pred

#~~~4. 数学运算 ~~~

def math_expression_generator(arr):
    
    op = {
            10,   # = "/"
            11,   # = "+"
            12,   # = "-"
            13    # = "*"
         }   
    
    m_exp = []
    temp = []
    
    # 创建一个分隔所有元素的列表
    for item in arr:
        if item not in op:
            temp.append(item)
        else:
            m_exp.append(temp)
            m_exp.append(item)
            temp = []
    if temp:
        m_exp.append(temp)
    
    # 将元素转换为数字和运算符
    i = 0
    num = 0
    for item in m_exp:
        if type(item) == list: # 数字list
            if not item: # 跳过empty list
                m_exp[i] = ""
                i = i + 1
            else:
                num_len = len(item)
                for digit in item:
                    num_len = num_len - 1
                    num = num + ((10 ** num_len) * digit)
                m_exp[i] = str(num)
                num = 0
                i = i + 1
        else: # 运算符
            m_exp[i] = str(item)
            m_exp[i] = m_exp[i].replace("10","/")
            m_exp[i] = m_exp[i].replace("11","+")
            m_exp[i] = m_exp[i].replace("12","-")
            m_exp[i] = m_exp[i].replace("13","*")
            
            i = i + 1
    
    # 连接字符串列表，创建数学表达式。
    separator = ' '
    m_exp_str = separator.join(m_exp)
    
    return (m_exp_str)


def calculate(elements_pred):
    # 创建数学表达式
    m_exp_str = math_expression_generator(elements_pred)

    # 使用eval()计算数学表达式
    while True:
        try:
            answer = eval(m_exp_str)    #计算答案
            answer = round(answer, 2)
            equation  = m_exp_str + " = " + str(answer)
            print(equation)   #打印等式
            return equation
            break

        except SyntaxError:
            print("无效的预测表达式!")
            print("以下是预测的表达式:")
            print(m_exp_str)
            return m_exp_str
            break 
