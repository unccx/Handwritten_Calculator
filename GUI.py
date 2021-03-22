from tkinter import *
import numpy as np
from PIL import ImageGrab, Image
from tensorflow.keras.models import load_model
from handwritten_cal import predict_expression, calculate

model = load_model('model.h5')
image_folder = "img/"

root = Tk()
root.resizable(1, 1)
root.title("Handwritten Cal")

lastx, lasty = None, None
image_number = 0

canvas1 = Canvas(root, width=960, height=280, bg='white')
canvas1.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)

canvas2 = Canvas(root, width=320, height=60, bg='white')
canvas2.grid(row=1, column=1, pady=2, sticky=W, columnspan=2)

def clear_widget():
    global canvas1, canvas2
    canvas1.delete('all')
    canvas2.delete('all')


def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    canvas1.create_line((lastx, lasty, x, y), width=8, fill='black', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y


def activate_event(event):
    global lastx, lasty
    canvas1.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y


canvas1.bind('<Button-1>', activate_event)


def Recognize_Digit():
    global image_number
    filename = f'img_{image_number}.png'
    widget = canvas1

    x = root.winfo_rootx() + widget.winfo_rootx() + 10
    y = root.winfo_rooty() + widget.winfo_rooty() + 10
    x1 = x + widget.winfo_width()*2 - 20
    y1 = y + widget.winfo_height()*2 - 20
    print(x, y, x1, y1)

    # 获取并保存图像
    ImageGrab.grab().crop((x, y, x1, y1)).save(image_folder + filename)

    image = Image.open(image_folder + filename).convert("L")

    elements_pred = predict_expression(image, model)
    result = calculate(elements_pred)

    canvas2.delete('all')
    titleFont = ("微软雅黑", 20, "bold")
    canvas2.create_text(10, 30,
                        text = result,
                        font = titleFont,
                        fill= "Turquoise",
                        anchor = W,
                        justify = LEFT)


btn_save = Button(text="识别算式", command=Recognize_Digit)
btn_save.grid(row=2, column=0, pady=1, padx=1)
button_clear = Button(text="清空", command=clear_widget)
button_clear.grid(row=2, column=1, pady=1, padx=1)

if __name__ == '__main__':
    root.mainloop()
