import os
import time
import tkinter as tk
from tkinter import messagebox, ttk, filedialog

from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk

import subprocess
import tkdnd

import shutil

root = TkinterDnD.Tk()

IMAGE = r"./image_in/bird.png"
TILES = r"./tiles/block/gen_block/"


# 获取拖拽的文件路径

def on_drop(event):
    global IMAGE  # 使用 global 关键字声明全局变量 IMAGE
    file_path = event.data.strip('{}')  # 删除字符串前后的大括号
    # 打印拖拽的文件路径
    file_name = file_path.split("/")[-1]
    file_label.config(text="已获取到文件: " + file_name)

    # 获取当前程序所在目录的路径
    current_directory = os.path.dirname(os.path.abspath(__file__))
    image_in_directory = os.path.join(current_directory, "image_in")

    # 如果 "image_in" 文件夹不存在，则创建它
    if not os.path.exists(image_in_directory):
        os.makedirs(image_in_directory)

    # 组合新的文件路径，将文件复制到 "image_in" 文件夹中
    new_file_path = os.path.join(image_in_directory, file_name)
    shutil.copy(file_path, new_file_path)
    print("已复制文件到:", new_file_path)

    IMAGE = new_file_path
    print("拖放的文件路径:", new_file_path)

    # 如果拖放的是图片，则显示图片
    if new_file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        show_image(new_file_path)
    else:
        messagebox.showinfo("提示", "拖放的不是图片文件！")


# 显示图片
def show_image(file_path):
    image = Image.open(file_path)
    image.thumbnail((460, 460))  # 缩放图片大小
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo



# 加载图片并显示在 Label 中
def show_pix():
    try:
        image = Image.open("out.png")
        image = image.resize((460, 460))  # 调整图片大小以适应标签
        photo = ImageTk.PhotoImage(image)
        pix_image.config(image=photo)
        pix_image.image = photo  # 防止被垃圾回收
    except FileNotFoundError:
        # 如果没有图片，显示空白
        pix_image.config(image=None)

# def execute_command_with_timer():
#     start_time = time.time()  # 记录开始时间
#     # python tiler.py path/to/image path/to/tiles_folder/
#     # python ./tiler.py ./image_in/bird.png ./tiles/block/gen_block/
#     global IMAGE  # 使用 global 关键字声明全局变量 IMAGE
#     command = r"./tiler.py " + IMAGE + " " + "./tiles/block/gen_block/"  # 获取输入框中的命令
#     print(f"command is: {command}")
#     try:
#         output = subprocess.check_output(['powershell', 'python', command], shell=True,
#                                          encoding='utf-8')  # 使用 PowerShell 执行命令
#         result_text.delete('1.0', tk.END)  # 清空结果文本框
#         result_text.insert(tk.END, output)  # 将命令输出显示到结果文本框
#     except subprocess.CalledProcessError as e:
#         result_text.delete('1.0', tk.END)  # 清空结果文本框
#         result_text.insert(tk.END, "错误: " + str(e))
#     end_time = time.time()  # 记录结束时间
#     elapsed_time = end_time - start_time  # 计算执行时间
#     time_var.set("命令执行时间：{}秒".format(elapsed_time))  # 更新标签文本内容


def open_folder():
    # 获取当前程序所在目录的路径
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # 组合路径，打开当前程序所在目录下的 pix_out 文件夹
    folder_path = os.path.join(current_directory, "pix_out")
    os.system("explorer " + folder_path)
# 执行命令
def execute_command():
    # python tiler.py path/to/image path/to/tiles_folder/
    # python ./tiler.py ./image_in/bird.png ./tiles/block/gen_block/
    global IMAGE  # 使用 global 关键字声明全局变量 IMAGE
    command = r"./tiler.py " + IMAGE + " " + "./tiles/block/gen_block/"  # 获取输入框中的命令
    print(f"command is: {command}")
    try:
        output = subprocess.check_output(['powershell', 'python', command], shell=True, encoding='utf-8')  # 使用 PowerShell 执行命令
        result_text.delete('1.0', tk.END)  # 清空结果文本框
        result_text.insert(tk.END, output)  # 将命令输出显示到结果文本框
    except subprocess.CalledProcessError as e:
        result_text.delete('1.0', tk.END)  # 清空结果文本框
        result_text.insert(tk.END, "错误: " + str(e))


# def on_entry_click(event):
#     if entry.get() == "像素的缩放比例(0~1)":
#         entry.delete(0, "end") # 删除Entry中的所有文本
#         entry.config(fg="black") # 更改文本颜色为黑色
#
# def on_focus_out(event):
#     if entry.get() == "":
#         entry.insert(0, "像素的缩放比例(0~1)") # 在Entry中插入占位文本
#         entry.config(fg="grey") # 更改文本颜色为灰色


class EntryWithPlaceholder(tk.Entry):
    def __init__(self, master=None, placeholder="", placeholder_color="grey", *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.placeholder = placeholder
        self.placeholder_color = placeholder_color
        self.default_fg_color = self["fg"]
        self.bind("<FocusIn>", self.on_entry_click)
        self.bind("<FocusOut>", self.on_focus_out)
        self.insert_placeholder()

    def insert_placeholder(self):
        self.insert(0, self.placeholder)
        self.config(fg=self.placeholder_color)

    def on_entry_click(self, event):
        if self.get() == self.placeholder:
            self.delete(0, "end")
            self.config(fg=self.default_fg_color)

    def on_focus_out(self, event):
        if self.get() == "":
            self.insert_placeholder()


def update_conf_file():
    # 读取选项菜单的当前值
    overlap_tiles_value = OVERLAP_TILES.get()
    # 根据选项菜单的值修改conf.py文件中的OVERLAP_TILES的值
    with open("conf.py", "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open("conf.py", "w", encoding="utf-8") as f:
        for line in lines:
            if line.startswith("OVERLAP_TILES"):
                # 如果当前行包含OVERLAP_TILES，就更新它的值
                line = f"OVERLAP_TILES = {overlap_tiles_value}\n"
            f.write(line)

    # 执行命令来重新加载配置文件
    subprocess.run(["python", "conf.py"])

def on_option_select(event):
    # 当选项菜单的值发生变化时，更新conf.py文件
    update_conf_file()


# 设置窗口标题
root.title("生成像素图")
# 设置窗口大小
root.geometry("1080x960")


# 创建样式
style = ttk.Style()

# 设置标签的样式
style.configure('TitleLabel.TLabel', font=('微软雅黑', 14), foreground='#333')

# 设置按钮的样式
style.configure('Custom.TButton', font=('微软雅黑', 12, 'bold'), foreground='white', background='#4CAF50')

# 设置选择器的样式
style.configure('TMenubutton', font=('思源黑体', 12, 'bold'), foreground='blue', padding=5)

# 提示
style.configure("Note.TLabel", font=('思源黑体', 8), foreground='gray', padding=5)
# 类别
style.configure("Title.TLabel", font=('思源黑体', 12, 'bold'), foreground='gray', padding=5)


# 设置拖拽功能
file_label = ttk.Label(root, text="请把图片拖拽到这里", style="Title.TLabel")
file_label.grid(row=0, column=0, padx=10, pady=10)

image_label = tk.Label(root)
image_label.grid(row=1, column=0, padx=10, pady=10, rowspan=12)  # 行,列, 跨 6 行

root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', on_drop)





config_label = ttk.Label(root, text="配置",  style="Title.TLabel")
config_label.grid(row=0, column=1, padx=0, pady=0)

# 添加输入框和按钮
# 提示信息
entry_label1 = ttk.Label(root, text="像素的缩放比例(0~1)",  style="TMenubutton")
entry_label1.grid(row=2, column=1, padx=0, pady=0)
entry_note_label1 = ttk.Label(root, text="像素的缩放比例(0~1)",   style="Note.TLabel")
entry_note_label1.grid(row=3, column=1, padx=0, pady=0)
# 输入框
# entry = tk.Entry(root, width=50)
# entry.grid(row=2, column=2, columnspan=2, padx=10, pady=10)

# 创建一个 Entry 元素
# entry = tk.Entry(root, width=20, fg="grey")
# entry.insert(0, "像素的缩放比例(0~1)")  # 插入占位文本
# entry.bind("<FocusIn>", on_entry_click)  # 绑定获得焦点事件
# entry.bind("<FocusOut>", on_focus_out)  # 绑定失去焦点事件
# entry.grid(row=7, column=1,  padx=10, pady=10)


entry2 = EntryWithPlaceholder(root, placeholder="像素颜色数量", width=20)
entry2.grid(row=4, column=1, padx=10, pady=10)

# 创建一个 EntryWithPlaceholder 元素
entry1 = EntryWithPlaceholder(root, placeholder="像素的缩放比例: 0~1", width=20)
entry1.grid(row=5, column=1,  padx=10, pady=10)



entry3 = EntryWithPlaceholder(root, placeholder="要平铺的图像的缩放比例: [0~1]", width=25)
entry3.grid(row=6, column=1, padx=10, pady=10)


# 添加两个选择器
# 像素块是否可以重叠
option_label1 = ttk.Label(root, text="像素块是否可以重叠", style="Note.TLabel")
option_label1.grid(row=7, column=1)
OVERLAP_TILES = tk.StringVar(root)
OVERLAP_TILES.set("选择")
option_menu1 = tk.OptionMenu(root, OVERLAP_TILES, "True", "False", command=on_option_select)
option_menu1.grid(row=8, column=1, padx=10, pady=10)


# 显示渲染过程
option_label2 = ttk.Label(root, text="显示渲染过程", style="Note.TLabel")
option_label2.grid(row=9, column=1, padx=10, pady=10)
option_var2 = tk.StringVar(root)
option_var2.set("选择")
option_menu2 = tk.OptionMenu(root, option_var2, "False", "True")
option_menu2.grid(row=10, column=1, padx=10, pady=10)


# 按钮信息
entry_label1 = ttk.Label(root, text="点击按钮开始生成像素图",  style="Note.TLabel")
entry_label1.grid(row=11, column=1, padx=0, pady=0)

# 按钮
execute_button = tk.Button(root, text="开始生成", command=execute_command)
execute_button.grid(row=12, column=1, padx=10, pady=10)



# 按钮
execute_button = tk.Button(root, text="打开像素图文件夹", command=open_folder)
execute_button.grid(row=13, column=1, padx=10, pady=10)



# 按钮
# execute_button = tk.Button(root, text="开始生成", command=execute_command_with_timer)
# execute_button.grid(row=12, column=1, padx=10, pady=10)

# 显示执行时间的标签
# time_var = tk.StringVar()  # 创建一个StringVar
# time_label = tk.Label(root, textvariable=time_var, font=("Helvetica", 12))
# time_label.grid(row=14, column=1, padx=10, pady=10)



# 添加一个结果文本框
result_text = tk.Text(root, width=50, height=10)
result_text.grid(row=14, column=0, columnspan=4)



# =============================================================================================================================
# 创建一个标签来显示生成的像素图
pix_label = ttk.Label(root, text="生成的像素图", style="Title.TLabel")
pix_label.grid(row=0, column=3, padx=10, pady=10)
# pix_label.config(width=200, height=200)  # 设置标签大小
pix_image = ttk.Label(root, text="")
pix_image.grid(row=1, column=3, padx=10, pady=10,rowspan=12)
# 调用函数以显示图片
show_pix()

# 禁止窗口自动调整大小
root.pack_propagate(False)
# 更新窗口小部件
root.update_idletasks()
# 设置窗口大小以适应内容
# root.geometry(f"{root.winfo_reqwidth()}x{root.winfo_reqheight()}")



# 定义一个定时器函数，每隔一段时间检查一次文件是否存在并重新加载图片
def check_image_existence():
    if os.path.exists("out.png"):
        show_pix()
    root.after(1000, check_image_existence)  # 每隔一秒钟检查一次
check_image_existence()

root.mainloop()

