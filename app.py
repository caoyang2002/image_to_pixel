
usage = """
使用方法:
image_in 是原始图片文件
pix_out 是像素文件
输出的文件名固定为 out.png 新文件会覆盖旧文件
"""

tiler = """
import cv2
import numpy as np
import os
import sys
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
import math
import pickle
import conf
from time import sleep

COLOR_DEPTH = conf.COLOR_DEPTH
RESIZING_SCALES = conf.RESIZING_SCALES
PIXEL_SHIFT = conf.PIXEL_SHIFT
POOL_SIZE = conf.POOL_SIZE
OVERLAP_TILES = conf.OVERLAP_TILES
IMAGE_SCALE = conf.IMAGE_SCALE
def color_quantization(img, n_colors):
    return np.round(img / 255 * n_colors) / n_colors * 255
def read_image(path, mainImage=False):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    img = color_quantization(img.astype('float'), COLOR_DEPTH)

    if mainImage:
        
        img = cv2.resize(img, (0, 0), fx=IMAGE_SCALE, fy=IMAGE_SCALE)
    
    return img.astype('uint8')




def resize_image(img, ratio):
    
    img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))
    return img
def mode_color(img, ignore_alpha=False):
    
    counter = defaultdict(int)
    
    total = 0
    
    for y in img:
        
        for x in y:
            if len(x) < 4 or ignore_alpha or x[3] != 0:
                counter[tuple(x[:3])] += 1
            else:
                counter[(-1, -1, -1)] += 1
            total += 1
    if total > 0:
        
        mode_color = max(counter, key=counter.get)
        
        if mode_color == (-1, -1, -1):
            return None, None
        else:
            
            return mode_color, counter[mode_color] / total
    else:
        
        return None, None
def show_image(img, wait=True):
    
    cv2.imshow('img', img)
    
    if wait:
        
        cv2.waitKey(0)
    
    else:
        
        cv2.waitKey(1)
def load_tiles(paths):
    print('Loading tiles')
    tiles = defaultdict(list)
    for path in paths:
        
        if os.path.isdir(path):
            
            for tile_name in tqdm(os.listdir(path)):
                
                tile = read_image(os.path.join(path, tile_name))
                
                mode, rel_freq = mode_color(tile, ignore_alpha=True)
                
                if mode is not None:
                    
                    for scale in RESIZING_SCALES:
                        
                        t = resize_image(tile, scale)
                        
                        res = tuple(t.shape[:2])
                        
                        tiles[res].append({
                            'tile': t, 
                            'mode': mode, 
                            'rel_freq': rel_freq  
                        })

            
            with open('tiles.pickle', 'wb') as f:
                
                pickle.dump(tiles, f)     
        else:         
            with open(path, 'rb') as f:
                
                tiles = pickle.load(f)
    
    return tiles

def image_boxes(img, res):
    if not PIXEL_SHIFT:
        shift = np.flip(res)
    else:

        shift = PIXEL_SHIFT

    boxes = []
    
    for y in range(0, img.shape[0], shift[1]):
        
        for x in range(0, img.shape[1], shift[0]):
            
            boxes.append({
                
                'img': img[y:y + res[0], x:x + res[1]],
                
                'pos': (x, y)
            })

    
    return boxes
def color_distance(c1, c2):
    
    c1_int = [int(x) for x in c1]
    
    c2_int = [int(x) for x in c2]
    
    return math.sqrt((c1_int[0] - c2_int[0]) ** 2 + (c1_int[1] - c2_int[1]) ** 2 + (c1_int[2] - c2_int[2]) ** 2)
def most_similar_tile(box_mode_freq, tiles):
    
    if not box_mode_freq[0]:
        
        return (0, np.zeros(shape=tiles[0]['tile'].shape))
    else:
        
        min_distance = None
        
        min_tile_img = None
        
        for t in tiles:
            
            dist = (1 + color_distance(box_mode_freq[0], t['mode'])) / box_mode_freq[1]
            
            if min_distance is None or dist < min_distance:
                
                min_distance = dist
                
                min_tile_img = t['tile']
        
        return (min_distance, min_tile_img)
def get_processed_image_boxes(image_path, tiles):
    
    print('Getting and processing boxes')
    
    img = read_image(image_path, mainImage=True)
    
    pool = Pool(POOL_SIZE)
    
    all_boxes = []

    for res, ts in tqdm(sorted(tiles.items(), reverse=True)):
        
        boxes = image_boxes(img, res)
        
        modes = pool.map(mode_color, [x['img'] for x in boxes])
        
        most_similar_tiles = pool.starmap(most_similar_tile, zip(modes, [ts for x in range(len(modes))]))

        i = 0
        
        for min_dist, tile in most_similar_tiles:
            
            boxes[i]['min_dist'] = min_dist
            
            boxes[i]['tile'] = tile
            
            i += 1
        
        all_boxes += boxes
    
    return all_boxes, img.shape

def place_tile(img, box):
    
    p1 = np.flip(box['pos'])
    
    p2 = p1 + box['img'].shape[:2]
    
    img_box = img[p1[0]:p2[0], p1[1]:p2[1]]
    
    mask = box['tile'][:, :, 3] != 0
    
    mask = mask[:img_box.shape[0], :img_box.shape[1]]
    
    if OVERLAP_TILES or not np.any(img_box[mask]):
        
        img_box[mask] = box['tile'][:img_box.shape[0], :img_box.shape[1], :][mask]

def create_tiled_image(boxes, res, render=False):
    
    print('Creating tiled image')
    
    img = np.zeros(shape=(res[0], res[1], 4), dtype=np.uint8)

    
    for box in tqdm(sorted(boxes, key=lambda x: x['min_dist'], reverse=OVERLAP_TILES)):
        
        place_tile(img, box)
        
        if render:
            
            show_image(img, wait=False)
            
            sleep(0.025)
    
    return img

def main():
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = conf.IMAGE_TO_TILE
    if len(sys.argv) > 2:
        tiles_paths = sys.argv[2:]
    else:
        tiles_paths = conf.TILES_FOLDER.split(' ')

    if not os.path.exists(image_path):
        print('Image not found')
        exit(-1)
    for path in tiles_paths:
        if not os.path.exists(path):
            print('Tiles folder not found')
            exit(-1)
    tiles = load_tiles(tiles_paths)
    boxes, original_res = get_processed_image_boxes(image_path, tiles)
    img = create_tiled_image(boxes, original_res, render=conf.RENDER)
    cv2.imwrite(conf.OUT, img)
if __name__ == "__main__":
    main()
"""

gen_tiles = """
import cv2 
import numpy as np
import os
import sys
from tqdm import tqdm
import math
import conf

# DEPTH = 4 -> 4 * 4 * 4 = 64 colors
DEPTH = conf.DEPTH
# list of rotations, in degrees, to apply over the original image
ROTATIONS = conf.ROTATIONS

img_path = sys.argv[1]
img_dir = os.path.dirname(img_path)
img_name, ext = os.path.basename(img_path).rsplit('.', 1)
out_folder = img_dir + '/gen_' + img_name

if not os.path.exists(out_folder):
    os.mkdir(out_folder)

img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
img = img.astype('float')

height, width, channels = img.shape
center = (width/2, height/2)

for b in tqdm(np.arange(0, 1.01, 1 / DEPTH)):
    for g in np.arange(0, 1.01, 1 / DEPTH):
        for r in np.arange(0, 1.01, 1 / DEPTH):
            mult_vector = [b, g, r]
            if channels == 4:
                mult_vector.append(1)
            new_img = img * mult_vector
            new_img = new_img.astype('uint8')
            for rotation in ROTATIONS:
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1)
                abs_cos = abs(rotation_matrix[0,0])
                abs_sin = abs(rotation_matrix[0,1])
                new_w = int(height * abs_sin + width * abs_cos)
                new_h = int(height * abs_cos + width * abs_sin)
                rotation_matrix[0, 2] += new_w/2 - center[0]
                rotation_matrix[1, 2] += new_h/2 - center[1]
                cv2.imwrite(
                    f'{out_folder}/{img_name}_{round(r,1)}_{round(g,1)}_{round(b,1)}_r{rotation}.{ext}',
                    cv2.warpAffine(new_img, rotation_matrix, (new_w, new_h)),
                    # compress image
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])

"""


conf = """
DEPTH = 4
ROTATIONS = [0]

#############################

COLOR_DEPTH = 32  # 32
IMAGE_SCALE = 1
RESIZING_SCALES = [0.12] # [0.5, 0.4, 0.3, 0.2, 0.1]
PIXEL_SHIFT = None
OVERLAP_TILES = False
RENDER = False
POOL_SIZE = 8
OUT = r'./pix_out/out.png'
IMAGE_TO_TILE = r"image_in/bird.png"
TILES_FOLDER = r"tiles/block/gen_block"

"""


import os
import time
import tkinter as tk
from tkinter import messagebox, ttk, filedialog

from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk, ImageDraw
from tkinter import PhotoImage
import subprocess
import tkdnd

import shutil

import sys
import os

root = TkinterDnD.Tk()

IMAGE = r"./image_in/bird.png"
TILES = r"./tiles/block/gen_block/"


def check_and_install_python310():
    # 检查 Python 3.10 是否已经安装
    try:
        # 使用 PowerShell 执行命令，检查 Python 版本
        result = subprocess.check_output(['powershell', '-Command', 'python --version'], shell=False, encoding='utf-8')
        # 解析输出以确定 Python 版本
        version_str = result.strip().split()[1]  # 假设输出格式为 "Python 3.x.x"
        version_info = tuple(map(int, version_str.split('.')))
        if version_info >= (3, 10):
            print("Python 3.10 或更高版本已安装。")
        else:
            print(f"检测到 Python {version_info[0]}.{version_info[1]}，需要安装 Python 3.10 或更高版本。")
            install_python310()
    except subprocess.CalledProcessError:
        print("Python 3.10 未安装或无法启动。")
        install_python310()


def install_python310():
    print("正在尝试安装 Python 3.10...")
    # 下载 Python 3.10 安装程序
    installer_url = "https://www.python.org/ftp/python/3.10.0/Python-3.10.0-amd64.exe"
    installer_path = "Python-3.10.0-amd64.exe"
    download_cmd = ['powershell', '-Command', f'Invoke-WebRequest -Uri "{installer_url}" -OutFile "{installer_path}"']
    subprocess.run(download_cmd, check=True)

    # 启动安装程序
    install_cmd = ['start', '/wait', installer_path, '/quiet', 'InstallNow', 'CustomInstallString=/RegServer']
    subprocess.run(install_cmd, check=True)

    # 检查是否安装成功
    try:
        # 再次检查 Python 版本
        subprocess.check_output(['powershell', '-Command', 'python --version'], shell=False, encoding='utf-8')
        print("Python 3.10 安装成功。")
    except subprocess.CalledProcessError:
        print("Python 3.10 安装失败。")


# 调用函数
check_and_install_python310()








def convert_to_rgba(file_path):
    # 尝试将图像转换为 RGBA 格式并保存
    try:
        # 打开图像文件
        with Image.open(file_path) as img:
            # 转换为 RGBA 格式
            rgba_img = img.convert('RGBA')
            # 保存图像
            rgba_img.save(file_path, 'PNG')  # 通常 PNG 支持 RGBA 格式
            print(f"文件 '{file_path}' 已转换为 RGBA 格式。")
    except IOError as e:
        print(f"无法打开或转换图像文件: {e}")

# 复制文件并转换为 RGBA
def copy_and_convert_to_rgba(source_path, target_path):
    # 检查目标文件是否存在
    if os.path.exists(target_path):
        os.remove(target_path)
        print(f"已删除旧文件: {target_path}")

    # 复制文件到目标路径
    shutil.copy(source_path, target_path)
    print(f"已复制文件到: {target_path}")

    # 转换为目标文件的 RGBA 格式
    convert_to_rgba(target_path)





# 获取拖拽的文件路径

def on_drop(event):
    global IMAGE  # 使用 global 关键字声明全局变量 IMAGE
    file_path = event.data.strip('{}')  # 删除字符串前后的大括号
    # 打印拖拽的文件路径
    file_name = file_path.split("/")[-1]
    file_label.config(text="已获取到文件: " + file_name)

    # 获取当前程序所在目录的路径
    current_path = os.path.abspath(sys.argv[0])
    print(f"当前程序所在路径: {current_path}")
    current_directory = os.path.dirname(current_path)
    # current_directory = os.path.dirname(os.path.abspath(__file__))
    # 获取当前路径
    image_in_directory = os.path.join(current_directory, "image_in")


    # 如果 "image_in" 文件夹不存在，则创建它
    if not os.path.exists(image_in_directory):
        os.makedirs(image_in_directory)

    # 组合新的文件路径，将文件复制到 "image_in" 文件夹中
    new_file_path = os.path.join(image_in_directory, file_name)
    # 如果目标文件已存在，先删除它
    if os.path.exists(new_file_path):
        os.remove(new_file_path)
        print(f"已删除旧文件: {new_file_path}")
    copy_and_convert_to_rgba(file_path, new_file_path)
    # shutil.copy(file_path, new_file_path)
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
        image = Image.open("./pix_out/out.png")
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
    current_path = os.path.abspath(sys.argv[0])
    print(f"当前程序所在路径: {current_path}")
    current_directory = os.path.dirname(current_path)

    # 组合路径，打开当前程序所在目录下的 pix_out 文件夹
    folder_path = os.path.join(current_directory, "pix_out")
    os.system("explorer " + folder_path)
# 执行命令
def execute_command():
    # python tiler.py path/to/image path/to/tiles_folder/
    # python ./tiler.py ./image_in/bird.png ./tiles/block/gen_block/
    global IMAGE  # 使用 global 关键字声明全局变量 IMAGE
    command = r"./tiler.py " + IMAGE + " " + "./tiles/base/gen_block/"  # 获取输入框中的命令
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

# ========================================================================================= 更新配置文件
def update_conf_file():
    # 读取选项菜单的当前值
    overlap_tiles_value = OVERLAP_TILES.get()
    render_value = RENDER.get()
    resizing_scales_value = f"[{RESIZING_SCALES.get()}]"  # RESIZING_SCALES.get()
    image_scale_value = IMAGE_SCALE.get()
    color_depth_value = COLOR_DEPTH.get()
    # 根据选项菜单的值修改conf.py文件中的OVERLAP_TILES的值
    with open("conf.py", "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open("conf.py", "w", encoding="utf-8") as f:
        for line in lines:
            # 检查是否需要更新的配置项
            if line.startswith("OVERLAP_TILES"):
                f.write(f"OVERLAP_TILES = {overlap_tiles_value}\n")
            elif line.startswith("RENDER"):
                f.write(f"RENDER = {render_value}\n")
            elif line.startswith("RESIZING_SCALES"):
                f.write(f"RESIZING_SCALES = {resizing_scales_value}\n")
            elif line.startswith("IMAGE_SCALE"):
                f.write(f"IMAGE_SCALE = {image_scale_value}\n")
            elif line.startswith("COLOR_DEPTH"):
                f.write(f"COLOR_DEPTH = {color_depth_value}\n")
            else:
                # 对于不需要更新的行，直接写入原始行
                f.write(line)

    # 执行命令来重新加载配置文件
    subprocess.run(["python", "conf.py"])

def on_option_select(event):
    # 当选项菜单的值发生变化时，更新conf.py文件
    update_conf_file()

def create_folders_and_write_files():
    check_and_install_python310()
    # 定义文件夹名称和要写入的文件名及内容
    folder_names = ['image_in', 'pix_out','tiles','tiles/base','tiles/base/gen_block']
    file_names = ['tiler.py', 'conf.py', '使用方法.txt', "gen_tiles.py"]
    file_contents = [tiler, conf, usage, gen_tiles]

    # 检查并创建文件夹
    for folder in folder_names:
        folder_path = os.path.join(os.getcwd(), folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"创建了文件夹：{folder}")
        else:
            print(f"文件夹已存在：{folder}")

    # 创建文件并写入内容
    for file_name, content in zip(file_names, file_contents):
        file_path = os.path.join(os.getcwd(), file_name)  # 文件位于当前工作目录
        if not os.path.isfile(file_path):
            with open(file_path, 'w') as file:
                file.write(content)
                print(f"文件 {file_name} 已创建并写入内容。")
        else:
            print(f"文件 {file_name} 已存在。")
    # 初始化像素块文件
    # python ./gen_tiles.py your/image/path`, 例如: `python ./gen_tiles.py ./tiles/block/block.png`
    # 脚本 `gen_tiles.py` 可以在这项任务中提供帮助；它基于源像素块构建具有多种颜色的像素块（注意：推荐源文件具有 RGB 颜色值 (240,240,240)）。使用方式为 `python gen_tiles.py path/to/image`，它会在基础图像的相同路径下创建一个带有 'gen_' 前缀的文件夹。
    # - 生成像素图片

    # 设置图像的大小
    width, height = 100, 100

    # 创建一个100x100像素的白色背景图像
    image = Image.new('RGB', (width, height), color='white')
    # 保存图像到文件
    image.save('tiles/base/block.png', 'PNG')

    # 创建
    command =  "./gen_tiles.py " + "./tiles/base/block.png"  # 获取输入框中的命令
    print(f"command is: {command}")
    try:
        output = subprocess.check_output(['powershell', 'python', command], shell=True,
                                         encoding='utf-8')  # 使用 PowerShell 执行命令
        print(output)
    except subprocess.CalledProcessError as e:
        # result_text.insert(tk.END, "错误: " + str(e))
        print(e)




print("正在启动, 请稍后...")

# 设置窗口标题
root.title("像素画")
# 设置窗口大小
root.geometry("1080x960")


logo_data = b"iVBORw0KGgoAAAANSUhEUgAAAR4AAAETCAYAAADgV2kIAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAehSURBVHhe7d1Pi53VAcfx9tX0LcTFuBC6deOu3bqvFhczq0JegiAEEcRFJa78L2JFXARjCAX/oZEIGoPWGCySCdGocLzSWw63HHmeK+f85pmnny98ljPDZZ7zW5177++KJIUzPJLiGR5J8QyPpHiGR1I8wyMpnuGRFM/wSIpneCTFMzyS4hkeSfEMj6R4hkdSPMMjKZ7hkRTP8EiKZ3gkxTM8kuIZHknxDI+keH2H5957Szk4gOU5Oto+pKHu3i7lcHO81uzKK9sXu3+bn+6Y4WGpDE9/hgcmGJ7+DA9MMDz9GR6YYHj6MzwwwfD0Z3hgguHpz/DABMPTn+GBCYanP8MDEwxPf4YHJhie/gwPTDA8/RkemGB4+jM8MMHw9Gd4YILh6c/wwATD099ihufsWVim8+e3D2moH7/f/M0/r9uX72xf7P71HR5JmpHhkRTP8EiKZ3gkxTM8kuIZHknxDI+keIZHUry+w+PmMkv32IOlvPrIeC//pX3bd028ZQJmMjz9GB6YyfD0Y3hgJsPTj+GBmQxPP4YHZjI8/RgemMnw9GN4YCbD04/hgZkMTz+GB2YyPP0YHpjJ8PRjeGAmw9OP4YGZDE8/hgdmMjz9GB6YyfD0Y3hgJsPTz2KG59FHYdkunC/lo+fH+/DZUl58aN1ufLg9+PvXd3gkaUaGR1I8wyMpnuGRFM/wSIpneCTFMzyS4mWH56fvS/nhDnAaDCw7PJfPtW95AsszMMMDtA3M8ABtAzM8QNvADA/QNjDDA7QNzPAAbQMzPEDbwAwP0DYwwwO0DczwAG0DMzxA28AMD9A2MMMDtA3M8ABtAzM8QNvAssPzxT9L+eR14DQYWHZ4JGmT4ZEUz/BIimd4JMUzPJLiGR5J8QyPpHh9h+fmx+v27fVSbn+dcfMaLNvd3/6lf32H5+j3pRxufuVaPXFf+4bnCIdnYNmuvLU9+Pu3OU0dMzz9tP7RsCSGJ8TwQGV4QgwPVIYnxPBAZXhCDA9UhifE8EBleEIMD1SGJ8TwQGV4QgwPVIYnxPBAZXhCDA9UhifE8EBleEIMD1SGJ8TwQGV4QgwPVIsZnkuPr9uVl0q5/nbGpedg2b79anvw96/v8EjSjAyPpHiGR1I8wyMpnuGRFM/wSIpneCTF6zs8n11Yt6/eL+Xfn2Z89i4s251b24O/f32Hx83lflo3RWFJvGUixPBAZXhCDA9UhifE8EBleEIMD1SGJ8TwQGV4QgwPVIYnxPBAZXhCDA9UhifE8EBleEIMD1SGJ8TwQGV4QgwPVIYnxPBAZXhCDA9UhifE8EC1mOH54Nl1+++HgSV88CYs2/E324O/f32HR5JmZHgkxTM8kuIZHknxDI+keIZHUjzDIylednh++aK61p0VYHkGlh2ey+fat3SB5RmY4QHaBmZ4gLaBGR6gbWCGB2gbmOEB2gZmeIC2gRkeoG1ghgdoG5jhAdoGZniAtoEZHqBtYIYHaBuY4QHaBmZ4gLaBGR6gbWDZ4Tn+138+DAxYvoFlh0eSNhkeSfEMj6R4hkdSPMMjKZ7hkRTP8EiK13d4Lj0HC/XMxuOlXP1HKdffHu/axVJeeGHdbtzYHvz96zs8R/eUcngGFugPG5vH/ekH2rd0e3vxr6UcHKzbxc24/sYMD/8nDE93hgemGJ7uDA9MMTzdGR6YYni6MzwwxfB0Z3hgiuHpzvDAFMPTneGBKYanO8MDUwxPd4YHphie7gwPTDE83RkemGJ4ujM8MMXwdGd4YIrh6c7wwBTD091ihufmNVioTzc+LuXWF6Xc/nq8443PP1+3777bHvz96zs8kjQjwyMpnuGRFM/wSIpneCTFMzyS4hkeSfHGDc87r5XyxpNw8j55/eQ98/dSnnrq9OvUuOF58uH/uTkKJ6R1szjt/j+2b/+eNp0yPKxfawjSDM9Ohof1aw1BmuHZyfCwfq0hSDM8Oxke1q81BGmGZyfDw/q1hiDN8OxkeFi/1hCkGZ6dDA/r1xqCNMOzk+Fh/VpDkGZ4djI8rF9rCNIMz06Gh/VrDUGa4dnJ8LB+rSFIMzw7GR7WrzUEaYZnJ8PD+rWGIM3w7GR4WL/WEKQZnp0MD+vXGoI0w7PTuOG5e6eUO7fg5P2weRZP2u3jUo5XoFPjhkeSfiXDIyme4ZEUz/BIimd4JMUzPJLiGR5J8bLDc+1CKR89D5wGA8sOz+Vz7VudwPIMzPAAbQMzPEDbwAwP0DYwwwO0DczwAG0DMzxA28AMD9A2MMMDtA3M8ABtAzM8QNvADA/QNjDDA7QNzPAAbQMzPEDbwAwP0Daw7PBI0ibDIyme4ZEUz/BIimd4JMUzPJLiGR5J8QyPpHh9h+f839btzcdKee/pjLNnYdmuXt0e/P3rOzxH95RyeGa9nvhT+4bnCAcHsGwXL24P/v4Znn0YHqgMT4jhgcrwhBgeqAxPiOGByvCEGB6oDE+I4YHK8IQYHqgMT4jhgcrwhBgeqAxPiOGByvCEGB6oDE+I4YHK8IQYHqgMT4jhgcrwhBgeqBYzPJI0I8MjKZ7hkRTP8EiKZ3gkxTM8kuIZHknxDI+keIZHUjzDIyme4ZEUz/BIimd4JMUzPJLClfIzTSUxF2L1THUAAAAASUVORK5CYII="
logo_image = PhotoImage(data=logo_data)

root.iconphoto(True, logo_image)

# 启动时检查并创建文件夹
create_folders_and_write_files()


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

entry_note_label1 = ttk.Label(root, text="构成像素图的颜色数量, 任意整数",   style="Note.TLabel")
entry_note_label1.grid(row=1, column=1, padx=0, pady=0)
COLOR_DEPTH = EntryWithPlaceholder(root, placeholder="例如: 2/4/8/16/32...", width=30)
COLOR_DEPTH.grid(row=2, column=1, padx=10, pady=10)

# 创建一个 EntryWithPlaceholder 元素
entry_note_label1 = ttk.Label(root, text="原始图像的缩放比例, 任意整数",   style="Note.TLabel")
entry_note_label1.grid(row=3, column=1, padx=0, pady=0)
IMAGE_SCALE = EntryWithPlaceholder(root, placeholder="例如: 0.5/1/2/4...", width=30)
IMAGE_SCALE.grid(row=4, column=1,  padx=10, pady=10)


entry_note_label1 = ttk.Label(root, text="像素的缩放比例, 数值越大,像素块越大, 0~1 的小数",   style="Note.TLabel")
entry_note_label1.grid(row=5, column=1, padx=0, pady=0)
RESIZING_SCALES = EntryWithPlaceholder(root, placeholder="例如: 0.1/0.3,0.2,0.1/0.8,0.6,0.4,0.2", width=30)
RESIZING_SCALES.grid(row=6, column=1, padx=10, pady=10)


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
RENDER = tk.StringVar(root)
RENDER.set("选择")
option_menu2 = tk.OptionMenu(root, RENDER, "False", "True")
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


option_label2 = ttk.Label(root, text="如果图片太大, 则会等待很长时间", style="Note.TLabel")
option_label2.grid(row=14, column=1, padx=10, pady=10)
# 添加一个结果文本框
result_text = tk.Text(root, width=50, height=10)
result_text.grid(row=15, column=0, columnspan=4)



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
    if os.path.exists("./pix_out/out.png"):
        show_pix()
    root.after(1000, check_image_existence)  # 每隔一秒钟检查一次

check_image_existence()

root.mainloop()
