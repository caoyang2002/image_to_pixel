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


# number of colors per image
# 每张图片的颜色数量
COLOR_DEPTH = conf.COLOR_DEPTH
# tiles scales
# 像素块的缩放比例
RESIZING_SCALES = conf.RESIZING_SCALES
# number of pixels shifted to create each box (x,y)
# 为了创建每个盒子（x,y）而偏移的像素数量
PIXEL_SHIFT = conf.PIXEL_SHIFT
# multiprocessing pool size
# 多进程池的大小
POOL_SIZE = conf.POOL_SIZE
# if tiles can overlap
# 像素块是否可以重叠
OVERLAP_TILES = conf.OVERLAP_TILES


# reduces the number of colors in an image
def color_quantization(img, n_colors):
    return np.round(img / 255 * n_colors) / n_colors * 255


# returns an image given its path
# 根据给定的路径返回图像
def read_image(path):
    # 使用cv2.imread函数读取图像，不进行任何颜色转换
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # 将图像数据类型转换为浮点数，进行颜色量化处理，然后转换回uint8类型
    img = color_quantization(img.astype('float'), COLOR_DEPTH)
    return img.astype('uint8')


# scales an image
# 对图像进行缩放
def resize_image(img, ratio):
    # 使用cv2.resize函数根据给定的比例缩放图像
    img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))
    return img


# the most frequent color in an image and its relative frequency
# 计算图像中最频繁出现的颜色及其相对频率
def mode_color(img):
    # 使用defaultdict来计数每种颜色的出现次数
    counter = defaultdict(int)
    # 记录图像中非透明像素的总数
    total = 0
    # 遍历图像的每一行
    for y in img:
        # 遍历行中的每个像素
        for x in y:
            # 如果像素的alpha通道不为0，或者图像没有alpha通道（即长度小于4）
            if len(x) < 4 or x[3] != 0:
                # 将RGB值作为键，增加计数
                counter[tuple(x[:3])] += 1
            else:
                # 记录完全透明的像素
                counter[(-1, -1, -1)] += 1
            # 增加总像素数
            total += 1
    # 如果有非透明像素
    if total > 0:
        # 找到出现次数最多的颜色
        mode_color = max(counter, key=counter.get)
        # 如果最多的是透明像素，则返回None
        if mode_color == (-1, -1, -1):
            return None, None
        else:
            # 返回最频繁的颜色和其在图像中的相对频率
            return mode_color, counter[mode_color] / total
    else:
        # 如果没有非透明像素，也返回None
        return None, None


# displays an image
# 显示图像的函数
def show_image(img, wait=True):
    # 使用OpenCV的imshow函数在窗口中显示图像
    cv2.imshow('img', img)
    # 如果wait参数为True
    if wait:
        # 等待用户按键，程序暂停，直到有按键输入
        cv2.waitKey(0)
    # 如果wait参数为False
    else:
        # 等待1ms，以便实现图像的短暂显示
        cv2.waitKey(1)


# load and process the tiles
# 加载和处理瓦片的函数
def load_tiles(paths):
    # 打印加载瓦片的提示信息
    print('Loading tiles')
    # 使用defaultdict，便于向tiles中添加瓦片数据
    tiles = defaultdict(list)

    # 对于传入的每个路径
    for path in paths:
        # 如果路径是一个目录
        if os.path.isdir(path):
            # 遍历目录下的所有文件
            for tile_name in tqdm(os.listdir(path)):
                # 读取每个瓦片图像
                tile = read_image(os.path.join(path, tile_name))
                # 获取瓦片的主要颜色和相对频率
                mode, rel_freq = mode_color(tile)
                # 如果获取到的主要颜色不是None
                if mode != None:
                    # 遍历所有的缩放比例
                    for scale in RESIZING_SCALES:
                        # 根据比例缩放瓦片
                        t = resize_image(tile, scale)
                        # 获取缩放后瓦片的分辨率
                        res = tuple(t.shape[:2])
                        # 将缩放后的瓦片添加到tiles字典中
                        tiles[res].append({
                            'tile': t,
                            'mode': mode,
                            'rel_freq': rel_freq
                        })

            # 将tiles数据持久化存储到pickle文件
            with open('tiles.pickle', 'wb') as f:
                # 使用pickle.dump序列化tiles对象并写入文件
                pickle.dump(tiles, f)

        # load pickle with tiles (one file only)
        # 如果路径不是目录，即是一个文件
        else:
            # 打开pickle文件进行读取
            with open(path, 'rb') as f:
                # 使用pickle.load反序列化文件内容到tiles变量
                tiles = pickle.load(f)
    # 返回tiles对象，包含了所有加载和处理后的瓦片数据
    return tiles


# returns the boxes (image and start pos) from an image, with 'res' resolution
# 从图像中返回盒子（区域）的函数，具有指定分辨率'res'
def image_boxes(img, res):
    # 如果PIXEL_SHIFT配置为False或0
    if not PIXEL_SHIFT:
        # 将分辨率翻转作为偏移量
        shift = np.flip(res)
    else:
        # 否则使用PIXEL_SHIFT作为偏移量
        shift = PIXEL_SHIFT

    # 初始化盒子列表
    boxes = []
    # 在图像的Y轴上遍历，步长为shift[1]
    for y in range(0, img.shape[0], shift[1]):
        # 在图像的X轴上遍历，步长为shift[0]
        for x in range(0, img.shape[1], shift[0]):
            # 将遍历到的每个区域添加到boxes列表中
            boxes.append({
                # 区域图像
                'img': img[y:y + res[0], x:x + res[1]],
                # 区域在原图中的起始位置
                'pos': (x, y)
            })

    # 返回包含所有盒子的列表
    return boxes


# euclidean distance between two colors
# 计算两种颜色之间的欧几里得距离
def color_distance(c1, c2):
    # 将颜色c1的每个分量转换为整数
    c1_int = [int(x) for x in c1]
    # 将颜色c2的每个分量转换为整数
    c2_int = [int(x) for x in c2]
    # 计算颜色c1和c2之间的欧几里得距离，并返回
    return math.sqrt((c1_int[0] - c2_int[0]) ** 2 + (c1_int[1] - c2_int[1]) ** 2 + (c1_int[2] - c2_int[2]) ** 2)


# returns the most similar tile to a box (in terms of color)
# 返回与一个盒子在颜色上最相似的瓦片
def most_similar_tile(box_mode_freq, tiles):
    # 如果盒子的主要颜色不存在
    if not box_mode_freq[0]:
        # 返回零距离和空白图像
        return (0, np.zeros(shape=tiles[0]['tile'].shape))
    else:
        # 初始化最小距离为None
        min_distance = None
        # 初始化最相似瓦片图像为None
        min_tile_img = None
        # 对于tiles列表中的每个瓦片
        for t in tiles:
            # 计算与盒子颜色的相似度距离
            dist = (1 + color_distance(box_mode_freq[0], t['mode'])) / box_mode_freq[1]
            # 如果当前距离更小
            if min_distance == None or dist < min_distance:
                # 更新最小距离
                min_distance = dist
                # 更新最相似的瓦片图像
                min_tile_img = t['tile']
        # 返回最小距离和最相似的瓦片图像
        return (min_distance, min_tile_img)


# builds the boxes and finds the best tile for each one
# 构建盒子并为每个盒子找到最佳瓦片
def get_processed_image_boxes(image_path, tiles):
    # 打印提示信息
    print('Gettting and processing boxes')
    # 读取图像
    img = read_image(image_path)
    # 创建多进程池
    pool = Pool(POOL_SIZE)
    # 初始化所有盒子的列表
    all_boxes = []

    # 对tiles字典按键（分辨率）降序排序后进行遍历
    for res, ts in tqdm(sorted(tiles.items(), reverse=True)):
        # 为当前分辨率生成所有盒子
        boxes = image_boxes(img, res)
        # 多进程计算每个盒子的主要颜色
        modes = pool.map(mode_color, [x['img'] for x in boxes])
        # 多进程找到每个盒子最相似的瓦片
        most_similar_tiles = pool.starmap(most_similar_tile, zip(modes, [ts for x in range(len(modes))]))

        # 初始化盒子索引
        i = 0
        # 对于每个最相似的瓦片
        for min_dist, tile in most_similar_tiles:
            # 更新盒子的最小距离
            boxes[i]['min_dist'] = min_dist
            # 更新盒子的最相似瓦片
            boxes[i]['tile'] = tile
            # 盒子索引加1
            i += 1
        # 将当前分辨率的所有盒子添加到总列表中
        all_boxes += boxes
    # 返回所有盒子和图像的形状
    return all_boxes, img.shape


# places a tile in the image
# 将瓦片放置到图像中
def place_tile(img, box):
    # 获取盒子的起始位置（翻转位置元组）
    p1 = np.flip(box['pos'])
    # 获取盒子的结束位置（起始位置+盒子图像的尺寸）
    p2 = p1 + box['img'].shape[:2]
    # 创建一个布尔掩码，其中alpha通道非零的地方为True
    mask = box['tile'][:, :, 3] != 0
    # 获取当前盒子在原图中的位置对应的图像区域
    img_box = img[p1[0]:p2[0], p1[1]:p2[1]]
    # 将掩码大小调整为与图像区域匹配
    mask = mask[:img_box.shape[0], :img_box.shape[1]]
    # 如果允许瓦片重叠，或者该区域没有其他瓦片
    if OVERLAP_TILES or not np.any(img_box[mask]):
        # 将瓦片放置到图像区域中
        img_box[mask] = box['tile'][:img_box.shape[0], :img_box.shape[1], :][mask]


# tiles the image
# 使用给定的盒子集合创建平铺图像
def create_tiled_image(boxes, res, render=False):
    # 打印提示信息，开始创建平铺图像
    print('Creating tiled image')
    # 创建一个黑色的空白图像数组，4表示RGBA颜色通道
    img = np.zeros(shape=(res[0], res[1], 4), dtype=np.uint8)

    # 遍历盒子列表，盒子根据最小距离排序，如果允许瓦片重叠，则逆序排列
    for box in tqdm(sorted(boxes, key=lambda x: x['min_dist'], reverse=OVERLAP_TILES)):
        # 将当前盒子的瓦片放置到图像中
        place_tile(img, box)
        # 如果设置为渲染模式
        if render:
            # 显示当前的平铺图像
            show_image(img, wait=False)
            # 暂停0.025秒，以便观察
            sleep(0.025)
    # 返回最终的平铺图像
    return img


# main
# 主函数
def main():
    # 检查命令行参数，获取要平铺的图像路径
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # 从配置文件中获取图像路径
        image_path = conf.IMAGE_TO_TILE
    # 检查命令行参数，获取瓦片路径列表
    if len(sys.argv) > 2:
        # 从命令行参数中获取瓦片路径
        tiles_paths = sys.argv[2:]
    else:
        # 从配置文件中获取瓦片路径列表
        tiles_paths = conf.TILES_FOLDER.split(' ')

    # 检查图像路径是否存在
    if not os.path.exists(image_path):
        # 如果不存在，打印错误信息并退出程序
        print('Image not found')
        exit(-1)
    # 检查所有瓦片路径是否存在
    for path in tiles_paths:
        if not os.path.exists(path):
            # 如果瓦片路径不存在，打印错误信息并退出程序
            print('Tiles folder not found')
            exit(-1)
    # 加载和处理瓦片
    tiles = load_tiles(tiles_paths)
    # 获取处理后的盒子和原始图像分辨率
    boxes, original_res = get_processed_image_boxes(image_path, tiles)
    # 使用盒子创建平铺图像
    img = create_tiled_image(boxes, original_res, render=conf.RENDER)
    # 将最终的平铺图像写入文件
    cv2.imwrite(conf.OUT, img)

# 程序入口点
if __name__ == "__main__":
    # 如果直接运行此脚本，则调用主函数
    main()