# import cv2
# import numpy as np
# import os
# import sys
# from tqdm import tqdm
# import math
# import conf
#
# # DEPTH = 4 -> 4 * 4 * 4 = 64 colors
# DEPTH = conf.DEPTH
# # list of rotations, in degrees, to apply over the original image
# ROTATIONS = conf.ROTATIONS
# # 从命令行参数中获取图像路径
# img_path = sys.argv[1]
# # 获取图像所在的目录
# img_dir = os.path.dirname(img_path)
# # 从图像路径中提取图像名称和扩展名
# img_name, ext = os.path.basename(img_path).rsplit('.', 1)
# # 构造输出文件夹的路径，该文件夹将包含生成的图像
# out_folder = img_dir + '/gen_' + img_name
#
# # 检查输出文件夹是否存在，如果不存在则创建它
# if not os.path.exists(out_folder):
#     os.mkdir(out_folder)
#
#
# # 使用cv2.imread读取图像，cv2.IMREAD_UNCHANGED 标志保留图像的透明度（如果存在）
# img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
# # 将图像数据类型转换为浮点数，以便进行数学运算
# img = img.astype('float')
#
# # 获取图像的高度、宽度和通道数
# height, width, channels = img.shape
# # 计算图像的中心点
# center = (width/2, height/2)
#
# # 使用三个嵌套循环遍历蓝色（b）、绿色（g）和红色（r）的值，这些值的范围从0到1，步长为1/DEPTH。这将生成DEPTH^3种不同的颜色组合
# for b in tqdm(np.arange(0, 1.01, 1 / DEPTH)):
#     for g in np.arange(0, 1.01, 1 / DEPTH):
#         for r in np.arange(0, 1.01, 1 / DEPTH):
#             mult_vector = [b, g, r]
#             # 如果图像有4个通道（包括透明度），则在乘法向量中添加1以保持透明度不变
#             if channels == 4:
#                 mult_vector.append(1)
#             # 将颜色向量与图像的每个像素相乘，创建一个新图像
#             new_img = img * mult_vector
#             # 将新图像转换为无符号8位整数格式，这是图像文件格式通常使用的格式
#             new_img = new_img.astype('uint8')
#
#             # 遍历ROTATIONS列表中的每个旋转角度
#             for rotation in ROTATIONS:
#                 # 使用cv2.getRotationMatrix2D创建一个旋转矩阵
#                 rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1)
#                 abs_cos = abs(rotation_matrix[0,0])
#                 abs_sin = abs(rotation_matrix[0,1])
#                 # 计算旋转后的图像的新宽度和新高度
#                 new_w = int(height * abs_sin + width * abs_cos)
#                 new_h = int(height * abs_cos + width * abs_sin)
#                 # 调整旋转矩阵以确保图像旋转后居中
#                 rotation_matrix[0, 2] += new_w/2 - center[0]
#                 rotation_matrix[1, 2] += new_h/2 - center[1]
#                 # 使用cv2.warpAffine将图像应用旋转矩阵，并将结果写入到输出文件夹中的新文件。文件名基于原始图像名称、颜色值和旋转角度构建
#                 cv2.imwrite(
#                     f'{out_folder}/{img_name}_{round(r,1)}_{round(g,1)}_{round(b,1)}_r{rotation}.{ext}',
#                     cv2.warpAffine(new_img, rotation_matrix, (new_w, new_h)),
#                     # compress image
#                     # 设置JPEG压缩级别为9，以控制输出图像的压缩率
#                     [cv2.IMWRITE_PNG_COMPRESSION, 9])

from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from typing import Tuple, List
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm
import math
import conf
import multiprocessing

import click


# DEPTH = 4 -> 4 * 4 * 4 = 64 colors
DEPTH = conf.DEPTH
# list of rotations, in degrees, to apply over the original image
ROTATIONS = conf.ROTATIONS
THREADS = multiprocessing.cpu_count()


def get_tile_dir(img_dir: Path, img_name: str) -> Path:
    tile_dir = img_dir / Path(f"gen_{img_name}")

    if not tile_dir.exists():
        tile_dir.mkdir()

    return tile_dir


def get_img(img: Path) -> np.array:
    img = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
    return img.astype("float")


def get_dimensions(img: np.array) -> Tuple[int, int, Tuple[float, float]]:
    height, width = img.shape[:2]
    center = (width / 2, height / 2)

    return height, width, center


def make_rotation(
    img_name: str,
    ext: str,
    tile_dir: Path,
    new_img: np.array,
    dimensions: Tuple[int, int, Tuple[float, float]],
    rotation: float,
    colors: Tuple[float, float, float],
    bar: tqdm,
):
    height, width, center = dimensions
    b, g, r = colors

    rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1)
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_w = int(height * abs_sin + width * abs_cos)
    new_h = int(height * abs_cos + width * abs_sin)
    rotation_matrix[0, 2] += new_w / 2 - center[0]
    rotation_matrix[1, 2] += new_h / 2 - center[1]
    cv2.imwrite(
        f"{tile_dir}/{img_name}_{round(r,1)}_{round(g,1)}_{round(b,1)}_r{rotation}.{ext}",
        cv2.warpAffine(new_img, rotation_matrix, (new_w, new_h)),
        # compress image
        [cv2.IMWRITE_PNG_COMPRESSION, 9],
    )
    bar.update()


def generate_tiles(
    img: np.array,
    img_name: str,
    ext: str,
    tile_dir: Path,
    depth: int,
    rotations: List[int],
    pool: ThreadPoolExecutor,
):
    dimensions = get_dimensions(img)
    b_range = np.arange(0, 1.01, 1 / depth)
    g_range = np.arange(0, 1.01, 1 / depth)
    r_range = np.arange(0, 1.01, 1 / depth)
    operations = len(b_range) ** 3 * len(rotations)
    progress_bar = tqdm(total=operations)

    for b, g, r in product(b_range, g_range, r_range):
        colors = b, g, r
        new_img = img * [b, g, r, 1]
        new_img = new_img.astype("uint8")
        for rotation in rotations:
            pool.submit(
                make_rotation,
                img_name,
                ext,
                tile_dir,
                new_img,
                dimensions,
                rotation,
                colors,
                progress_bar,
            )


@click.command()
@click.option(
    "-d",
    "--depth",
    default=DEPTH,
    help="Color depth.",
    show_default=True,
    type=click.INT,
)
@click.option(
    "-r",
    "--rotations",
    default=ROTATIONS,
    help="Rotations.",
    multiple=True,
    show_default=True,
    type=click.INT,
)
@click.argument("img", type=click.Path(exists=True))
def cmd(img: str, depth: int, rotations: List[int]):
    img_path = Path(img)
    img_dir = img_path.parent
    img_name, ext = img_path.name.split(".")
    tile_dir = get_tile_dir(img_dir, img_name)
    img = get_img(img_path)

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        generate_tiles(img, img_name, ext, tile_dir, depth, rotations, pool)


if __name__ == "__main__":
    cmd()
