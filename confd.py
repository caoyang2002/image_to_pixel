# GEN TILES CONFS
# 生成马赛克图片的配置

# number of divisions per channel (R, G and B)
# DEPTH = 4 -> 4 * 4 * 4 = 64 colors
# 每个颜色通道（红、绿、蓝）的划分数
DEPTH = 4
# list of rotations, in degrees, to apply over the original image
# 应用于原始图像的旋转列表，单位为度
ROTATIONS = [0]


#############################


# TILER CONFS

# number of divisions per channel
# (COLOR_DEPTH = 32 -> 32 * 32 * 32 = 32768 colors)
# 每个颜色通道的划分数 32
COLOR_DEPTH = 8  # 32
# Scale of the image to be tiled (1 = default resolution)
# 要平铺的图像的缩放比例（1 = 默认分辨率）
IMAGE_SCALE = 1
# tiles scales (1 = default resolution)
# 瓦片的缩放比例（1 = 默认分辨率）
RESIZING_SCALES = [0.12] # [0.5, 0.4, 0.3, 0.2, 0.1]
# number of pixels shifted to create each box (tuple with (x,y))
# if value is None, shift will be done accordingly to tiles dimensions
# 创建每个盒子时移动的像素数（(x,y) 的元组）
# 如果值为 None，则根据瓦片的尺寸进行移动
# PIXEL_SHIFT = (5, 5)
PIXEL_SHIFT = None
# if tiles can overlap
# 瓦片是否可以重叠
OVERLAP_TILES = False
# OVERLAP_TILES = True
# render image as its being built
# 渲染图像时边构建边显示
RENDER = False
# RENDER = True
# multiprocessing pool size
# 多进程池的大小
POOL_SIZE = 8

# out file name
OUT = './pix_out/out.png'
# image to tile (ignored if passed as the 1st arg)
# IMAGE_TO_TILE = None
IMAGE_TO_TILE = r"image_in/bird_x.png"
# IMAGE_TO_TILE = r"image_in/bird_x.png"
# folder with tiles (ignored if passed as the 2nd arg)
# TILES_FOLDER = None
# TILES_FOLDER = r"tiles/lego/gen_lego_v"
TILES_FOLDER = r"tiles/block/gen_block"
