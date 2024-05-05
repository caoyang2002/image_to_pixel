# Image to Pixel

[English](/README.md) · **简体中文** 

![Cover_image](images/title_stripes.png)

👷 使用像素块构建像素图。

[更改日志](Docs/CHANGELOG.md)


>[!NOTE]
> 我使用了 [@alexdelorenzo/tiler](https://github.com/alexdelorenzo/tiler) 的代码, 但是不清楚如何 @ 他的库
> 原作者: [@nuno-faria/tiler](https://github.com/nuno-faria/tiler)
> 
> 更改说明:
> 1. 没有改原作者的图片和英文注释, 仅仅是添加 (好像原作者没有继续维护了)


# 使用

## 一、介绍

Tiler 是一个使用各种小图像（像素块）来创建像素图的工具。它与其他马赛克工具不同，因为它可以适应多种形状和大小的像素块（即不仅限于正方形）。

可以使用圆形、线条、波浪、十字绣、乐高积木、Minecraft 块、回形针、字母等来构建图像……可能性是无限的！

## 二、安装

- 克隆仓库：`git clone https://github.com/caoyang2002/tiler.git` ；
- 安装 Python 3；
- 安装 pip（可选，用于安装依赖项）；
- 安装依赖项：`pip install -r requirements.txt`

## 三、使用方法

- 创建一个包含用于构建图像的像素块的文件夹（仅包含像素块）：
  1. 使用脚本构建: `python ./gen_tiles.py your/image/path`, 例如: `python ./gen_tiles.py ./tiles/block/block.png`
  > 脚本 `gen_tiles.py` 可以在这项任务中提供帮助；它基于源像素块构建具有多种颜色的像素块（注意：推荐源文件具有 RGB 颜色值 (240,240,240)）。使用方式为 `python gen_tiles.py path/to/image`，它会在基础图像的相同路径下创建一个带有 'gen_' 前缀的文件夹。
- 生成像素图片 
  1. 运行 `python tiler.py path/to/image path/to/tiles_folder/` 比如 `python ./tiler.py ./image_in/bird.png ./tiles/block/gen_block/`

  >[!tip]
  > 如果原始图片太大, 则会等待较长的时间, 以 2400x2400 为例,需要等待 10 分钟


## 四、配置

所有配置都可以在 `conf.py` 文件中更改。

#### `gen_tiles.py`

- `DEPTH` - 每个颜色通道的划分数（例如：DEPTH = 4 -> 4 * 4 * 4 = 64 种颜色）；
- `ROTATIONS` - 应用于原始图像的旋转列表，以度为单位（例如：[0, 90]）。

#### `tiler.py`

- `COLOR_DEPTH` - 每个颜色通道的划分数（例如：COLOR_DEPTH = 4 -> 4 * 4 * 4 = 64 种颜色）；
- `IMAGE_SCALE` - 应用于要平铺的图像的缩放比例（1 = 默认缩放）；
- `RESIZING_SCALES` - 应用于每个瓷砖的缩放比例（例如：[1, 0.75, 0.5, 0.25]）；
- `PIXEL_SHIFT` - 创建每个框时移动的像素数（例如：(5,5)）；如果没有指定，则移动距离将与瓷砖尺寸相同；
- `OVERLAP_TILES` - 是否允许像素块重叠；
- `RENDER` - 在构建图像时渲染图像；
- `POOL_SIZE` - 多进程池大小；
- `IMAGE_TO_TILE` - 图像到瓷砖的路径（如果作为第一个参数传递则忽略）；
- `TILES_FOLDER` - 包含像素块的文件夹（如果作为第二个参数传递则忽略）；
- `OUT` - 输出图像文件名。



# 开发
- [iamge_in](/image_in) 测试输入文件
- [tiles](/tiles) 像素块文件

## 参考
https://github.com/alexdelorenzo/tiler


