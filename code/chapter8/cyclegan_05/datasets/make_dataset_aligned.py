import os

from PIL import Image

# 参数 folder 是输入的文件夹路径，函数将返回该文件夹下的所有 .png 和 .jpg 图片的完整路径
def get_file_paths(folder):
    # 创建一个空列表，用于存储找到的图片文件路径。
    image_file_paths = []
    # root：当前遍历的目录路径；
# dirs：当前目录下的子目录列表；
# filenames：当前目录下的文件名列表。
    for root, dirs, filenames in os.walk(folder):
        # 对文件名进行排序（按字母顺序）
        filenames = sorted(filenames)
        for filename in filenames:
            # 将 root 转换为绝对路径，防止相对路径错误；
            input_path = os.path.abspath(root)
            # 拼出文件的完整路径
            file_path = os.path.join(input_path, filename)
            if filename.endswith('.png') or filename.endswith('.jpg'):
                image_file_paths.append(file_path)
        # break 语句在第一次循环后立即中断，使函数只处理当前文件夹，不进入子文件夹。
# 因此该函数只返回最顶层目录下的图片，不会递归。
        break  # prevent descending into subfolders
    return image_file_paths

# 图像配对拼接函数（水平拼接版），
# 类似于 GAN（特别是 Pix2Pix）数据预处理中的 “A | B” 拼接操作
def align_images(a_file_paths, b_file_paths, target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for i in range(len(a_file_paths)):
        img_a = Image.open(a_file_paths[i])
        img_b = Image.open(b_file_paths[i])
        assert(img_a.size == img_b.size)
        # 新建一张空白图像用于拼接。
# 模式 "RGB"：表示彩色三通道。
# 尺寸：
# 宽度 = 两张图宽度之和；
# 高度 = 与原图一致。
        aligned_image = Image.new("RGB", (img_a.size[0] * 2, img_a.size[1]))
        # 使用 .paste() 方法在指定位置粘贴图像：
# (0, 0) → 左上角（放 A 图）
# (img_a.size[0], 0) → 在右半部分（放 B 图）
        aligned_image.paste(img_a, (0, 0))
        aligned_image.paste(img_b, (img_a.size[0], 0))
        # 将拼接后的图像保存。
# 文件名格式为 4 位数字（从 0000.jpg 开始）。
        aligned_image.save(os.path.join(target_path, '{:04d}.jpg'.format(i)))

# 从 dataset_path 目录中读取 trainA/trainB、testA/testB 四个子文件夹下的图像，
# 将 A 和 B 的对应图片按序拼接成新的 “A|B” 图像，生成 train/ 和 test/ 两个输出文件夹。
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-path',
        dest='dataset_path',
        help='Which folder to process (it should have subfolders testA, testB, trainA and trainB'
    )
    args = parser.parse_args()

    dataset_folder = args.dataset_path
    print(dataset_folder)

    test_a_path = os.path.join(dataset_folder, 'testA')
    test_b_path = os.path.join(dataset_folder, 'testB')
    # 使用你前面定义的 get_file_paths 函数，
# 获取各自目录下所有图片文件的路径列表。
    test_a_file_paths = get_file_paths(test_a_path)
    test_b_file_paths = get_file_paths(test_b_path)
    # 确保 testA 与 testB 文件数量一致
    assert(len(test_a_file_paths) == len(test_b_file_paths))
    # 定义输出目录，保存拼接后的 “A|B” 图像
    test_path = os.path.join(dataset_folder, 'test')

    train_a_path = os.path.join(dataset_folder, 'trainA')
    train_b_path = os.path.join(dataset_folder, 'trainB')
    train_a_file_paths = get_file_paths(train_a_path)
    train_b_file_paths = get_file_paths(train_b_path)
    assert(len(train_a_file_paths) == len(train_b_file_paths))
    # 输出路径，用于保存拼接后的训练集图片
    train_path = os.path.join(dataset_folder, 'train')
    # 分别对测试集、训练集执行 align_images 函数。
    align_images(test_a_file_paths, test_b_file_paths, test_path)
    align_images(train_a_file_paths, train_b_file_paths, train_path)
