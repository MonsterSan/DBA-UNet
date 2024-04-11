from PIL import Image


def binarize_image(input_image_path, output_image_path, threshold):
    # 打开图像
    image = Image.open(input_image_path)

    # 将图像转换为灰度图
    gray_image = image.convert('L')

    # 创建一个新的二值化图像
    binary_image = gray_image.point(lambda x: 0 if x < threshold else 255, '1')

    # 保存图像
    binary_image.save(output_image_path)


# 输入图像路径和输出图像路径
input_image_path = 'D:\\PycharmProjects\\ms-ddr\\064_180_19-PID.png'
output_image_path = 'D:\\PycharmProjects\\ms-ddr\\064_180_19-PID-binary.png'

# 设置阈值（0-255之间）
threshold = 128

# 调用函数进行二值化
binarize_image(input_image_path, output_image_path, threshold)
