import cv2
import numpy as np

def resize_and_pad(image, target_size=(640, 640), color=(114, 114, 114)):
    """
    调整图像尺寸并填充至目标大小，同时保持比例不变。

    :param image: 输入图像 (H, W, C)
    :param target_size: 目标尺寸 (宽, 高)，默认为 (640, 640)
    :param color: 填充颜色，默认为灰色 (114, 114, 114)
    :return: 填充后的图像和缩放比例
    """
    original_h, original_w = image.shape[:2]
    target_w, target_h = target_size

    # 计算缩放比例和新尺寸
    scale = min(target_w / original_w, target_h / original_h)
    new_w, new_h = int(original_w * scale), int(original_h * scale)

    # 缩放图像
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 计算填充
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    # 创建填充后的图像
    padded_image = cv2.copyMakeBorder(
        resized_image,
        pad_h, target_h - new_h - pad_h,  # 上下填充
        pad_w, target_w - new_w - pad_w,  # 左右填充
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded_image, scale, (pad_w, pad_h)

# 测试
if __name__ == "__main__":
    # 加载测试图像
    img = cv2.imread("/home/lvyong/yolov5-cys/data/images/000445.jpg")

    # 调整和填充图像
    target_size = (640, 640)
    padded_img, scale, padding = resize_and_pad(img, target_size)

    # 显示结果
    print(f"缩放比例: {scale}")
    print(f"填充: {padding}")
    cv2.imshow("Original Image", img)
    cv2.imshow("Padded Image", padded_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()