import cv2
import numpy as np

def invert_colors(image):
    """对图像进行颜色反转，将黑色变为白色，白色变为黑色"""
    return cv2.bitwise_not(image)

def extract_keypoints(image, max_points=20):
    """使用SIFT提取关键点，并按响应值排序选择前max_points个"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(gray, None)

    # 按响应值排序关键点，选择前max_points个
    if len(keypoints) > max_points:
        keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:max_points]

    return keypoints

def select_middle_keypoints(keypoints, num_middle_points=20):
    """从排序后的关键点中选择中间的num_middle_points个"""
    if len(keypoints) <= num_middle_points:
        return keypoints

    # 找到中间的部分
    start_idx = len(keypoints) // 2 - num_middle_points // 2
    end_idx = start_idx + num_middle_points

    return keypoints[start_idx:end_idx]

def draw_keypoints(image, keypoints):
    """在图像中绘制关键点"""
    output_image = image.copy()

    # 绘制关键点（红色圆点）
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(output_image, (x, y), 5, (0, 0, 255), -1)

    return output_image

def main(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}.")
        return

    # 对图像进行颜色反转
    inverted_image = invert_colors(image)

    # 提取前20个按响应值排序的关键点
    keypoints = extract_keypoints(inverted_image, max_points=20)

    # 从中选择中间的8个关键点
    middle_keypoints = select_middle_keypoints(keypoints, num_middle_points=20)

    # 绘制关键点
    result_image = draw_keypoints(inverted_image, middle_keypoints)

    # 显示结果图像
    cv2.imshow("Keypoints", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img_path = '/home/xu/object/wangbo/test_1/4.bmp'  # 你的图像路径
    main(img_path)
