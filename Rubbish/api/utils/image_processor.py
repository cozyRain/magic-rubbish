"""
图像预处理模块
包含降噪、增强、裁剪、无效检测等功能
"""

import cv2
import numpy as np
import os


class ImagePreprocessor:
    """图像预处理器"""

    @staticmethod
    def denoise_and_enhance(image_path):
        """
        图像预处理流程：
        1. 高斯模糊去噪
        2. 自适应亮度/对比度调整
        3. 主体裁剪
        """
        img = cv2.imread(image_path)
        if img is None:
            return None, False

        # 高斯模糊去噪
        denoised = cv2.GaussianBlur(img, (5, 5), 0)

        # 自适应亮度/对比度调整
        hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        brightness_mean = np.mean(v)
        if brightness_mean < 100:  # 暗光
            v = cv2.add(v, 50)
        elif brightness_mean > 200:  # 过曝
            v = cv2.add(v, -30)

        v = cv2.convertScaleAbs(v, alpha=1.2, beta=10)
        hsv_enhanced = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

        # 主体裁剪
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            h, w = enhanced.shape[:2]

            if area > (h * w * 0.05):
                x, y, w_box, h_box = cv2.boundingRect(largest_contour)
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w_box = min(w - x, w_box + 2 * padding)
                h_box = min(h - y, h_box + 2 * padding)
                cropped = enhanced[y:y+h_box, x:x+w_box]

                preprocessed_path = image_path.replace(".", "_preprocessed.")
                cv2.imwrite(preprocessed_path, cropped)
                return preprocessed_path, True

        preprocessed_path = image_path.replace(".", "_enhanced.")
        cv2.imwrite(preprocessed_path, enhanced)
        return preprocessed_path, True


def is_invalid_image(image_path):
    """
    检测是否为无效图片
    返回: (是否无效, 原因)
    """
    img = cv2.imread(image_path)
    if img is None:
        return True, "无法读取图片"

    h, w = img.shape[:2]

    # 色彩饱和度检测
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    sat_std = np.std(saturation)

    if sat_std < 15:
        return True, "检测到纯色背景图片，请上传真实的垃圾照片"

    # 边缘检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / (h * w)

    if edge_ratio < 0.01:
        return True, "图像过于简单，请上传包含垃圾物品的清晰图片"

    # 人脸检测（简单版）
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            return True, "检测到人脸，请上传垃圾物品图片"
    except:
        pass

    return False, ""