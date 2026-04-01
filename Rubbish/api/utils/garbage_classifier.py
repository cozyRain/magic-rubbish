"""
垃圾分类AI核心模块 - 腾讯云API版本
无需安装PaddlePaddle，使用云端API或模拟器
"""

import os
import cv2
import numpy as np
import base64
import hashlib
from collections import Counter
from django.conf import settings

# 导入图像处理器
from .image_processor import ImagePreprocessor

# 腾讯云API配置（从环境变量读取）
TENCENT_SECRET_ID = os.environ.get('TENCENT_SECRET_ID', '')
TENCENT_SECRET_KEY = os.environ.get('TENCENT_SECRET_KEY', '')

# 垃圾分类映射表
GARBAGE_MAP = {
    "易拉罐": "recyclable", "塑料瓶": "recyclable", "纸张": "recyclable",
    "玻璃": "recyclable", "金属": "recyclable", "纸箱": "recyclable",
    "报纸": "recyclable", "杂志": "recyclable", "饮料瓶": "recyclable",
    "果皮": "kitchen", "菜叶": "kitchen", "剩饭": "kitchen",
    "苹果核": "kitchen", "香蕉皮": "kitchen", "鸡蛋壳": "kitchen",
    "电池": "harmful", "灯管": "harmful", "药品": "harmful",
    "过期药品": "harmful", "废电池": "harmful",
    "纸巾": "other", "塑料袋": "other", "一次性餐具": "other",
}

# 关键词白名单
HARMFUL_KEYWORDS = ["电池", "灯管", "药品", "农药", "废电池", "过期药"]
KITCHEN_KEYWORDS = ["果皮", "菜叶", "剩饭", "蛋壳", "苹果核", "香蕉皮"]
RECYCLABLE_KEYWORDS = ["塑料瓶", "易拉罐", "纸张", "玻璃", "金属", "纸箱"]

# 全局分类器实例
_classifier = None


class MockClassifier:
    """模拟分类器（用于测试，不需要API密钥）"""

    def __init__(self):
        self.mock_items = [
            ("塑料瓶", "recyclable", 0.95),
            ("易拉罐", "recyclable", 0.92),
            ("电池", "harmful", 0.88),
            ("果皮", "kitchen", 0.85),
            ("纸巾", "other", 0.80),
            ("纸张", "recyclable", 0.90),
            ("玻璃瓶", "recyclable", 0.87),
            ("剩饭", "kitchen", 0.83),
            ("药品", "harmful", 0.89),
            ("塑料袋", "other", 0.78),
        ]

    def classify_image(self, image_path):
        """
        模拟识别
        基于图片内容的确定性模拟（保证同一张图结果稳定）
        """
        try:
            with open(image_path, 'rb') as f:
                content = f.read()
                # 使用图片内容的哈希值决定结果
                hash_val = int(hashlib.md5(content).hexdigest()[:8], 16)

            index = hash_val % len(self.mock_items)
            return self.mock_items[index]
        except Exception as e:
            print(f"模拟识别失败: {e}")
            return "未知物品", "other", 0.50


class TencentGarbageClassifier:
    """腾讯云垃圾分类识别器"""

    def __init__(self, secret_id=None, secret_key=None):
        self.secret_id = secret_id or TENCENT_SECRET_ID
        self.secret_key = secret_key or TENCENT_SECRET_KEY
        self.use_real_api = bool(self.secret_id and self.secret_key and self.secret_id != 'your-secret-id')

        if not self.use_real_api:
            print("⚠️ 未配置腾讯云API密钥，使用模拟识别器")
            self.mock_classifier = MockClassifier()

    def classify_image(self, image_path):
        """
        识别单张图片
        返回: (物品名称, 分类代码, 置信度)
        """
        # 如果没有配置API密钥，使用模拟器
        if not self.use_real_api:
            return self.mock_classifier.classify_image(image_path)

        # 使用腾讯云API识别
        try:
            return self._classify_with_tencent_api(image_path)
        except Exception as e:
            print(f"腾讯云API识别失败: {e}，回退到模拟识别")
            return self.mock_classifier.classify_image(image_path)

    def _classify_with_tencent_api(self, image_path):
        """使用腾讯云API进行识别"""
        try:
            # 导入腾讯云SDK
            from tencentcloud.common import credential
            from tencentcloud.common.profile.client_profile import ClientProfile
            from tencentcloud.common.profile.http_profile import HttpProfile
            from tencentcloud.tiia.v20190529 import tiia_client, models

            # 读取图片并转为base64
            with open(image_path, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')

            # 实例化认证对象
            cred = credential.Credential(self.secret_id, self.secret_key)

            # 实例化http选项
            httpProfile = HttpProfile()
            httpProfile.endpoint = "tiia.tencentcloudapi.com"

            # 实例化client选项
            clientProfile = ClientProfile()
            clientProfile.httpProfile = httpProfile

            # 实例化客户端
            client = tiia_client.TiiaClient(cred, "ap-beijing", clientProfile)

            # 实例化请求对象
            req = models.DetectLabelRequest()
            req.ImageBase64 = image_base64
            req.Scenes = ["CAMERA"]

            # 发送请求
            resp = client.DetectLabel(req)

            # 解析结果
            if resp.Labels:
                # 查找垃圾分类相关标签
                for label in resp.Labels:
                    label_name = label.Name
                    label_confidence = label.Confidence / 100

                    # 映射到四大分类
                    category = self._map_to_category(label_name)

                    # 如果找到了相关分类，返回结果
                    if category:
                        return label_name, category, label_confidence

                # 如果没有找到明确分类，返回第一个标签
                first_label = resp.Labels[0]
                return first_label.Name, "other", first_label.Confidence / 100

            # 没有识别结果
            return "未知物品", "other", 0.5

        except ImportError:
            print("腾讯云SDK未安装，使用模拟识别")
            return self.mock_classifier.classify_image(image_path)
        except Exception as e:
            print(f"腾讯云API调用失败: {e}")
            raise e

    def _map_to_category(self, label_name):
        """将标签映射到四大分类"""
        label_lower = label_name.lower()

        # 可回收垃圾关键词
        recyclable_keywords = ["塑料", "易拉罐", "纸张", "玻璃", "金属", "纸箱", "瓶子", "包装"]
        for keyword in recyclable_keywords:
            if keyword in label_lower:
                return "recyclable"

        # 厨余垃圾关键词
        kitchen_keywords = ["果皮", "菜叶", "剩饭", "厨余", "食物", "餐厨", "残渣"]
        for keyword in kitchen_keywords:
            if keyword in label_lower:
                return "kitchen"

        # 有害垃圾关键词
        harmful_keywords = ["电池", "灯管", "药品", "有害", "农药", "化学品"]
        for keyword in harmful_keywords:
            if keyword in label_lower:
                return "harmful"

        # 默认为其他垃圾
        return "other"


def get_classifier(use_mock=False):
    """
    获取分类器实例（单例模式）
    use_mock: 是否强制使用模拟模式
    """
    global _classifier
    if _classifier is None:
        if use_mock:
            print("使用模拟分类器（测试模式）")
            _classifier = MockClassifier()
        else:
            print("初始化垃圾分类识别器...")
            _classifier = TencentGarbageClassifier()
            print("垃圾分类识别器初始化完成")
    return _classifier


def apply_rule_fallback(predict_item, predict_category, confidence):
    """
    置信度+规则双保险
    返回: (最终分类, 是否应用规则)
    """
    # 规则1：关键词白名单强制兜底
    for keyword in HARMFUL_KEYWORDS:
        if keyword in predict_item:
            return "harmful", True

    for keyword in KITCHEN_KEYWORDS:
        if keyword in predict_item:
            return "kitchen", True

    for keyword in RECYCLABLE_KEYWORDS:
        if keyword in predict_item:
            return "recyclable", True

    # 规则2：低置信度修正
    if confidence < 0.6:
        # 如果置信度太低，根据物品名称判断
        if any(k in predict_item for k in HARMFUL_KEYWORDS):
            return "harmful", True
        elif any(k in predict_item for k in KITCHEN_KEYWORDS):
            return "kitchen", True
        elif any(k in predict_item for k in RECYCLABLE_KEYWORDS):
            return "recyclable", True
        else:
            return "other", True

    return predict_category, False


def classify_single(classifier, image_path):
    """
    单张图片识别
    返回: (物品名称, 分类代码, 置信度)
    """
    item, category, confidence = classifier.classify_image(image_path)
    return item, category, confidence


def classify_with_voting(classifier, image_path):
    """
    多尺度投票识别
    返回: (物品名称, 分类代码, 置信度)
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None, 0

    h, w = img.shape[:2]
    scales = [1.0, 0.8, 1.2]
    results = []

    import tempfile

    for scale in scales:
        new_w = int(w * scale)
        new_h = int(h * scale)
        scaled = cv2.resize(img, (new_w, new_h))

        # 保存临时文件
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, scaled)

        try:
            item, category, confidence = classifier.classify_image(temp_path)
            results.append({
                "item": item,
                "confidence": confidence,
                "category": category
            })
        except Exception as e:
            print(f"尺度 {scale} 识别失败: {e}")
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if not results:
        return None, None, 0

    # 投票选择类别
    category_votes = [r["category"] for r in results]
    most_common_category = Counter(category_votes).most_common(1)[0][0]

    # 选择置信度最高的作为物品名称
    best_result = max(results, key=lambda x: x["confidence"])

    return best_result["item"], most_common_category, best_result["confidence"]


def classify_with_optimizations(classifier, image_path, use_preprocess=True, use_voting=True):
    """
    带优化的识别流程
    返回: {
        'predicted_item': str,
        'predicted_category': str,
        'confidence': float,
        'preprocessed': bool,
        'voting_used': bool,
        'rule_applied': bool
    }
    """
    result = {
        'preprocessed': False,
        'voting_used': False,
        'rule_applied': False
    }

    current_path = image_path

    # 图像预处理
    if use_preprocess:
        try:
            preprocessor = ImagePreprocessor()
            processed_path, success = preprocessor.denoise_and_enhance(current_path)
            if success and processed_path and processed_path != current_path:
                current_path = processed_path
                result['preprocessed'] = True
        except Exception as e:
            print(f"预处理失败: {e}")

    # 多尺度投票识别
    if use_voting:
        try:
            item, category, confidence = classify_with_voting(classifier, current_path)
            result['voting_used'] = True
        except Exception as e:
            print(f"投票识别失败，回退到单张识别: {e}")
            item, category, confidence = classify_single(classifier, current_path)
    else:
        item, category, confidence = classify_single(classifier, current_path)

    # 如果识别失败，使用默认值
    if item is None:
        item = "未知物品"
        category = "other"
        confidence = 0.5

    # 规则兜底
    final_category, rule_applied = apply_rule_fallback(item, category, confidence)
    result['rule_applied'] = rule_applied

    # 构建返回结果
    result.update({
        'predicted_item': item,
        'predicted_category': final_category,
        'confidence': confidence
    })

    # 清理临时预处理图片
    if result['preprocessed'] and current_path != image_path and current_path and os.path.exists(current_path):
        try:
            os.remove(current_path)
        except:
            pass

    return result