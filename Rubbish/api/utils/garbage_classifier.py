"""
垃圾分类AI核心模块
包含模型加载、识别优化等核心功能
"""

import os
import cv2
import numpy as np
import paddlehub as hub
from collections import Counter
from .image_processor import ImagePreprocessor

# 全局模型实例
_model = None

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


def get_classifier():
    """获取分类器实例（单例模式）"""
    global _model
    if _model is None:
        try:
            print("正在加载AI模型...")
            _model = hub.Module(name="garbage_classification")
            print("AI模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    return _model


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

    # 规则2：低置信度修正（简单处理，可扩展）
    if confidence < 0.6:
        # 如果置信度太低且识别为其他垃圾，保守处理
        if predict_category == "other":
            return "other", True

    return predict_category, False


def classify_single(model, image_path):
    """
    单张图片识别
    返回: (物品名称, 分类代码, 置信度)
    """
    result = model.classify_images(paths=[image_path])
    res = result[0]
    item = res["label"]
    confidence = res["confidence"]
    category = GARBAGE_MAP.get(item, "other")
    return item, category, confidence


def classify_with_voting(model, image_path):
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

    for scale in scales:
        new_w = int(w * scale)
        new_h = int(h * scale)
        scaled = cv2.resize(img, (new_w, new_h))

        # 保存临时文件
        temp_path = image_path.replace('.', f'_scale_{scale}.')
        cv2.imwrite(temp_path, scaled)

        try:
            result = model.classify_images(paths=[temp_path])
            res = result[0]
            results.append({
                "item": res["label"],
                "confidence": res["confidence"],
                "category": GARBAGE_MAP.get(res["label"], "other")
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


def classify_with_optimizations(model, image_path, use_preprocess=True, use_voting=True):
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
            if success and processed_path != current_path:
                current_path = processed_path
                result['preprocessed'] = True
        except Exception as e:
            print(f"预处理失败: {e}")

    # 多尺度投票识别
    if use_voting:
        try:
            item, category, confidence = classify_with_voting(model, current_path)
            result['voting_used'] = True
        except Exception as e:
            print(f"投票识别失败，回退到单张识别: {e}")
            item, category, confidence = classify_single(model, current_path)
    else:
        item, category, confidence = classify_single(model, current_path)

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
    if result['preprocessed'] and current_path != image_path and os.path.exists(current_path):
        try:
            os.remove(current_path)
        except:
            pass

    return result