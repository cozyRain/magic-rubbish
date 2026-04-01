from django.db import models
import os
import uuid


def upload_to(instance, filename):
    """自定义上传路径"""
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4().hex}.{ext}"
    return os.path.join('uploads', filename)


class RecognitionRecord(models.Model):
    """识别记录模型"""

    CATEGORY_CHOICES = [
        ('recyclable', '可回收垃圾'),
        ('kitchen', '厨余垃圾'),
        ('harmful', '有害垃圾'),
        ('other', '其他垃圾'),
    ]

    # 图片信息
    image = models.ImageField(upload_to=upload_to, verbose_name='垃圾图片')
    image_url = models.URLField(blank=True, null=True, verbose_name='图片URL')

    # 识别结果
    predicted_item = models.CharField(max_length=100, verbose_name='识别物品')
    predicted_category = models.CharField(max_length=50, choices=CATEGORY_CHOICES, verbose_name='识别分类')
    confidence = models.FloatField(verbose_name='置信度')

    # 优化标识
    preprocessed = models.BooleanField(default=False, verbose_name='是否预处理')
    voting_used = models.BooleanField(default=False, verbose_name='是否多尺度投票')
    rule_applied = models.BooleanField(default=False, verbose_name='是否规则兜底')

    # 用户反馈
    is_correct = models.BooleanField(default=True, verbose_name='识别是否正确')
    actual_category = models.CharField(max_length=50, blank=True, null=True,
                                       choices=CATEGORY_CHOICES, verbose_name='实际分类')
    feedback_note = models.TextField(blank=True, verbose_name='反馈备注')

    # 元数据
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    ip_address = models.GenericIPAddressField(blank=True, null=True, verbose_name='IP地址')

    class Meta:
        db_table = 'recognition_records'
        verbose_name = '识别记录'
        verbose_name_plural = '识别记录'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.predicted_item} - {self.confidence:.2%} - {self.created_at}"

    def get_category_display_name(self):
        """获取分类显示名称"""
        return dict(self.CATEGORY_CHOICES).get(self.predicted_category, '未知')


class ErrorSample(models.Model):
    """错误样本库模型"""

    image = models.ImageField(upload_to='error_samples/', verbose_name='错误样本图片')
    predicted_item = models.CharField(max_length=100, verbose_name='识别物品')
    predicted_category = models.CharField(max_length=50, verbose_name='识别分类')
    actual_category = models.CharField(max_length=50, verbose_name='实际分类')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='添加时间')

    class Meta:
        db_table = 'error_samples'
        verbose_name = '错误样本'
        verbose_name_plural = '错误样本'

    def __str__(self):
        return f"{self.predicted_item} -> {self.actual_category}"