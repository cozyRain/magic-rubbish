from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    verbose_name = '垃圾分类识别系统'

    def ready(self):
        """应用启动时初始化AI模型"""
        import os
        import sys

        # 避免在迁移时加载模型
        if 'migrate' in sys.argv or 'makemigrations' in sys.argv:
            return

        # 延迟加载模型，避免阻塞启动
        from .utils.garbage_classifier import get_classifier
        try:
            classifier = get_classifier()
            print("✅ AI模型初始化成功")
        except Exception as e:
            print(f"⚠️ AI模型初始化失败: {e}")