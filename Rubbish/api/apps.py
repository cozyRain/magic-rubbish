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
        try:
            from .utils.garbage_classifier import get_classifier
            # 使用模拟模式启动（避免API密钥问题）
            # 如需使用真实API，设置 use_mock=False 并配置环境变量
            classifier = get_classifier(use_mock=True)
            print("✅ AI模型初始化成功（模拟模式）")
            print("   💡 提示：如需使用真实API，请配置腾讯云密钥")
        except ImportError as e:
            print(f"⚠️ 模块导入失败: {e}")
            print("   请确保已安装所有依赖: pip install -r requirements.txt")
        except Exception as e:
            print(f"⚠️ AI模型初始化失败: {e}")
            print("   系统将使用模拟识别模式，不影响基本功能")