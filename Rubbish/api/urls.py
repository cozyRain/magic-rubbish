from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# 创建路由器
router = DefaultRouter()
router.register(r'records', views.RecognitionRecordViewSet, basename='records')
router.register(r'errors', views.ErrorSampleViewSet, basename='errors')

urlpatterns = [
    # 网页视图
    path('', views.index, name='index'),
    path('result/<int:record_id>/', views.result_page, name='result_page'),
    path('report/', views.report_page, name='report_page'),

    # API接口
    path('api/', include(router.urls)),
    path('api/recognize/', views.recognize_garbage, name='recognize'),
    path('api/feedback/', views.submit_feedback, name='feedback'),
    path('api/test-report/', views.get_test_report, name='test_report'),
    path('api/statistics/', views.get_statistics, name='statistics'),
]