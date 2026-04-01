import os
import json
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.conf import settings
from rest_framework.decorators import api_view, action
from rest_framework.response import Response
from rest_framework import viewsets, status
from rest_framework.parsers import MultiPartParser, FormParser

from .models import RecognitionRecord, ErrorSample
from .serializers import (
    RecognitionRecordSerializer, ErrorSampleSerializer,
    RecognitionRequestSerializer, RecognitionResponseSerializer,
    FeedbackRequestSerializer, TestReportSerializer
)
from .utils.garbage_classifier import get_classifier, classify_with_optimizations
from .utils.image_processor import is_invalid_image
from .utils.test_logger import get_test_logger

# 添加腾讯云环境变量配置（可选）
# 如需使用真实API，请在系统环境变量中设置：
# TENCENT_SECRET_ID=your-secret-id
# TENCENT_SECRET_KEY=your-secret-key
# 或在项目根目录创建 .env 文件

# 检查腾讯云配置
TENCENT_SECRET_ID = os.environ.get('TENCENT_SECRET_ID', '')
TENCENT_SECRET_KEY = os.environ.get('TENCENT_SECRET_KEY', '')
if TENCENT_SECRET_ID and TENCENT_SECRET_KEY:
    print("✅ 检测到腾讯云API密钥，将使用真实识别")
else:
    print("ℹ️ 未检测到腾讯云API密钥，将使用模拟识别模式")
    print("   如需使用真实识别，请配置环境变量 TENCENT_SECRET_ID 和 TENCENT_SECRET_KEY")

# ==================== 网页视图 ====================

def index(request):
    """首页"""
    # 获取最近的识别记录
    recent_records = RecognitionRecord.objects.all()[:10]
    context = {
        'recent_records': recent_records,
        'total_count': RecognitionRecord.objects.count(),
        'error_count': RecognitionRecord.objects.filter(is_correct=False).count(),
    }
    return render(request, 'index.html', context)


def result_page(request, record_id):
    """结果页面"""
    record = get_object_or_404(RecognitionRecord, id=record_id)
    context = {
        'record': record,
        'category_display': record.get_category_display_name(),
        'advice': get_advice(record.predicted_category)
    }
    return render(request, 'result.html', context)


def report_page(request):
    """测试报告页面"""
    records = RecognitionRecord.objects.all()
    total = records.count()
    correct = records.filter(is_correct=True).count()
    error = total - correct

    # 统计优化应用情况
    preprocessed_count = records.filter(preprocessed=True).count()
    voting_count = records.filter(voting_used=True).count()
    rule_count = records.filter(rule_applied=True).count()

    # 错误详情
    errors_list = []
    for record in records.filter(is_correct=False):
        errors_list.append({
            'id': record.id,
            'image_url': record.image.url if record.image else '',
            'predicted_item': record.predicted_item,
            'predicted_category': record.get_category_display_name(),
            'actual_category': dict(RecognitionRecord.CATEGORY_CHOICES).get(
                record.actual_category, '未知'
            ),
            'confidence': record.confidence,
            'created_at': record.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })

    context = {
        'total_count': total,
        'correct_count': correct,
        'error_count': error,
        'accuracy': (correct / total * 100) if total > 0 else 0,
        'preprocessed_count': preprocessed_count,
        'voting_count': voting_count,
        'rule_count': rule_count,
        'errors': errors_list,
    }
    return render(request, 'test_report.html', context)


# ==================== API视图 ====================

@api_view(['POST'])
@csrf_exempt
def recognize_garbage(request):
    """
    垃圾分类识别API
    接收图片，返回识别结果
    """
    # 验证请求
    serializer = RecognitionRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response({
            'success': False,
            'error': '参数错误',
            'details': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)

    validated_data = serializer.validated_data
    image = validated_data['image']
    use_preprocess = validated_data.get('use_preprocess', True)
    use_voting = validated_data.get('use_voting', True)
    use_filter = validated_data.get('use_filter', True)

    # 保存临时文件
    temp_path = default_storage.save(f'temp/{image.name}', image)
    full_temp_path = os.path.join(settings.MEDIA_ROOT, temp_path)

    try:
        # 无效画面过滤
        if use_filter:
            is_invalid, reason = is_invalid_image(full_temp_path)
            if is_invalid:
                return Response({
                    'success': False,
                    'error': f'无效图片: {reason}',
                    'predicted_item': None,
                    'predicted_category': None,
                    'confidence': 0
                }, status=status.HTTP_400_BAD_REQUEST)

        # 获取分类器并识别
        classifier = get_classifier()
        result = classify_with_optimizations(
            classifier,
            full_temp_path,
            use_preprocess=use_preprocess,
            use_voting=use_voting
        )

        # 获取投放建议
        advice = get_advice(result['predicted_category'])

        # 保存识别记录
        record = RecognitionRecord.objects.create(
            image=temp_path,
            predicted_item=result['predicted_item'],
            predicted_category=result['predicted_category'],
            confidence=result['confidence'],
            preprocessed=result.get('preprocessed', False),
            voting_used=result.get('voting_used', False),
            rule_applied=result.get('rule_applied', False),
            ip_address=get_client_ip(request)
        )

        response_data = {
            'success': True,
            'record_id': record.id,
            'predicted_item': result['predicted_item'],
            'predicted_category': result['predicted_category'],
            'category_display': record.get_category_display_name(),
            'confidence': result['confidence'],
            'advice': advice,
            'preprocessed': result.get('preprocessed', False),
            'voting_used': result.get('voting_used', False),
            'rule_applied': result.get('rule_applied', False)
        }

        return Response(response_data, status=status.HTTP_200_OK)

    except Exception as e:
        # 清理临时文件
        if os.path.exists(full_temp_path):
            os.remove(full_temp_path)

        return Response({
            'success': False,
            'error': f'识别失败: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def submit_feedback(request):
    """
    提交识别反馈
    """
    serializer = FeedbackRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response({
            'success': False,
            'error': '参数错误'
        }, status=status.HTTP_400_BAD_REQUEST)

    validated_data = serializer.validated_data
    record_id = validated_data['record_id']
    is_correct = validated_data['is_correct']
    actual_category = validated_data.get('actual_category', '')
    note = validated_data.get('note', '')

    try:
        record = RecognitionRecord.objects.get(id=record_id)
        record.is_correct = is_correct
        if actual_category:
            record.actual_category = actual_category
        if note:
            record.feedback_note = note
        record.save()

        # 如果是错误识别，保存到错误样本库
        if not is_correct and actual_category and record.image:
            ErrorSample.objects.create(
                image=record.image,
                predicted_item=record.predicted_item,
                predicted_category=record.predicted_category,
                actual_category=actual_category
            )

        return Response({
            'success': True,
            'message': '反馈已提交'
        }, status=status.HTTP_200_OK)

    except RecognitionRecord.DoesNotExist:
        return Response({
            'success': False,
            'error': '记录不存在'
        }, status=status.HTTP_404_NOT_FOUND)


@api_view(['GET'])
def get_test_report(request):
    """
    获取测试报告
    """
    records = RecognitionRecord.objects.all()
    total = records.count()
    correct = records.filter(is_correct=True).count()
    error = total - correct

    # 统计优化应用情况
    preprocessed_count = records.filter(preprocessed=True).count()
    voting_count = records.filter(voting_used=True).count()

    # 错误详情
    errors_list = []
    for record in records.filter(is_correct=False)[:100]:  # 最多返回100条
        errors_list.append({
            'id': record.id,
            'image_url': record.image.url if record.image else '',
            'predicted_item': record.predicted_item,
            'predicted_category': record.get_category_display_name(),
            'actual_category': dict(RecognitionRecord.CATEGORY_CHOICES).get(
                record.actual_category, '未知'
            ),
            'confidence': record.confidence,
            'created_at': record.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })

    report_data = {
        'total_count': total,
        'correct_count': correct,
        'error_count': error,
        'accuracy': (correct / total * 100) if total > 0 else 0,
        'preprocessed_count': preprocessed_count,
        'voting_count': voting_count,
        'errors': errors_list
    }

    return Response(report_data, status=status.HTTP_200_OK)


@api_view(['GET'])
def get_statistics(request):
    """
    获取统计信息
    """
    total = RecognitionRecord.objects.count()
    correct = RecognitionRecord.objects.filter(is_correct=True).count()

    # 按分类统计
    category_stats = {}
    for category_code, category_name in RecognitionRecord.CATEGORY_CHOICES:
        count = RecognitionRecord.objects.filter(predicted_category=category_code).count()
        category_stats[category_name] = count

    # 按天统计
    from django.db.models import Count
    from django.db.models.functions import TruncDate

    daily_stats = RecognitionRecord.objects.annotate(
        date=TruncDate('created_at')
    ).values('date').annotate(
        count=Count('id')
    ).order_by('-date')[:30]

    return Response({
        'total': total,
        'correct': correct,
        'accuracy': (correct / total * 100) if total > 0 else 0,
        'category_distribution': category_stats,
        'daily_records': list(daily_stats)
    }, status=status.HTTP_200_OK)


class RecognitionRecordViewSet(viewsets.ModelViewSet):
    """识别记录视图集"""
    queryset = RecognitionRecord.objects.all()
    serializer_class = RecognitionRecordSerializer

    def get_queryset(self):
        queryset = super().get_queryset()

        # 过滤参数
        is_correct = self.request.query_params.get('is_correct')
        if is_correct is not None:
            queryset = queryset.filter(is_correct=is_correct.lower() == 'true')

        category = self.request.query_params.get('category')
        if category:
            queryset = queryset.filter(predicted_category=category)

        return queryset


class ErrorSampleViewSet(viewsets.ModelViewSet):
    """错误样本视图集"""
    queryset = ErrorSample.objects.all()
    serializer_class = ErrorSampleSerializer


# ==================== 辅助函数 ====================

def get_advice(category):
    """获取投放建议"""
    advice_map = {
        'recyclable': '🗑️ 请投入蓝色垃圾桶，回收前请清洁干净，压扁后投放',
        'kitchen': '🌿 请投入绿色垃圾桶，建议沥干水分，去除包装袋',
        'harmful': '⚠️ 请投入红色垃圾桶，小心轻放避免破损，特殊有害垃圾请单独处理',
        'other': '⚫ 请投入黑色垃圾桶，无法回收利用的废弃物'
    }
    return advice_map.get(category, '请按照当地垃圾分类标准投放')


def get_client_ip(request):
    """获取客户端IP"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip