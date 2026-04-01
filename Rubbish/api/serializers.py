"""
序列化器模块
定义API数据的序列化和反序列化规则
"""

from rest_framework import serializers
from .models import RecognitionRecord, ErrorSample


class RecognitionRecordSerializer(serializers.ModelSerializer):
    """识别记录序列化器"""

    category_display = serializers.SerializerMethodField()
    created_at_display = serializers.SerializerMethodField()

    class Meta:
        model = RecognitionRecord
        fields = '__all__'
        read_only_fields = ['id', 'created_at']

    def get_category_display(self, obj):
        return obj.get_category_display_name()

    def get_created_at_display(self, obj):
        return obj.created_at.strftime('%Y-%m-%d %H:%M:%S')


class ErrorSampleSerializer(serializers.ModelSerializer):
    """错误样本序列化器"""

    created_at_display = serializers.SerializerMethodField()

    class Meta:
        model = ErrorSample
        fields = '__all__'
        read_only_fields = ['id', 'created_at']

    def get_created_at_display(self, obj):
        return obj.created_at.strftime('%Y-%m-%d %H:%M:%S')


class RecognitionRequestSerializer(serializers.Serializer):
    """识别请求序列化器"""

    image = serializers.ImageField(required=True)
    use_preprocess = serializers.BooleanField(default=True, required=False)
    use_voting = serializers.BooleanField(default=True, required=False)
    use_filter = serializers.BooleanField(default=True, required=False)


class RecognitionResponseSerializer(serializers.Serializer):
    """识别响应序列化器"""

    success = serializers.BooleanField()
    predicted_item = serializers.CharField()
    predicted_category = serializers.CharField()
    confidence = serializers.FloatField()
    advice = serializers.CharField()
    preprocessed = serializers.BooleanField()
    voting_used = serializers.BooleanField()
    rule_applied = serializers.BooleanField()
    record_id = serializers.IntegerField(required=False)
    error = serializers.CharField(required=False)


class FeedbackRequestSerializer(serializers.Serializer):
    """反馈请求序列化器"""

    record_id = serializers.IntegerField(required=True)
    is_correct = serializers.BooleanField(required=True)
    actual_category = serializers.CharField(required=False, allow_blank=True)
    note = serializers.CharField(required=False, allow_blank=True)


class TestReportSerializer(serializers.Serializer):
    """测试报告序列化器"""

    total_count = serializers.IntegerField()
    correct_count = serializers.IntegerField()
    error_count = serializers.IntegerField()
    accuracy = serializers.FloatField()
    preprocessed_count = serializers.IntegerField()
    voting_count = serializers.IntegerField()
    errors = serializers.ListField()