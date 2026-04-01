"""
测试日志模块
记录识别结果和缺陷报告
"""

import os
import json
import datetime
from django.conf import settings


class TestLogger:
    """测试日志记录器"""

    def __init__(self, log_file=None):
        if log_file is None:
            log_file = os.path.join(settings.BASE_DIR, 'test_logs', 'recognition_log.json')

        self.log_file = log_file
        self._ensure_log_dir()
        self.records = []
        self._load_records()

    def _ensure_log_dir(self):
        """确保日志目录存在"""
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def _load_records(self):
        """加载已有记录"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    self.records = json.load(f)
            except:
                self.records = []

    def _save_records(self):
        """保存记录"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    def add_record(self, record_data):
        """添加识别记录"""
        record = {
            'timestamp': datetime.datetime.now().isoformat(),
            **record_data
        }
        self.records.append(record)
        self._save_records()
        return record

    def get_defect_report(self):
        """获取缺陷报告"""
        errors = [r for r in self.records if not r.get('is_correct', True)]

        report = {
            'total_count': len(self.records),
            'correct_count': len(self.records) - len(errors),
            'error_count': len(errors),
            'accuracy': (len(self.records) - len(errors)) / len(self.records) * 100 if self.records else 0,
            'errors': errors
        }

        return report

    def export_report(self):
        """导出报告到文件"""
        report = self.get_defect_report()

        report_path = os.path.join(
            settings.BASE_DIR,
            'test_reports',
            f'report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )

        report_dir = os.path.dirname(report_path)
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return report_path


# 全局日志实例
_logger = None


def get_test_logger():
    """获取测试日志实例"""
    global _logger
    if _logger is None:
        _logger = TestLogger()
    return _logger