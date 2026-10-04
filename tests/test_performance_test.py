"""
游戏内性能测试与压测系统 - 验收测试
运行方式：python -m pytest tests/test_performance_test.py -q
共 14 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from testing.performance import PerformanceTest


class TestPerformanceTest:
    def setup_method(self):
        self.system = PerformanceTest()

    # 01: 负载生成支持多用户并发
    def test_case_01(self):
        result = self.system.load_generator(concurrency=8, duration=0.2)
        assert result["concurrency"] == 8
        assert result["total_requests"] > 8
        single = self.system.load_generator(concurrency=1, duration=0.2)
        assert result["total_requests"] > single["total_requests"]

    # 02: 负载生成有持续时间，持续时间内不断发请求
    def test_case_02(self):
        long_run = self.system.load_generator(concurrency=2, duration=0.5)
        short_run = self.system.load_generator(concurrency=2, duration=0.1)
        assert long_run["duration"] == 0.5
        assert long_run["total_requests"] > 1
        assert long_run["total_requests"] > short_run["total_requests"]

    # 03: 压力测试并发逐级递增
    def test_case_03(self):
        result = self.system.stress_test(stage_duration=0.1)
        levels = [s["concurrency"] for s in result["stages"]]
        assert len(levels) > 1
        assert levels == sorted(levels)
        assert all(b > a for a, b in zip(levels, levels[1:]))

    # 04: 压力测试能找到性能拐点/服务器极限
    def test_case_04(self):
        result = self.system.stress_test(stage_duration=0.1)
        assert result["breaking_point"] is not None
        assert result["breaking_point"] <= result["max_concurrency_tested"]
        assert result["max_concurrency_tested"] >= self.system._capacity

    # 05: 压力测试结束后验证服务器恢复
    def test_case_05(self):
        result = self.system.stress_test(stage_duration=0.1)
        assert "recovery" in result
        assert result["recovered"] is True
        assert result["recovery"]["error_rate"] == 0.0

    # 06: 延迟测试输出统计指标（平均/P95/P99 等）
    def test_case_06(self):
        result = self.system.latency_test(samples=50)
        for key in ("avg", "min", "max", "p50", "p95", "p99", "stddev"):
            assert key in result
        assert result["min"] <= result["avg"] <= result["max"]
        assert result["p50"] <= result["p95"] <= result["p99"]

    # 07: 延迟测试输出延迟分布
    def test_case_07(self):
        result = self.system.latency_test(samples=50)
        dist = result["distribution"]
        assert dist["bucket_count"] > 1
        assert len(dist["buckets"]) == dist["bucket_count"]
        assert sum(b["count"] for b in dist["buckets"]) == 50

    # 08: 吞吐量测试测量每秒请求数
    def test_case_08(self):
        result = self.system.throughput_test(duration=0.2)
        assert len(result["measurements"]) > 1
        assert result["max_rps"] > 0
        assert all(m["requests_per_second"] > 0 for m in result["measurements"])

    # 09: 吞吐量测试有瓶颈分析
    def test_case_09(self):
        result = self.system.throughput_test(duration=0.2)
        assert result["bottleneck"]["type"]
        assert result["bottleneck"]["description"]
        assert result["saturation_point"] > 0

    # 10: 吞吐量测试给出优化建议
    def test_case_10(self):
        result = self.system.throughput_test(duration=0.2)
        assert isinstance(result["suggestions"], list)
        assert len(result["suggestions"]) > 0
        assert all(isinstance(s, str) and len(s) > 0 for s in result["suggestions"])

    # 11: 报告可生成并可导出分享
    def test_case_11(self, tmp_path):
        self.system.latency_test(samples=20)
        report = self.system.report()
        assert report["text"]
        path = tmp_path / "performance_report.txt"
        self.system.report(path=str(path))
        assert path.exists()
        assert path.stat().st_size > 0

    # 12: 报告包含图表
    def test_case_12(self):
        self.system.latency_test(samples=20)
        self.system.throughput_test(duration=0.2)
        report = self.system.report()
        assert len(report["charts"]) > 0
        assert any("█" in chart for chart in report["charts"].values())

    # 13: 报告包含历史记录与性能变化趋势
    def test_case_13(self):
        self.system.latency_test(samples=20)
        self.system.latency_test(samples=20)
        report = self.system.report()
        assert len(report["history"]) >= 2
        assert all("timestamp" in h and "metrics" in h for h in report["history"])
        assert report["trend"] in ("improving", "degrading", "stable",
                                   "insufficient_data")

    # 14: reset 清空历史；完整流程可串联执行
    def test_case_14(self):
        self.system.load_generator(concurrency=2, duration=0.1)
        self.system.stress_test(stage_duration=0.1)
        self.system.latency_test(samples=20)
        self.system.throughput_test(duration=0.2)
        report = self.system.report()
        for section in ("load_generator", "stress_test",
                        "latency_test", "throughput_test"):
            assert section in report["summary"]["available_results"]
        self.system.reset()
        assert self.system._history == []
        assert self.system._results == {}
        assert self.system._state == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
