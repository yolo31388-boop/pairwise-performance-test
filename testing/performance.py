"""
游戏内性能测试与压测系统 - 核心模块

提供负载生成、压力测试、延迟测试、吞吐量测试与报告功能。
纯标准库实现，使用虚拟时钟模拟并发请求，保证测试快速且确定。
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import heapq
import math
import statistics
import time


@dataclass
class Config:
    capacity: int = 50
    base_latency: float = 0.01


class PerformanceTest:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []
        self._results = {}
        self._capacity = int(self.config.get("capacity", Config.capacity))
        self._base_latency = float(self.config.get("base_latency", Config.base_latency))

    def update(self, dt: float):
        self._state["elapsed"] = self._state.get("elapsed", 0.0) + dt

    def reset(self):
        self._state = {}
        self._history = []
        self._results = {}

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _default_handler(self, load: int = 1) -> Tuple[float, bool]:
        """模拟服务器：负载越高延迟越大，超过两倍容量后开始出错。"""
        ratio = max(1, load) / max(1, self._capacity)
        latency = self._base_latency * (1.0 + ratio ** 2)
        error = ratio > 2.0
        return latency, error

    def _call_handler(self, handler: Callable, load: int) -> Tuple[float, bool]:
        """调用请求处理器并归一化结果为 (latency, error)。"""
        try:
            result = handler(load)
        except TypeError:
            result = handler()
        if isinstance(result, tuple):
            return float(result[0]), bool(result[1])
        if result is None:
            return self._base_latency, False
        return float(result), False

    def _simulate_load(self, handler: Callable, concurrency: int,
                       duration: float) -> Dict:
        """用虚拟时钟模拟 concurrency 个并发用户持续 duration 秒发请求。"""
        concurrency = max(1, int(concurrency))
        duration = max(0.0, float(duration))
        heap = [(0.0, user) for user in range(concurrency)]
        heapq.heapify(heap)
        total = 0
        errors = 0
        latencies = []
        while heap:
            now, user = heapq.heappop(heap)
            if now > duration:
                break
            latency, error = self._call_handler(handler, concurrency)
            total += 1
            errors += int(error)
            latencies.append(latency)
            heapq.heappush(heap, (now + max(latency, 1e-9), user))
        return {
            "concurrency": concurrency,
            "duration": duration,
            "total_requests": total,
            "errors": errors,
            "error_rate": errors / total if total else 0.0,
            "requests_per_second": total / duration if duration > 0 else 0.0,
            "avg_latency": statistics.fmean(latencies) if latencies else 0.0,
            "latencies": latencies,
        }

    def _record(self, kind: str, metrics: Dict):
        self._history.append({
            "type": kind,
            "timestamp": time.time(),
            "metrics": metrics,
        })

    @staticmethod
    def _percentile(sorted_data: List[float], p: float) -> float:
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * p
        low = math.floor(k)
        high = math.ceil(k)
        if low == high:
            return sorted_data[int(k)]
        return sorted_data[low] + (sorted_data[high] - sorted_data[low]) * (k - low)

    @staticmethod
    def _histogram(data: List[float], bucket_count: int = 10) -> Dict:
        if not data:
            return {"bucket_count": 0, "buckets": []}
        lo, hi = min(data), max(data)
        width = (hi - lo) / bucket_count if hi > lo else 1.0
        buckets = []
        for i in range(bucket_count):
            start = lo + i * width
            end = start + width
            if i == bucket_count - 1:
                count = sum(1 for v in data if start <= v <= hi)
            else:
                count = sum(1 for v in data if start <= v < end)
            buckets.append({"low": start, "high": end, "count": count})
        return {"bucket_count": bucket_count, "buckets": buckets}

    @staticmethod
    def _ascii_bar_chart(values: List[float], labels: Optional[List[str]] = None,
                         width: int = 40) -> str:
        if not values:
            return "(no data)"
        peak = max(values) or 1.0
        lines = []
        for i, value in enumerate(values):
            label = labels[i] if labels else str(i)
            bar_len = max(1, int(value / peak * width))
            lines.append("{} | {} {:.4g}".format(label, "█" * bar_len, value))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 负载生成
    # ------------------------------------------------------------------
    def load_generator(self, handler: Optional[Callable] = None,
                       concurrency: int = 1, duration: float = 1.0) -> Dict:
        """生成负载：concurrency 个并发虚拟用户在 duration 秒内持续发请求。"""
        handler = handler or self._default_handler
        result = self._simulate_load(handler, concurrency, duration)
        self._results["load_generator"] = result
        self._record("load_generator", {
            "concurrency": result["concurrency"],
            "duration": result["duration"],
            "total_requests": result["total_requests"],
            "requests_per_second": result["requests_per_second"],
            "error_rate": result["error_rate"],
        })
        return result

    # ------------------------------------------------------------------
    # 压力测试
    # ------------------------------------------------------------------
    def stress_test(self, handler: Optional[Callable] = None,
                    start: int = 1, step: Optional[int] = None,
                    max_concurrency: Optional[int] = None,
                    stage_duration: float = 0.5,
                    recovery_concurrency: int = 1) -> Dict:
        """压力测试：并发逐级递增找到性能拐点，压测后验证服务器恢复。"""
        handler = handler or self._default_handler
        max_concurrency = int(max_concurrency or self._capacity * 3)
        step = int(step or max(1, self._capacity // 5))
        stages = []
        breaking_point = None
        baseline_latency = None
        concurrency = max(1, int(start))
        while concurrency <= max_concurrency:
            run = self._simulate_load(handler, concurrency, stage_duration)
            stage = {
                "concurrency": concurrency,
                "total_requests": run["total_requests"],
                "requests_per_second": run["requests_per_second"],
                "avg_latency": run["avg_latency"],
                "error_rate": run["error_rate"],
            }
            stages.append(stage)
            if baseline_latency is None:
                baseline_latency = max(stage["avg_latency"], 1e-9)
            if breaking_point is None and (
                    stage["error_rate"] > 0.05
                    or stage["avg_latency"] > 3 * baseline_latency):
                breaking_point = concurrency
            concurrency += step
        # 恢复阶段：高压过后用低负载验证服务器是否恢复正常
        recovery_run = self._simulate_load(handler, recovery_concurrency,
                                           stage_duration)
        recovery = {
            "concurrency": recovery_concurrency,
            "avg_latency": recovery_run["avg_latency"],
            "error_rate": recovery_run["error_rate"],
        }
        recovered = (
            recovery["error_rate"] <= 0.05
            and (baseline_latency is None
                 or recovery["avg_latency"] <= 2 * baseline_latency)
        )
        result = {
            "stages": stages,
            "breaking_point": breaking_point,
            "max_concurrency_tested": stages[-1]["concurrency"] if stages else 0,
            "baseline_latency": baseline_latency or 0.0,
            "recovery": recovery,
            "recovered": recovered,
        }
        self._results["stress_test"] = result
        self._record("stress_test", {
            "breaking_point": breaking_point,
            "max_concurrency_tested": result["max_concurrency_tested"],
            "recovered": recovered,
        })
        return result

    # ------------------------------------------------------------------
    # 延迟测试
    # ------------------------------------------------------------------
    def latency_test(self, handler: Optional[Callable] = None,
                     samples: int = 100) -> Dict:
        """延迟测试：采样响应时间，输出统计指标与延迟分布。"""
        handler = handler or self._default_handler
        samples = max(1, int(samples))
        latencies = [self._call_handler(handler, 1)[0] for _ in range(samples)]
        ordered = sorted(latencies)
        result = {
            "samples": samples,
            "min": ordered[0],
            "max": ordered[-1],
            "avg": statistics.fmean(latencies),
            "stddev": statistics.pstdev(latencies) if len(latencies) > 1 else 0.0,
            "p50": self._percentile(ordered, 0.50),
            "p95": self._percentile(ordered, 0.95),
            "p99": self._percentile(ordered, 0.99),
            "distribution": self._histogram(latencies),
        }
        self._results["latency_test"] = result
        self._record("latency_test", {
            "avg": result["avg"],
            "p95": result["p95"],
            "p99": result["p99"],
        })
        return result

    # ------------------------------------------------------------------
    # 吞吐量测试
    # ------------------------------------------------------------------
    def throughput_test(self, handler: Optional[Callable] = None,
                        duration: float = 0.5,
                        concurrency_levels: Optional[List[int]] = None) -> Dict:
        """吞吐量测试：测量各并发级别下的 RPS，做瓶颈分析并给出优化建议。"""
        handler = handler or self._default_handler
        levels = concurrency_levels or sorted({1, max(1, self._capacity // 4),
                                               max(1, self._capacity // 2),
                                               self._capacity,
                                               self._capacity * 2})
        measurements = []
        for level in levels:
            run = self._simulate_load(handler, level, duration)
            measurements.append({
                "concurrency": level,
                "requests_per_second": run["requests_per_second"],
                "avg_latency": run["avg_latency"],
                "error_rate": run["error_rate"],
            })
        max_rps = max(m["requests_per_second"] for m in measurements)
        # 饱和点：RPS 增长不足 5% 的第一个并发级别
        saturation_point = measurements[-1]["concurrency"]
        for prev, cur in zip(measurements, measurements[1:]):
            if cur["requests_per_second"] < prev["requests_per_second"] * 1.05:
                saturation_point = cur["concurrency"]
                break
        # 瓶颈分析
        overloaded = [m for m in measurements if m["error_rate"] > 0.05]
        if overloaded:
            bottleneck = {
                "type": "overload_errors",
                "description": "并发达到 {} 时错误率超过 5%，服务器过载".format(
                    overloaded[0]["concurrency"]),
            }
            suggestions = [
                "限制单实例最大并发，超出部分排队或快速失败",
                "增加服务器实例并前置负载均衡分摊压力",
                "为下游调用设置超时与熔断，避免故障蔓延",
            ]
        elif saturation_point <= self._capacity * 2:
            bottleneck = {
                "type": "server_capacity",
                "description": "并发 {} 时吞吐量饱和（峰值 {:.1f} RPS），"
                               "继续加压延迟显著上升".format(saturation_point, max_rps),
            }
            suggestions = [
                "提升单实例处理能力：优化热点路径与锁竞争",
                "引入缓存减少重复计算与数据库访问",
                "水平扩容并配合负载均衡提高整体吞吐",
            ]
        else:
            bottleneck = {
                "type": "none",
                "description": "测试范围内未发现明显瓶颈",
            }
            suggestions = ["继续提高并发级别以探索系统极限"]
        result = {
            "measurements": measurements,
            "max_rps": max_rps,
            "saturation_point": saturation_point,
            "bottleneck": bottleneck,
            "suggestions": suggestions,
        }
        self._results["throughput_test"] = result
        self._record("throughput_test", {
            "max_rps": max_rps,
            "saturation_point": saturation_point,
            "bottleneck": bottleneck["type"],
        })
        return result

    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------
    def _trend(self) -> str:
        values = [h["metrics"]["avg"] for h in self._history
                  if h["type"] == "latency_test" and "avg" in h["metrics"]]
        if len(values) < 2:
            return "insufficient_data"
        first, last = values[0], values[-1]
        if first <= 0:
            return "stable"
        ratio = last / first
        if ratio < 0.95:
            return "improving"
        if ratio > 1.05:
            return "degrading"
        return "stable"

    def report(self, path: Optional[str] = None) -> Dict:
        """生成测试报告：包含摘要、ASCII 图表、历史记录与性能趋势。"""
        charts = {}
        latency_avgs = [h["metrics"]["avg"] for h in self._history
                        if h["type"] == "latency_test" and "avg" in h["metrics"]]
        if latency_avgs:
            charts["latency_trend"] = self._ascii_bar_chart(
                latency_avgs,
                ["run#{}".format(i + 1) for i in range(len(latency_avgs))])
        throughput = self._results.get("throughput_test")
        if throughput:
            charts["throughput"] = self._ascii_bar_chart(
                [m["requests_per_second"] for m in throughput["measurements"]],
                ["c={}".format(m["concurrency"])
                 for m in throughput["measurements"]])
        stress = self._results.get("stress_test")
        if stress:
            charts["stress_latency"] = self._ascii_bar_chart(
                [s["avg_latency"] for s in stress["stages"]],
                ["c={}".format(s["concurrency"]) for s in stress["stages"]])
        if not charts:
            charts["history"] = "(no test results yet)"
        trend = self._trend()
        summary = {
            "tests_run": len(self._history),
            "trend": trend,
            "available_results": sorted(self._results.keys()),
        }
        lines = ["=== 性能测试报告 ===",
                 "测试次数: {}".format(summary["tests_run"]),
                 "性能趋势: {}".format(trend)]
        for name, chart in charts.items():
            lines.append("")
            lines.append("[{}]".format(name))
            lines.append(chart)
        text = "\n".join(lines)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return {
            "summary": summary,
            "charts": charts,
            "history": list(self._history),
            "trend": trend,
            "text": text,
        }
