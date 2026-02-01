# -*- coding: utf-8 -*-
"""
指标计算模块
核心功能：
1. SA贡献度精准度验证：皮尔逊相关系数、MAE/RMSE（评估贡献度预测准确性）；
2. 公平性指标：基尼系数、综合公平性指数、方差/标准差等；
3. 鲁棒性指标：性能保持率、鲁棒性得分、性能波动系数等；
4. 效率指标：时间/资源/通信效率相关量化；
5. 统一封装MetricsCalculator类，支持一站式指标计算。
"""

import time
import numpy as np
import psutil
import torch
from scipy.stats import pearsonr, variation
from typing import Dict, List, Optional, Tuple, Union

# ======================== 核心：SA贡献度精准度验证指标 ========================
def calculate_pearson_corr(
    true_contributions: Union[np.ndarray, List[float]],
    pred_contributions: Union[np.ndarray, List[float]]
) -> float:
    """
    计算皮尔逊相关系数，验证SA贡献度的预测精准度
    （核心指标：越接近1表示预测贡献度与真实贡献度的线性相关性越强，精准度越高）
    Args:
        true_contributions: 真实贡献度数组（如人工标注/理论Shapley值）
        pred_contributions: SA算法预测的贡献度数组
    Returns:
        corr: 皮尔逊相关系数（-1~1），异常时返回0.0
    """
    try:
        # 转换为numpy数组并清理无效值
        true_arr = np.array(true_contributions, dtype=np.float64).flatten()
        pred_arr = np.array(pred_contributions, dtype=np.float64).flatten()
        
        # 过滤NaN/Inf值
        valid_mask = np.isfinite(true_arr) & np.isfinite(pred_arr)
        true_arr = true_arr[valid_mask]
        pred_arr = pred_arr[valid_mask]
        
        if len(true_arr) < 2 or len(pred_arr) < 2:
            return 0.0
        
        corr, _ = pearsonr(true_arr, pred_arr)
        return float(corr) if np.isfinite(corr) else 0.0
    except Exception:
        return 0.0

def calculate_contribution_error_metrics(
    true_contributions: Union[np.ndarray, List[float]],
    pred_contributions: Union[np.ndarray, List[float]]
) -> Dict[str, float]:
    """
    计算SA贡献度预测的误差指标（辅助验证精准度）
    Args:
        true_contributions: 真实贡献度数组
        pred_contributions: 预测贡献度数组
    Returns:
        error_metrics: 误差指标字典（MAE/RMSE/MAPE）
    """
    try:
        true_arr = np.array(true_contributions, dtype=np.float64).flatten()
        pred_arr = np.array(pred_contributions, dtype=np.float64).flatten()
        
        # 过滤无效值
        valid_mask = np.isfinite(true_arr) & np.isfinite(pred_arr)
        true_arr = true_arr[valid_mask]
        pred_arr = pred_arr[valid_mask]
        
        if len(true_arr) == 0:
            return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}
        
        # 平均绝对误差（MAE）
        mae = np.mean(np.abs(true_arr - pred_arr))
        # 均方根误差（RMSE）
        rmse = np.sqrt(np.mean((true_arr - pred_arr) ** 2))
        # 平均绝对百分比误差（MAPE）（避免除0）
        mape = np.mean(np.abs((true_arr - pred_arr) / (true_arr + 1e-8))) * 100
        
        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape)
        }
    except Exception:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

# ======================== 公平性指标（复用公平性验证实验） ========================
def calculate_gini_coefficient(values: Union[np.ndarray, List[float]]) -> float:
    """
    计算基尼系数（衡量客户端性能分布公平性，0~1，0=完全公平，1=完全不公平）
    Args:
        values: 客户端性能指标列表（如准确率、损失）
    Returns:
        gini: 基尼系数，异常时返回0.0
    """
    try:
        values = np.array(values, dtype=np.float64).flatten()
        if len(values) == 0 or np.all(values == values[0]):
            return 0.0
        
        values_sorted = np.sort(values)
        n = len(values_sorted)
        cumsum = np.cumsum(values_sorted)
        
        # 基尼系数核心公式
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        return float(np.clip(gini, 0.0, 1.0))  # 限制在0~1范围内
    except Exception:
        return 0.0

def calculate_fairness_metrics(
    client_performances: Union[Dict[int, float], List[float]]
) -> Dict[str, float]:
    """
    计算多维度公平性指标（整合基尼系数、方差、极差等）
    Args:
        client_performances: 客户端性能字典 {client_id: perf} 或列表
    Returns:
        fairness_metrics: 公平性指标字典
    """
    try:
        # 统一转换为数组
        if isinstance(client_performances, dict):
            performances = np.array(list(client_performances.values()), dtype=np.float64)
        else:
            performances = np.array(client_performances, dtype=np.float64)
        
        performances = performances[np.isfinite(performances)]  # 过滤无效值
        if len(performances) == 0:
            return {
                "mean": 0.0, "std": 0.0, "var": 0.0, "cv": 0.0,
                "min": 0.0, "max": 0.0, "range": 0.0, "gini": 0.0,
                "fairness_index": 0.0
            }
        
        # 基础统计量
        mean_perf = np.mean(performances)
        std_perf = np.std(performances)
        var_perf = np.var(performances)
        cv_perf = variation(performances) if mean_perf != 0 else 0.0  # 变异系数
        min_perf = np.min(performances)
        max_perf = np.max(performances)
        range_perf = max_perf - min_perf
        
        # 核心公平性指标
        gini = calculate_gini_coefficient(performances)
        
        # 综合公平性指数（0~1，越高越公平）
        fairness_index = (1 - gini) * (1 - cv_perf) * (min_perf / (mean_perf + 1e-8))
        fairness_index = np.clip(fairness_index, 0.0, 1.0)
        
        return {
            "mean": float(mean_perf),
            "std": float(std_perf),
            "var": float(var_perf),
            "cv": float(cv_perf),          # 变异系数（相对离散程度）
            "min": float(min_perf),
            "max": float(max_perf),
            "range": float(range_perf),    # 性能极差
            "gini": float(gini),           # 核心公平性指标
            "fairness_index": float(fairness_index)  # 综合公平性指数
        }
    except Exception as e:
        return {
            "mean": 0.0, "std": 0.0, "var": 0.0, "cv": 0.0,
            "min": 0.0, "max": 0.0, "range": 0.0, "gini": 0.0,
            "fairness_index": 0.0
        }

# ======================== 鲁棒性指标（复用效率鲁棒性验证实验） ========================
def calculate_robustness_metrics(
    baseline_perf: float,
    perturbed_perfs: Union[np.ndarray, List[float]]
) -> Dict[str, float]:
    """
    计算鲁棒性指标（衡量算法在扰动场景下的稳定性）
    Args:
        baseline_perf: 基准场景下的性能（如准确率）
        perturbed_perfs: 扰动场景（规模/噪声/故障/异构）下的性能列表
    Returns:
        robustness_metrics: 鲁棒性指标字典
    """
    try:
        perturbed_arr = np.array(perturbed_perfs, dtype=np.float64).flatten()
        perturbed_arr = perturbed_arr[np.isfinite(perturbed_arr)]
        
        if baseline_perf == 0 or len(perturbed_arr) == 0:
            return {
                "baseline_perf": 0.0, "avg_retention_rate_pct": 0.0,
                "perf_std": 0.0, "perf_cv": 0.0, "robustness_score": 0.0
            }
        
        # 性能保持率（%）：扰动后性能 / 基准性能 * 100
        retention_rates = (perturbed_arr / baseline_perf) * 100
        avg_retention_rate = np.mean(retention_rates)
        
        # 性能波动指标
        perf_std = np.std(perturbed_arr)
        perf_mean = np.mean(perturbed_arr)
        perf_cv = perf_std / perf_mean if perf_mean != 0 else 0.0
        
        # 鲁棒性得分（0~1，越高越鲁棒）：综合保持率和波动
        robustness_score = (avg_retention_rate / 100) * (1 - perf_cv)
        robustness_score = np.clip(robustness_score, 0.0, 1.0)
        
        return {
            "baseline_perf": float(baseline_perf),
            "avg_retention_rate_pct": float(avg_retention_rate),
            "perf_std": float(perf_std),
            "perf_cv": float(perf_cv),          # 性能变异系数
            "robustness_score": float(robustness_score)  # 综合鲁棒性得分
        }
    except Exception:
        return {
            "baseline_perf": 0.0, "avg_retention_rate_pct": 0.0,
            "perf_std": 0.0, "perf_cv": 0.0, "robustness_score": 0.0
        }

# ======================== 效率指标（复用效率鲁棒性验证实验） ========================
def calculate_efficiency_metrics(
    start_time: float,
    end_time: float,
    client_params_sizes: Optional[List[List[int]]] = None,
    process: Optional[psutil.Process] = None
) -> Dict[str, float]:
    """
    计算效率指标（时间/资源/通信）
    Args:
        start_time: 训练开始时间戳
        end_time: 训练结束时间戳
        client_params_sizes: 每轮各客户端传输参数大小（字节），格式[[round1_c1, round1_c2], [round2_c1, ...]]
        process: psutil.Process对象（用于监控资源占用）
    Returns:
        efficiency_metrics: 效率指标字典
    """
    try:
        # 时间效率
        total_time = end_time - start_time
        num_rounds = len(client_params_sizes) if client_params_sizes else 0
        avg_round_time = total_time / num_rounds if num_rounds > 0 else 0.0
        
        # 资源效率（内存/CPU/GPU）
        memory_usage = 0.0
        cpu_usage = 0.0
        gpu_memory = 0.0
        
        if process is not None:
            memory_usage = process.memory_info().rss / (1024 * 1024)  # 转换为MB
            cpu_usage = process.cpu_percent()
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # GPU显存（MB）
        
        # 通信效率
        total_comm_bytes = 0.0
        if client_params_sizes:
            total_comm_bytes = sum([sum(sizes) for sizes in client_params_sizes if sizes])
        total_comm_mb = total_comm_bytes / (1024 * 1024)  # 转换为MB
        avg_round_comm_mb = total_comm_mb / num_rounds if num_rounds > 0 else 0.0
        
        return {
            # 时间效率
            "total_time": float(total_time),
            "avg_round_time": float(avg_round_time),
            # 资源效率
            "memory_usage_mb": float(memory_usage),
            "cpu_usage_pct": float(cpu_usage),
            "gpu_memory_mb": float(gpu_memory),
            # 通信效率
            "total_comm_mb": float(total_comm_mb),
            "avg_round_comm_mb": float(avg_round_comm_mb)
        }
    except Exception:
        return {
            "total_time": 0.0, "avg_round_time": 0.0,
            "memory_usage_mb": 0.0, "cpu_usage_pct": 0.0, "gpu_memory_mb": 0.0,
            "total_comm_mb": 0.0, "avg_round_comm_mb": 0.0
        }

# ======================== 统一指标计算类（简化实验调用） ========================
class MetricsCalculator:
    """
    一站式指标计算类，整合所有实验所需指标，简化调用流程
    """
    def __init__(self):
        self.process = psutil.Process(os.getpid()) if psutil is not None else None
        self.results = {}  # 存储计算后的所有指标
    
    # SA贡献度精准度
    def calculate_sa_contribution_metrics(
        self,
        true_contributions: Union[np.ndarray, List[float]],
        pred_contributions: Union[np.ndarray, List[float]],
        save_key: str = "sa_contribution"
    ) -> Dict[str, float]:
        """计算SA贡献度精准度指标并存储"""
        pearson_corr = calculate_pearson_corr(true_contributions, pred_contributions)
        error_metrics = calculate_contribution_error_metrics(true_contributions, pred_contributions)
        
        metrics = {
            "pearson_corr": pearson_corr,
            **error_metrics
        }
        self.results[save_key] = metrics
        return metrics
    
    # 公平性指标
    def calculate_fairness(
        self,
        client_performances: Union[Dict[int, float], List[float]],
        save_key: str = "fairness"
    ) -> Dict[str, float]:
        """计算公平性指标并存储"""
        metrics = calculate_fairness_metrics(client_performances)
        self.results[save_key] = metrics
        return metrics
    
    # 鲁棒性指标
    def calculate_robustness(
        self,
        baseline_perf: float,
        perturbed_perfs: Union[np.ndarray, List[float]],
        save_key: str = "robustness"
    ) -> Dict[str, float]:
        """计算鲁棒性指标并存储"""
        metrics = calculate_robustness_metrics(baseline_perf, perturbed_perfs)
        self.results[save_key] = metrics
        return metrics
    
    # 效率指标
    def calculate_efficiency(
        self,
        start_time: float,
        end_time: float,
        client_params_sizes: Optional[List[List[int]]] = None,
        save_key: str = "efficiency"
    ) -> Dict[str, float]:
        """计算效率指标并存储"""
        metrics = calculate_efficiency_metrics(start_time, end_time, client_params_sizes, self.process)
        self.results[save_key] = metrics
        return metrics
    
    # 批量计算所有指标（适用于完整实验）
    def calculate_all(
        self,
        sa_true: Optional[Union[np.ndarray, List[float]]] = None,
        sa_pred: Optional[Union[np.ndarray, List[float]]] = None,
        client_perfs: Optional[Union[Dict[int, float], List[float]]] = None,
        robust_baseline: Optional[float] = None,
        robust_perturbed: Optional[Union[np.ndarray, List[float]]] = None,
        eff_start: Optional[float] = None,
        eff_end: Optional[float] = None,
        eff_params: Optional[List[List[int]]] = None
    ) -> Dict[str, Dict[str, float]]:
        """批量计算所有需要的指标"""
        if sa_true is not None and sa_pred is not None:
            self.calculate_sa_contribution_metrics(sa_true, sa_pred)
        if client_perfs is not None:
            self.calculate_fairness(client_perfs)
        if robust_baseline is not None and robust_perturbed is not None:
            self.calculate_robustness(robust_baseline, robust_perturbed)
        if eff_start is not None and eff_end is not None:
            self.calculate_efficiency(eff_start, eff_end, eff_params)
        return self.results
    
    def get_all_results(self) -> Dict[str, Dict[str, float]]:
        """获取所有已计算的指标"""
        return self.results
    
    def reset(self):
        """重置指标存储"""
        self.results = {} 