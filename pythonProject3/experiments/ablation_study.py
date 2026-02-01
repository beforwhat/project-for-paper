# experiments/ablation_study.py
"""
组件消融实验脚本
核心目标：
1. 消融5大核心组件，验证每个组件的必要性和收益：
   - SA贡献度（FedShap的Shapley权重替换为等权重）；
   - 优化后自适应裁剪DP（DP-FedAvg改用固定裁剪阈值）；
   - FedProx近端项（关闭FedProx的Proximal Term，退化为FedAvg）；
   - Ditto个性化正则（关闭Ditto的正则项，个性化模型完全自由训练）；
   - 自适应裁剪（DP-FedAvg改用无裁剪仅加噪）；
2. 严格遵循单一变量原则：仅关闭目标组件，其余参数/流程与基准版本完全一致；
3. 记录核心指标（性能：准确率/损失；隐私：ε有效值；公平性：基尼系数；效率：耗时）；
4. 输出消融对比报告、量化收益分析和可视化图表。
设计原则：
- 复用基础性能实验的核心框架，保证实验流程一致；
- 每个组件的“基准版本”和“消融版本”仅差异目标组件，其余完全对齐；
- 结果结构化保存，支持量化分析组件贡献度。
"""
import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

# 项目内模块导入
from configs.config_loader import load_config
from baselines import (
    FedAvgServer, FedAvgClient,
    DPFedAvgServer, DPFedAvgClient,
    FedProxServer, FedProxClient,
    DittoServer, DittoClient,
    FedShapServer, FedShapClient
)
from core.dp.adaptive_clipping_dp import AdaptiveClippingDP
from core.shap.shapley_calculator import ShapleyCalculator

# 可视化配置（与基础性能实验保持一致）
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
PLOT_FORMAT = "png"
PLOT_DPI = 300
COMPONENT_COLORS = {
    "基准版本": "#1f77b4",
    "消融版本": "#d62728"
}

# ======================== 消融组件定义（核心：单一变量） ========================
# 定义待消融的组件列表，每个组件包含：
# - name: 组件名称
# - baseline_alg: 基准算法（带组件）
# - ablation_alg: 消融算法（无组件，通过重写实现）
# - metrics: 关注的核心指标（performance/privacy/fairness/efficiency）
ABLATION_COMPONENTS = [
    {
        "name": "SA贡献度（FedShap）",
        "description": "FedShap的Shapley权重聚合 → 等权重聚合（退化为FedAvg）",
        "baseline": {
            "server_cls": FedShapServer,
            "client_cls": FedShapClient,
            "requires_global_distribution": False
        },
        "ablation": {
            "server_cls": FedShapServer,  # 重写聚合逻辑为等权重
            "client_cls": FedShapClient,
            "requires_global_distribution": False,
            "override_aggregate": True  # 标记需要重写聚合逻辑
        },
        "focus_metrics": ["performance", "fairness", "efficiency"]
    },
    {
        "name": "优化后自适应裁剪DP（DP-FedAvg）",
        "description": "DP-FedAvg的自适应裁剪 → 固定裁剪阈值",
        "baseline": {
            "server_cls": DPFedAvgServer,
            "client_cls": DPFedAvgClient,
            "requires_global_distribution": False
        },
        "ablation": {
            "server_cls": DPFedAvgServer,
            "client_cls": DPFedAvgClient,
            "requires_global_distribution": False,
            "override_dp_clip": True  # 标记需要重写DP裁剪逻辑
        },
        "focus_metrics": ["performance", "privacy", "efficiency"]
    },
    {
        "name": "FedProx近端项",
        "description": "FedProx的近端正则项 → 关闭（退化为FedAvg）",
        "baseline": {
            "server_cls": FedProxServer,
            "client_cls": FedProxClient,
            "requires_global_distribution": True
        },
        "ablation": {
            "server_cls": FedProxServer,
            "client_cls": FedProxClient,
            "requires_global_distribution": True,
            "override_proximal": True  # 标记需要关闭近端项
        },
        "focus_metrics": ["performance", "stability"]
    },
    {
        "name": "Ditto个性化正则",
        "description": "Ditto的个性化正则项 → 关闭（个性化模型完全自由训练）",
        "baseline": {
            "server_cls": DittoServer,
            "client_cls": DittoClient,
            "requires_global_distribution": True
        },
        "ablation": {
            "server_cls": DittoServer,
            "client_cls": DittoClient,
            "requires_global_distribution": True,
            "override_personal_reg": True  # 标记需要关闭个性化正则
        },
        "focus_metrics": ["performance", "personalization"]
    },
    {
        "name": "DP自适应裁剪（DP-FedAvg）",
        "description": "DP-FedAvg的自适应裁剪 → 无裁剪仅加噪",
        "baseline": {
            "server_cls": DPFedAvgServer,
            "client_cls": DPFedAvgClient,
            "requires_global_distribution": False
        },
        "ablation": {
            "server_cls": DPFedAvgServer,
            "client_cls": DPFedAvgClient,
            "requires_global_distribution": False,
            "override_clip_none": True  # 标记需要关闭裁剪
        },
        "focus_metrics": ["performance", "privacy"]
    }
]

# ======================== 消融版本算法重写（核心：单一变量） ========================
class AblationFedShapServer(FedShapServer):
    """消融SA贡献度：FedShap聚合逻辑改为等权重（退化为FedAvg）"""
    def aggregate_local_results(self, client_results_list, client_ids=None):
        # 提取客户端参数（忽略Shapley指标）
        client_params = [res["params"] for res in client_results_list]
        # 调用FedAvg的等权重聚合逻辑
        return super(FedAvgServer, self).aggregate_local_results(client_params_list=client_params)

class AblationDPFedAvgClient(DPFedAvgClient):
    """消融自适应裁剪：改为固定裁剪阈值"""
    def __init__(self, client_id, config):
        super().__init__(client_id, config)
        # 重写DP优化器为固定裁剪
        self.dp_optimizer = AdaptiveClippingDP(config=config)
        self.dp_optimizer.adaptive = False  # 关闭自适应，使用固定阈值
        self.dp_optimizer.clip_threshold = config.dp.base_clip_threshold  # 固定阈值

class AblationDPFedAvgClientNoClip(DPFedAvgClient):
    """消融裁剪：仅加噪，无裁剪"""
    def __init__(self, client_id, config):
        super().__init__(client_id, config)
        self.dp_optimizer.clip = False  # 关闭裁剪

class AblationFedProxClient(FedProxClient):
    """消融近端项：关闭Proximal Term"""
    def _calculate_proximal_term(self):
        return torch.tensor(0.0, device=self.device)  # 近端项为0

class AblationDittoClient(DittoClient):
    """消融个性化正则：关闭Ditto的正则项"""
    def _calculate_personal_regularization(self):
        return torch.tensor(0.0, device=self.device)  # 正则项为0

# ======================== 核心实验类 ========================
class AblationStudyExperiment:
    def __init__(self, config=None, save_results=True, save_path="./experiment_results/ablation_study"):
        """
        初始化消融实验
        Args:
            config: 配置对象
            save_results: 是否保存结果
            save_path: 结果保存路径
        """
        self.config = config if config is not None else load_config()
        self.save_results = save_results
        self.save_path = save_path
        self.device = torch.device(self.config.device)
        
        # 创建保存目录
        if self.save_results:
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs(os.path.join(self.save_path, "plots"), exist_ok=True)
            os.makedirs(os.path.join(self.save_path, "data"), exist_ok=True)
        
        # 实验结果存储
        self.ablation_results = {
            "component_metrics": {},  # 每个组件的基准/消融指标
            "gain_analysis": {}       # 组件收益分析（准确率提升、隐私改善等）
        }
        
        print(f"✅ 组件消融实验初始化完成 | 待消融组件数：{len(ABLATION_COMPONENTS)}")
        print(f"📌 实验配置：全局轮次={self.config.fed.global_rounds} | 客户端数={self.config.fed.num_clients}")

    def _prepare_algorithm(self, component, version):
        """
        准备基准/消融版本的算法（替换重写的类）
        Args:
            component: 组件配置
            version: baseline/ablation
        Returns:
            alg_config: 处理后的算法配置
        """
        alg_config = component[version].copy()
        
        # 根据消融标记替换对应的类
        if version == "ablation":
            # 消融SA贡献度
            if component["name"] == "SA贡献度（FedShap）" and alg_config.get("override_aggregate"):
                alg_config["server_cls"] = AblationFedShapServer
            # 消融自适应裁剪（固定阈值）
            elif component["name"] == "优化后自适应裁剪DP（DP-FedAvg）" and alg_config.get("override_dp_clip"):
                alg_config["client_cls"] = AblationDPFedAvgClient
            # 消融FedProx近端项
            elif component["name"] == "FedProx近端项" and alg_config.get("override_proximal"):
                alg_config["client_cls"] = AblationFedProxClient
            # 消融Ditto个性化正则
            elif component["name"] == "Ditto个性化正则" and alg_config.get("override_personal_reg"):
                alg_config["client_cls"] = AblationDittoClient
            # 消融DP裁剪（仅加噪）
            elif component["name"] == "DP自适应裁剪（DP-FedAvg）" and alg_config.get("override_clip_none"):
                alg_config["client_cls"] = AblationDPFedAvgClientNoClip
        
        return alg_config

    def _run_algorithm_version(self, component_name, alg_config):
        """
        运行单个版本（基准/消融）的算法，记录核心指标
        """
        print(f"\n--- 运行 {component_name} | 版本：{alg_config.get('version', 'unknown')} ---")
        start_time = time.time()
        
        # 1. 初始化服务端和客户端
        server = alg_config["server_cls"](config=self.config)
        server.global_model.to(self.device)
        
        clients = []
        for cid in range(self.config.fed.num_clients):
            client = alg_config["client_cls"](client_id=cid, config=self.config)
            client.local_model.to(self.device)
            clients.append(client)
        server.clients = clients
        
        # 2. 训练指标记录
        global_acc_list = []
        global_loss_list = []
        client_acc_list = []
        dp_epsilon_list = []  # 隐私预算记录（仅DP相关组件）
        gini_coefficient_list = []  # 公平性指标（基尼系数）
        
        # 3. 多轮训练
        for round_idx in range(self.config.fed.global_rounds):
            round_start = time.time()
            
            # 选择客户端
            selected_cids = server.select_clients(round_idx=round_idx)
            
            # 下发全局模型（如需）
            if alg_config["requires_global_distribution"]:
                server.distribute_global_model(selected_client_ids=selected_cids)
            
            # 客户端训练
            client_outputs = []
            for cid in selected_cids:
                output = clients[cid].local_train()
                client_outputs.append(output)
            
            # 聚合
            if "FedShap" in component_name:
                server.aggregate_local_results(client_results_list=client_outputs)
            else:
                client_params = [o for o in client_outputs]
                server.aggregate_local_results(client_params_list=client_params)
            
            # 评估全局指标
            global_acc, global_loss = server.evaluate_global_model()
            global_acc_list.append(global_acc)
            global_loss_list.append(global_loss)
            
            # 评估客户端准确率（计算基尼系数）
            client_accs = [clients[cid].evaluate_local_model() for cid in range(self.config.fed.num_clients)]
            client_acc_list.append(np.mean(client_accs))
            gini = self._calculate_gini(client_accs)
            gini_coefficient_list.append(gini)
            
            # 记录DP隐私预算（仅DP组件）
            if "DP-FedAvg" in component_name:
                dp_epsilon = clients[0].dp_optimizer.calculate_epsilon()  # 计算有效ε
                dp_epsilon_list.append(dp_epsilon)
            
            print(f"轮次 {round_idx+1} | 全局准确率：{global_acc:.2f}% | 基尼系数：{gini:.4f}")
        
        # 4. 计算汇总指标
        total_time = time.time() - start_time
        final_global_acc = global_acc_list[-1]
        final_global_loss = global_loss_list[-1]
        avg_client_acc = np.mean(client_acc_list)
        final_gini = gini_coefficient_list[-1]
        avg_dp_epsilon = np.mean(dp_epsilon_list) if dp_epsilon_list else 0.0
        
        # 5. 个性化指标（仅Ditto）
        personal_gain = 0.0
        if "Ditto" in component_name:
            baseline_acc = np.mean([clients[cid].evaluate_local_model() for cid in range(self.config.fed.num_clients)])
            personal_acc = np.mean([clients[cid].evaluate_personal_model() for cid in range(self.config.fed.num_clients)])
            personal_gain = personal_acc - baseline_acc
        
        return {
            "global_acc": global_acc_list,
            "global_loss": global_loss_list,
            "gini_coefficient": gini_coefficient_list,
            "dp_epsilon": dp_epsilon_list,
            "final_global_acc": final_global_acc,
            "final_global_loss": final_global_loss,
            "avg_client_acc": avg_client_acc,
            "final_gini": final_gini,
            "avg_dp_epsilon": avg_dp_epsilon,
            "personal_gain": personal_gain,
            "total_time": total_time,
            "avg_round_time": total_time / self.config.fed.global_rounds
        }

    def _calculate_gini(self, values):
        """计算基尼系数（衡量公平性，越小越公平）"""
        if len(values) == 0:
            return 0.0
        values = np.array(values)
        values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    def run(self):
        """运行所有组件的消融实验"""
        for component in ABLATION_COMPONENTS:
            comp_name = component["name"]
            print(f"\n========== 开始消融实验：{comp_name} ==========")
            print(f"组件描述：{component['description']}")
            
            # 1. 运行基准版本（有组件）
            baseline_alg = self._prepare_algorithm(component, "baseline")
            baseline_alg["version"] = "基准版本"
            baseline_results = self._run_algorithm_version(comp_name, baseline_alg)
            
            # 2. 运行消融版本（无组件）
            ablation_alg = self._prepare_algorithm(component, "ablation")
            ablation_alg["version"] = "消融版本"
            ablation_results = self._run_algorithm_version(comp_name, ablation_alg)
            
            # 3. 计算组件收益
            gain = self._calculate_component_gain(baseline_results, ablation_results, component["focus_metrics"])
            
            # 4. 保存结果
            self.ablation_results["component_metrics"][comp_name] = {
                "baseline": baseline_results,
                "ablation": ablation_results,
                "description": component["description"]
            }
            self.ablation_results["gain_analysis"][comp_name] = gain
            
            # 5. 生成该组件的消融对比图
            if self.save_results:
                self._generate_component_plot(comp_name, baseline_results, ablation_results)
        
        # 保存所有结果
        if self.save_results:
            self._save_results()
        
        # 输出消融报告
        self._print_ablation_report()
        
        return self.ablation_results

    def _calculate_component_gain(self, baseline, ablation, focus_metrics):
        """计算组件收益（基准-消融的差值）"""
        gain = {}
        # 性能收益（准确率提升）
        if "performance" in focus_metrics:
            gain["accuracy_gain"] = baseline["final_global_acc"] - ablation["final_global_acc"]
            gain["loss_reduction"] = ablation["final_global_loss"] - baseline["final_global_loss"]
        # 隐私收益（ε降低，隐私保护更好）
        if "privacy" in focus_metrics:
            gain["epsilon_reduction"] = ablation["avg_dp_epsilon"] - baseline["avg_dp_epsilon"]
        # 公平性收益（基尼系数降低，更公平）
        if "fairness" in focus_metrics:
            gain["gini_reduction"] = ablation["final_gini"] - baseline["final_gini"]
        # 个性化收益
        if "personalization" in focus_metrics:
            gain["personal_gain_reduction"] = ablation["personal_gain"] - baseline["personal_gain"]
        # 效率收益
        if "efficiency" in focus_metrics:
            gain["time_reduction"] = ablation["total_time"] - baseline["total_time"]
        return gain

    def _generate_component_plot(self, comp_name, baseline, ablation):
        """生成单个组件的消融对比图"""
        # 1. 全局准确率对比
        plt.figure(figsize=(10, 5))
        rounds = list(range(1, self.config.fed.global_rounds+1))
        plt.plot(rounds, baseline["global_acc"], label="基准版本（有组件）", color=COMPONENT_COLORS["基准版本"], linewidth=2)
        plt.plot(rounds, ablation["global_acc"], label="消融版本（无组件）", color=COMPONENT_COLORS["消融版本"], linewidth=2, linestyle="--")
        plt.title(f"{comp_name} - 全局准确率收敛对比", fontsize=12, fontweight="bold")
        plt.xlabel("全局轮次")
        plt.ylabel("全局准确率（%）")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(self.save_path, "plots", f"{comp_name}_acc.{PLOT_FORMAT}")
        plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        
        # 2. 公平性（基尼系数）对比（如适用）
        if baseline["gini_coefficient"]:
            plt.figure(figsize=(10, 5))
            plt.plot(rounds, baseline["gini_coefficient"], label="基准版本（有组件）", color=COMPONENT_COLORS["基准版本"], linewidth=2)
            plt.plot(rounds, ablation["gini_coefficient"], label="消融版本（无组件）", color=COMPONENT_COLORS["消融版本"], linewidth=2, linestyle="--")
            plt.title(f"{comp_name} - 基尼系数对比（越小越公平）", fontsize=12, fontweight="bold")
            plt.xlabel("全局轮次")
            plt.ylabel("基尼系数")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plot_path = os.path.join(self.save_path, "plots", f"{comp_name}_gini.{PLOT_FORMAT}")
            plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches="tight")
            plt.close()

    def _save_results(self):
        """保存消融实验结果"""
        # 1. 组件指标（JSON）
        metrics_path = os.path.join(self.save_path, "data", "component_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.ablation_results["component_metrics"], f, ensure_ascii=False, indent=4)
        
        # 2. 收益分析（CSV）
        gain_df = pd.DataFrame.from_dict(self.ablation_results["gain_analysis"], orient="index")
        gain_df.reset_index(inplace=True)
        gain_df.rename(columns={"index": "component"}, inplace=True)
        gain_path = os.path.join(self.save_path, "data", "gain_analysis.csv")
        gain_df.to_csv(gain_path, index=False, encoding="utf-8")
        
        print(f"\n📁 消融实验结果已保存至：{self.save_path}")

    def _print_ablation_report(self):
        """打印消融实验最终报告"""
        print("\n========== 组件消融实验 - 最终报告 ==========")
        print(f"{'组件名称':<20} {'准确率收益(%)':<15} {'基尼系数收益':<15} {'ε降低值':<15} {'个性化收益(%)':<15}")
        print("-" * 80)
        
        for comp_name, gain in self.ablation_results["gain_analysis"].items():
            acc_gain = gain.get("accuracy_gain", 0.0)
            gini_gain = gain.get("gini_reduction", 0.0)
            eps_gain = gain.get("epsilon_reduction", 0.0)
            personal_gain = gain.get("personal_gain_reduction", 0.0)
            
            print(
                f"{comp_name:<20} "
                f"{acc_gain:<15.2f} "
                f"{gini_gain:<15.4f} "
                f"{eps_gain:<15.2f} "
                f"{personal_gain:<15.2f}"
            )
        
        print("-" * 80)
        print("注：")
        print("1. 准确率收益>0：组件提升了全局准确率；")
        print("2. 基尼系数收益>0：组件提升了公平性（基尼系数降低）；")
        print("3. ε降低值>0：组件提升了隐私保护（有效ε更小）；")
        print("4. 个性化收益>0：组件提升了Ditto的个性化效果。")

# ======================== 外部调用函数 ========================
def run_ablation_study_experiment(config=None, save_results=True, save_path="./experiment_results/ablation_study"):
    """
    外部调用的核心函数：运行组件消融实验
    Args:
        config: 配置对象
        save_results: 是否保存结果
        save_path: 结果保存路径
    Returns:
        ablation_results: 消融实验结果
    """
    experiment = AblationStudyExperiment(config=config, save_results=save_results, save_path=save_path)
    results = experiment.run()
    return results

# ======================== 主函数 ========================
if __name__ == "__main__":
    results = run_ablation_study_experiment(
        save_results=True,
        save_path="./experiment_results/ablation_study_2026"
    )
    print("\n✅ 组件消融实验全部完成！")