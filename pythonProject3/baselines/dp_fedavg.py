# baselines/dp_fedavg.py
"""
带差分隐私的FedAvg（DP-FedAvg）算法实现
核心定位：FedAvg基础上新增差分隐私保护，作为隐私保护类联邦算法的基准
核心逻辑：
1. 客户端（DPFedAvgClient）：继承FedAvgClient，仅在本地训练的梯度环节加入「自适应裁剪+高斯噪声」；
2. 服务端（DPFedAvgServer）：完全复用FedAvgServer的等权重聚合逻辑，无任何修改；
设计原则：
- 仅新增DP相关逻辑，其余完全复用FedAvg（保证与基础FedAvg的唯一差异是DP）；
- 复用项目核心的AdaptiveClippingDP模块（自适应裁剪），保证DP逻辑的一致性；
- 接口与FedAvg完全对齐，便于公平对比（仅多DP配置项）。
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# 项目内模块导入
from baselines.fedavg import FedAvgClient, FedAvgServer  # 复用基础FedAvg
from core.dp.adaptive_clipping_dp import AdaptiveClippingDP  # 复用自适应裁剪DP模块
from configs.config_loader import load_config

class DPFedAvgClient(FedAvgClient):
    """
    DP-FedAvg客户端：继承FedAvgClient，仅新增梯度的DP保护（自适应裁剪+加噪）
    核心修改：local_train方法中，反向传播后对梯度做DP处理，再执行优化器step
    """
    def __init__(self, client_id: int, config=None):
        """
        初始化DP-FedAvg客户端（复用FedAvgClient初始化，新增DP优化器）
        Args:
            client_id: 客户端唯一标识
            config: 配置对象（需包含dp相关配置：epsilon/delta/clip_threshold等）
        """
        super().__init__(client_id=client_id, config=config)
        
        # 初始化自适应裁剪DP优化器（核心新增：DP隐私保护）
        self.dp_optimizer = AdaptiveClippingDP(config=config)
        
        print(f"✅ DPFedAvg客户端 [{self.client_id}] 初始化完成（基于FedAvg + 自适应裁剪DP）")
        print(f"📌 DP配置：ε={self.config.dp.epsilon} | δ={self.config.dp.delta} | 基础裁剪阈值={self.config.dp.base_clip_threshold}")

    def local_train(self):
        """
        重写FedAvgClient的local_train：仅新增梯度的DP保护步骤
        核心流程：前向传播→损失计算→反向传播→DP梯度处理→梯度下降
        """
        # 1. 初始化训练环境（完全复用FedAvg的逻辑）
        self.local_model.train()
        optimizer = optim.SGD(
            self.local_model.parameters(),
            lr=self.config.fed.local_lr,
            momentum=self.config.fed.momentum
        )
        loss_fn = F.cross_entropy if self.config.model.num_classes > 2 else F.binary_cross_entropy_with_logits

        # 2. 本地训练循环（核心新增DP梯度处理）
        for epoch in range(self.config.fed.local_epochs):
            epoch_loss = 0.0
            total_samples = 0
            pbar = tqdm(self.local_dataloader, desc=f"DPFedAvg客户端 [{self.client_id}] 训练Epoch {epoch+1}")
            
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # 前向传播（复用FedAvg）
                optimizer.zero_grad()
                outputs = self.local_model(images)
                loss = loss_fn(outputs, labels)
                
                # 反向传播（复用FedAvg）
                loss.backward()

                # ==============================================
                # 核心新增：DP梯度保护（自适应裁剪 + 高斯噪声）
                # ==============================================
                # 提取当前梯度（按参数名存储）
                current_gradient_dict = {}
                for name, param in self.local_model.named_parameters():
                    if param.grad is not None:
                        current_gradient_dict[name] = param.grad.clone()
                
                # 自适应裁剪 + 添加DP噪声
                protected_gradient_dict = self.dp_optimizer.adaptive_clip_and_add_noise(
                    model=self.local_model,
                    current_gradient_dict=current_gradient_dict
                )
                
                # 替换为带DP保护的梯度
                for name, param in self.local_model.named_parameters():
                    if name in protected_gradient_dict:
                        param.grad = protected_gradient_dict[name]
                # ==============================================

                # 梯度下降（复用FedAvg）
                optimizer.step()
                
                # 统计损失（复用FedAvg）
                epoch_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
                pbar.set_postfix({"batch_loss": loss.item(), "avg_loss": epoch_loss/total_samples})

        # 3. 训练完成，记录损失（复用FedAvg）
        self.local_train_loss = epoch_loss / total_samples
        print(f"\n📌 DPFedAvg客户端 [{self.client_id}] 本地训练完成 | 平均损失：{self.local_train_loss:.4f}")

        # 4. 返回本地模型参数（复用FedAvg）
        return self.get_model_parameters()

class DPFedAvgServer(FedAvgServer):
    """
    DP-FedAvg服务端：完全复用FedAvgServer的等权重聚合逻辑
    核心：服务端无需任何DP相关处理（DP仅在客户端侧），保证与FedAvg的聚合逻辑一致
    """
    def __init__(self, config=None):
        """
        初始化DP-FedAvg服务端（完全复用FedAvgServer）
        Args:
            config: 配置对象（仅需FedAvg相关配置，无需DP配置）
        """
        super().__init__(config=config)
        print(f"✅ DPFedAvg服务端初始化完成（完全复用FedAvg等权重聚合，无额外修改）")

# ======================== 独立测试示例（验证DP-FedAvg功能） ========================
if __name__ == "__main__":
    """
    测试DP-FedAvg核心逻辑：客户端带DP训练 → 服务端等权重聚合
    对比FedAvg：仅客户端梯度多了DP处理，服务端完全一致
    """
    # 1. 加载配置（需包含dp配置）
    config = load_config()
    # 测试用配置
    config.fed.num_clients = 2
    config.fed.local_epochs = 1
    config.fed.local_lr = 0.01
    config.dp.epsilon = 1.0
    config.dp.delta = 1e-5
    config.dp.base_clip_threshold = 1.0
    config.dp.sliding_window_size = 3
    config.dp.threshold_change_rate = 0.2

    # 2. 初始化DP-FedAvg服务端
    dp_fedavg_server = DPFedAvgServer(config=config)

    # 3. 初始化DP-FedAvg客户端
    client_list = []
    for client_id in range(config.fed.num_clients):
        client = DPFedAvgClient(client_id=client_id, config=config)
        client_list.append(client)

    # 4. 模拟一轮联邦训练
    print("\n=== 模拟DP-FedAvg一轮联邦训练 ===")
    # 4.1 客户端本地训练（带DP）
    client_params_list = []
    for client in client_list:
        client_params = client.local_train()
        client_params_list.append(client_params)

    # 4.2 服务端聚合（复用FedAvg等权重）
    dp_fedavg_server.aggregate_local_results(client_params_list=client_params_list)

    # 4.3 打印结果
    print("\n=== DP-FedAvg一轮训练完成 ===")
    print(f"服务端全局模型参数示例（conv1.weight.shape）：{dp_fedavg_server.global_model.conv1.weight.shape}")
    for idx, client in enumerate(client_list):
        print(f"DP-FedAvg客户端 [{idx}] 本地训练损失：{client.local_train_loss:.4f}")