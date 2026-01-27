# datasets/non_iid_partitioner.py
import numpy as np
from collections import defaultdict

class NonIIDPartitioner:
    """
    Non-IID划分工具类：封装多种Non-IID划分算法，供所有数据集复用
    """
    def __init__(self, num_clients, alpha=0.1):
        self.num_clients = num_clients
        self.alpha = alpha  # Dirichlet分布的α值，越小Non-IID程度越高
    
    def dirichlet_partition(self, labels):
        """
        核心方法：Dirichlet分布划分（标签异质性，最常用）
        Args:
            labels: 原始数据集的标签数组（np.array）
        Returns:
            client_indices: 客户端样本索引列表（shape: [num_clients, num_samples_per_client]）
        """
        num_classes = len(np.unique(labels))
        client_indices = [[] for _ in range(self.num_clients)]
        
        # 1. 按类别分组，保存每个类别的样本索引
        class_to_samples = defaultdict(list)
        for idx, label in enumerate(labels):
            class_to_samples[label].append(idx)
        
        # 2. 对每个类别，用Dirichlet分布分配给各客户端
        for class_id, sample_indices in class_to_samples.items():
            # 生成每个客户端对该类别的分配比例（Dirichlet分布）
            class_proportions = np.random.dirichlet(
                alpha=[self.alpha] * self.num_clients,
                size=1
            ).flatten()
            
            # 按比例分配该类别的样本给各客户端
            num_samples_in_class = len(sample_indices)
            sample_indices_per_client = self._split_samples_by_proportion(
                sample_indices, class_proportions, num_samples_in_class
            )
            
            # 合并到客户端索引列表
            for client_id, indices in enumerate(sample_indices_per_client):
                client_indices[client_id].extend(indices)
        
        # 3. 打乱每个客户端的样本顺序（可选，提升训练稳定性）
        for client_id in range(self.num_clients):
            np.random.shuffle(client_indices[client_id])
        
        return client_indices
    
    def _split_samples_by_proportion(self, sample_indices, proportions, num_samples):
        """
        辅助方法：按比例拆分样本索引（内部调用，不对外暴露）
        """
        sample_indices_per_client = []
        start_idx = 0
        for prop in proportions:
            # 计算该客户端分配到的样本数量
            num_samples_for_client = int(prop * num_samples)
            # 处理最后一个客户端，避免遗漏样本
            if len(sample_indices_per_client) == self.num_clients - 1:
                num_samples_for_client = num_samples - start_idx
            # 拆分样本索引
            end_idx = start_idx + num_samples_for_client
            sample_indices_per_client.append(sample_indices[start_idx:end_idx])
            start_idx = end_idx
        return sample_indices_per_client
    
    def quantity_unbalanced_partition(self, labels, min_sample_ratio=0.1):
        """
        扩展方法：数量异质性划分（客户端样本数量不一致）
        可根据需求扩展，此处仅提供核心逻辑
        """
        # 1. 先按均匀划分得到基础索引
        uniform_indices = self.dirichlet_partition(labels)  # 复用Dirichlet标签划分
        # 2. 按比例调整各客户端样本数量（如随机删除部分样本）
        client_sample_nums = [len(indices) for indices in uniform_indices]
        min_samples = int(min(client_sample_nums) * min_sample_ratio)
        for client_id in range(self.num_clients):
            # 随机保留部分样本，实现数量异质性
            np.random.shuffle(uniform_indices[client_id])
            uniform_indices[client_id] = uniform_indices[client_id][:np.random.randint(min_samples, len(uniform_indices[client_id]))]
        return uniform_indices