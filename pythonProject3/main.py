from configs.config_loader import load_config
from datasets import SVHNDataset, FEMNISTDataset

# 1. 加载配置
config = load_config()

# 2. 验证SVHN
# print("=== 验证SVHN ===")
# svhn_client0 = SVHNDataset(config=config, is_train=True, client_id=0)
# svhn_dataloader = svhn_client0.get_dataloader()
# for images, labels in svhn_dataloader:
#     print(f"SVHN图片形状：{images.shape}")  # torch.Size([64, 3, 32, 32])
#     print(f"SVHN标签形状：{labels.shape}")  # torch.Size([64])
#     break

# 3. 验证FEMNIST（方案1）
print("\n=== 验证FEMNIST ===")
femnist_client0 = FEMNISTDataset(config=config, is_train=True, client_id=0)
femnist_dataloader = femnist_client0.get_dataloader()
for images, labels in femnist_dataloader:
    print(f"FEMNIST图片形状：{images.shape}")  # torch.Size([64, 1, 28, 28])
    print(f"FEMNIST标签形状：{labels.shape}")  # torch.Size([64])
    break