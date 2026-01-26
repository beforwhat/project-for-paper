def get_dp_config():
    dp_config = {
        # 1. 基础差分隐私参数
        "enable_dp": True,  # 是否开启差分隐私，默认True
        "epsilon": 0.5,  # 目标隐私预算ε，默认0.5（强隐私约束）
        "delta": 1e-5,  # 隐私参数δ，默认1e-5（通常设为1/num_samples）
        "noise_multiplier": 1.0,  # 噪声乘数，默认1.0（噪声大小与裁剪阈值相关）


        "T0": 1.0,  # 初始裁剪阈值，默认1.0
        "T_min": 0.3,  # 最小裁剪阈值（0.3×T0），默认0.3
        "T_max": 2.0,  # 最大裁剪阈值（2.0×T0），默认2.0
        "lambda_": 0.07,  # 自适应裁剪基础调整系数，默认0.07（推荐0.05~0.1）
        "theta": 0.1,  # 微小波动过滤阈值，默认0.1（过滤小于0.1的归一化差值）
        "max_single_change": 0.2,  # 单轮阈值调整上限，默认0.2（不超过上一轮的20%）
        "smooth_coeff": 0.8,  # 阈值滑动平均平滑系数，默认0.8

        # 3. 梯度裁剪辅助参数
        "grad_norm_type": 2,  # 梯度范数计算类型，默认L2范数
        "clip_per_layer": False,  # 是否按层裁剪，默认False（按全局梯度裁剪）

        "moments_accountant_enable": True,  # 是否开启Moments Accountant，默认True
        "moments_accountant_order": 32,  # 矩估计阶数，默认32
    }

    return dp_config