import torch
import numpy as np

class AgentBase:
    """强化学习智能体基类"""
    def __init__(self, state_dim, hidden_dim, action_dim, device,
                 learning_rate=1e-3, gamma=0.99, **kwargs):
        # 环境相关参数
        self.state_dim = state_dim  # 状态维度
        self.action_dim = action_dim  # 动作维度
        self.hidden_dim = hidden_dim  # 隐藏维度
        self.device = device  # 计算设备

        # 学习参数
        self.learning_rate = learning_rate  # 学习率
        self.gamma = gamma  # 折扣因子

        # 神经网络组件占位符
        self.policy_net = None  # 策略网络
        self.target_net = None  # 目标网络
        self.optimizer = None  # 优化器

        # 其他可扩展参数
        self._init_additional_params(**kwargs)

    def _init_additional_params(self, **kwargs):
        """用于子类扩展初始化额外参数"""
        pass

    def take_action(self, state):
        """选择动作的接口方法"""
        raise NotImplementedError("子类必须实现take_action()方法")

    def update(self, transition_dict):
        """更新策略的接口方法"""
        raise NotImplementedError("子类必须实现update()方法")

    def _soft_update(self, target, source, tau):
        """软更新模板方法"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def _preprocess_state(self, state):
        """状态预处理模板方法"""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        return state

    def save_model(self, path):
        """模型保存模板方法"""
        if self.policy_net:
            torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        """模型加载模板方法"""
        if self.policy_net:
            self.policy_net.load_state_dict(torch.load(path))
            self.policy_net.eval()
