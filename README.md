# 🎨 NovelAI Discord Bot

一个简洁强大的Discord机器人，用于NovelAI图像生成。使用你自己的API密钥，轻松创建AI图像。
~~分享到DC群里开银趴~~

**所有代码来自Claude 3.7sonnet**

~~出事别找我~~

## ✨ 主要特性

- 🔑 **个人API密钥管理** - 支持个人使用或服务器共享
- 🖼️ **快速图像生成** - 简单的`/nai`命令即可生成图像
- ⚙️ **高级参数控制** - 通过`/naigen`命令完全掌控生成过程
- 🔄 **多模型支持** - 支持所有NovelAI模型，包括最新的nai-diffusion-4系列

## 📋 需求

- Python 3.8+
- Discord Bot Token
- NovelAI订阅与API密钥

## 🚀 快速开始

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/saltysalrua/nai-discordbot.git
cd nai-discordbot
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置机器人**
```bash
cp config.example.txt config.txt
# 编辑config.txt，填入你的Discord机器人令牌
```

4. **启动机器人**
```bash
python bot.py
# 或使用启动脚本
bash start.sh
```

## 💡 使用指南

### API密钥管理

```
/apikey [你的NovelAI密钥] [私人使用/服务器共享] [有效期小时数]
```

例如:
- `/apikey pst-abcdefg12345 私人使用 24` - 注册24小时有效的私人密钥
- `/apikey pst-abcdefg12345 服务器共享 0` - 注册永不过期的服务器共享密钥

### 图像生成

**基础命令:**
```
/nai [提示词] [模型]
```

**高级命令:**
```
/naigen [提示词] [参数选项...]
```

高级参数包括:
- 模型选择
- 图像尺寸
- 采样步数
- CFG比例
- 采样器类型
- 噪声调度
- 负面提示词
- SMEA设置
- 种子值

### 实用命令

```
/help            - 显示帮助信息
/sharedkeys      - 查看当前服务器中共享的API密钥
/checkapi        - 检查NovelAI API状态
/addsharing      - 在当前服务器共享你的密钥
/removesharing   - 停止在当前服务器共享
/deletekey       - 删除你注册的API密钥
```

## 🔍 模型参考

| 模型 | 描述 | 特点 |
|------|------|------|
| nai-diffusion-4-full | 最新完整模型 | 支持最新功能，但可能不稳定 |
| nai-diffusion-4-curated | V4精选版 | 经过筛选的训练数据 |
| nai-diffusion-3 | 稳定V3模型 | 支持SMEA，稳定可靠 |
| nai-diffusion-3-furry | furry特定模型 | 针对furry角色优化 |

> **提示:** 如果v4模型遇到问题，请尝试使用更稳定的nai-diffusion-3模型。

## ⚠️ 故障排除

- 确保你有有效的NovelAI订阅
- 检查API密钥是否正确
- 检查Discord Bot Token是否有效
- 使用`/checkapi`命令检查NovelAI API状态

## 📜 许可证

[MIT License](LICENSE)

## 免责声明

此项目不隶属于NovelAI。使用此机器人时，请确保遵守NovelAI的服务条款。用户需要拥有有效的NovelAI订阅并使用自己的API密钥。
