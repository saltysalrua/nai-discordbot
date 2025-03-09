# NovelAI Discord Bot

一个功能强大的Discord机器人，用于使用NovelAI API生成AI艺术图像。

## 主要特性

- 🔑 **个人API密钥管理**：用户使用自己的NovelAI API密钥，支持密钥过期时间设置
- 🖼️ **快速图像生成**：简单指令 `/nai [提示词] [模型]` 快速生成图像
- ⚙️ **高级参数设置**：交互式UI支持完整的NovelAI参数配置
- 🔒 **共享设置**：用户可选择将API密钥限定为个人使用或在特定服务器共享
- 🧩 **多模型支持**：支持所有NovelAI模型，包括最新的nai-diffusion-4

## 安装步骤

1. **克隆仓库**

```bash
git clone https://github.com/saltysalrua/nai-discordbot.git
cd novelai-discord-bot
```

2. **安装依赖**

```bash
pip install -r requirements.txt
```

3. **配置环境变量**

创建一个 `.env` 文件，参考 `.env.example`：

```bash
cp .env.example .env
```

编辑 `.env` 文件，添加你的Discord机器人令牌：

```
DISCORD_TOKEN=你的Discord机器人令牌
```

4. **创建Discord机器人**

- 访问 [Discord Developer Portal](https://discord.com/developers/applications)
- 创建一个新应用
- 在"Bot"选项卡中添加一个机器人
- 启用所有Privileged Gateway Intents
- 在"OAuth2 > URL Generator"中，选择`bot`范围和以下权限：
  - `Send Messages`
  - `Embed Links`
  - `Attach Files`
  - `Use Slash Commands`
- 使用生成的URL将机器人邀请到你的Discord服务器

5. **运行机器人**

```bash
python bot.py
```

## 使用指南

### API密钥管理

```
/apikey [你的NovelAI API密钥] [私人使用/服务器共享] [有效期小时数]
```

例如：
- `/apikey pst-abcdefg12345 私人使用 24` - 注册24小时有效的私人密钥
- `/apikey pst-abcdefg12345 服务器共享 0` - 注册永不过期的服务器共享密钥

查看当前密钥状态：
```
/apikey
```

删除已注册的密钥：
```
/deletekey
```

### 图像生成

基础生成命令：
```
/nai [提示词] [模型(可选)]
```

例如：
- `/nai 漂亮的动漫女孩，长蓝发` - 使用默认模型生成
- `/nai 可爱的猫娘 nai-diffusion-4-curated` - 使用指定模型生成

### 高级设置

1. 生成图像后点击「高级设置」按钮
2. 填写参数表单(提示词、模型、尺寸、步数、负面提示词)
3. 提交生成

### 完整参数设置

1. 点击「完整参数」按钮
2. 使用下拉菜单选择模型、尺寸、采样器和噪声调度
3. 使用按钮设置步数、CFG比例、SMEA选项和负面提示词
4. 点击「生成图像」按钮

## 可用模型

- `nai-diffusion-4-full` - 最新完整模型
- `nai-diffusion-4-curated` - V4精选版
- `nai-diffusion-3` - 上一代模型
- `nai-diffusion-3-furry` - furry模型

## 配置项

默认参数可在代码开头部分修改：

```python
# 默认参数
DEFAULT_MODEL = "nai-diffusion-4-full"
DEFAULT_SIZE = (832, 1216)  # (width, height)
DEFAULT_STEPS = 28
DEFAULT_SCALE = 6.5
DEFAULT_SAMPLER = "k_euler_ancestral"
DEFAULT_NOISE_SCHEDULE = "native"
DEFAULT_CFG_RESCALE = 0.1
```

## 许可证

[MIT License](LICENSE)

## 免责声明

此项目不隶属于NovelAI。使用此机器人时，请确保遵守NovelAI的服务条款。用户需要拥有有效的NovelAI订阅并使用自己的API密钥。
