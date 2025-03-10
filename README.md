# NovelAI Discord Bot

一个功能强大的Discord机器人，用于使用NovelAI API生成AI图片。

**所有代码都是Claude 3.7sonnet写的**

## 主要特性

- 🔑 **个人API密钥管理**：用户使用自己的NovelAI API密钥，支持密钥过期时间设置
- 🖼️ **快速图像生成**：简单指令 `/nai [提示词] [模型]` 快速生成图像
- ⚙️ **高级参数设置**：交互式UI支持完整的NovelAI参数配置
- 🔒 **共享设置**：用户可选择将API密钥限定为个人使用或在特定服务器共享
- 🧩 **多模型支持**：支持所有NovelAI模型，包括最新的nai-diffusion-4
- 🔧 **灵活配置**：支持通过配置文件或环境变量设置参数

## 安装步骤

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

有两种方式配置机器人：

### 方法1: 使用配置文件（推荐）

1. 复制示例配置文件：
```bash
cp config.example.txt config.txt
```

2. 编辑`config.txt`，填入你的Discord机器人令牌和其他配置：
```
# Discord机器人配置
DISCORD_TOKEN=你的discord机器人token
DEFAULT_MODEL=nai-diffusion-3
DEFAULT_SIZE_WIDTH=832
DEFAULT_SIZE_HEIGHT=1216
DEFAULT_STEPS=28
DEFAULT_SCALE=6.5
DEFAULT_SAMPLER=k_euler_ancestral
DEFAULT_NOISE_SCHEDULE=native
DEFAULT_CFG_RESCALE=0.1
```

### 方法2: 使用环境变量

设置环境变量`DISCORD_TOKEN`：

```bash
# Linux/macOS
export DISCORD_TOKEN=你的Discord机器人令牌

# Windows (CMD)
set DISCORD_TOKEN=你的Discord机器人令牌

# Windows (PowerShell)
$env:DISCORD_TOKEN="你的Discord机器人令牌"
```

4. **创建Discord机器人**

- 访问 [Discord Developer Portal](https://discord.com/developers/applications)
- 创建一个新应用
- 在"Bot"选项卡中添加一个机器
(还是自己去找教程吧)

5. **运行机器人**

```bash
# 直接启动
python bot.py

# 或使用启动脚本（会自动检查配置）
bash start.sh
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

共享密钥管理：
```
/addsharing      - 在当前服务器共享你的密钥
/removesharing   - 停止在当前服务器共享你的密钥
/sharedkeys      - 查看当前服务器中共享的API密钥信息
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

### 其他实用命令

```
/help       - 显示帮助信息
/checkapi   - 检查NovelAI API的可用性状态并提供故障排除建议
```

## 可用模型

- `nai-diffusion-4-full` - 最新完整模型（可能不稳定）
- `nai-diffusion-4-curated` - V4精选版（可能不稳定）
- `nai-diffusion-3` - 上一代模型（推荐，更稳定）
- `nai-diffusion-3-furry` - furry专用模型

## 模型特性和兼容性

不同模型支持不同的功能：

| 功能 | nai-diffusion-3 | nai-diffusion-3-furry | nai-diffusion-4-full | nai-diffusion-4-curated |
|------|----------------|----------------------|---------------------|------------------------|
| SMEA | ✅ 支持        | ✅ 支持              | ❌ 不支持           | ❌ 不支持              |
| 动态SMEA | ✅ 支持     | ✅ 支持              | ❌ 不支持           | ❌ 不支持              |
| Native噪声调度 | ✅ 支持 | ✅ 支持             | ❌ 不支持           | ❌ 不支持              |
| 特殊提示词格式 | ❌ 不需要 | ❌ 不需要           | ✅ 需要             | ✅ 需要               |

## 配置项

默认参数可在`config.txt`中修改：

```
# 默认参数
DEFAULT_MODEL=nai-diffusion-3
DEFAULT_SIZE_WIDTH=832
DEFAULT_SIZE_HEIGHT=1216
DEFAULT_STEPS=28
DEFAULT_SCALE=6.5
DEFAULT_SAMPLER=k_euler_ancestral
DEFAULT_NOISE_SCHEDULE=native
DEFAULT_CFG_RESCALE=0.1
DEFAULT_NEG_PROMPT=lowres, {bad}, error, fewer, extra, missing, worst quality, jpeg artifacts
```

## 故障排除

如果在使用v4模型时遇到问题：
1. 确保你的NovelAI订阅支持v4模型
2. 尝试使用更保守的参数设置
3. 如果依然遇到服务器错误(500)，请切换到更稳定的nai-diffusion-3模型

使用`/checkapi`命令可以检查API状态并获取更多故障排除建议。

## 许可证

[MIT License](LICENSE)

## 免责声明

此项目不隶属于NovelAI。使用此机器人时，请确保遵守NovelAI的服务条款。用户需要拥有有效的NovelAI订阅并使用自己的API密钥。
