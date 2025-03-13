#!/bin/bash

<<<<<<< HEAD
echo "========== NovelAI Discord Bot 启动脚本 v2.1.0 =========="
=======
echo "========== NovelAI Discord Bot 启动脚本 =========="
>>>>>>> 1a714346f1f021a5f2d3a69b4f5bbc7eafac125d

# 首先更新pip本身
echo "正在更新pip..."
python3 -m pip install --upgrade pip

# 安装所有依赖
echo "正在安装依赖..."
python3 -m pip install -r requirements.txt

# 检查配置文件是否存在
if [ ! -f "config.txt" ]; then
  echo "警告: 未找到config.txt配置文件"
  
  if [ -f "config.example.txt" ]; then
    echo "提示: 你可以复制config.example.txt为config.txt并进行编辑"
    echo "是否要现在创建config.txt? (y/n)"
    read answer
    
    if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
      cp config.example.txt config.txt
      echo "已创建config.txt，请编辑并设置你的Discord token"
      echo "按任意键继续..."
      read -n 1
    fi
  else
    echo "注意: 请确保设置了DISCORD_TOKEN环境变量，或创建config.txt文件"
  fi
else
  echo "已找到config.txt配置文件"
fi

<<<<<<< HEAD
# 检查数据目录
DATA_DIR="data"
if [ ! -d "$DATA_DIR" ]; then
  echo "创建数据目录..."
  mkdir -p "$DATA_DIR"
fi

=======
>>>>>>> 1a714346f1f021a5f2d3a69b4f5bbc7eafac125d
# 检查API密钥文件权限
if [ -f "api_keys.json" ]; then
  echo "发现API密钥存储文件"
  # 设置文件权限为仅所有者可读写
  chmod 600 api_keys.json
  echo "已设置api_keys.json为仅所有者可读写(权限600)"
fi

# 显示已安装的包
echo "已安装的Python包:"
<<<<<<< HEAD
python3 -m pip list | grep -E 'discord|requests|aiohttp|gitpython'

echo "========== 正在启动机器人 =========="
# 运行机器人
python3 bot.py

# 检查退出代码
if [ $? -ne 0 ]; then
  echo "机器人异常退出，退出代码: $?"
  echo "查看上方日志以获取更多信息"
  echo "按任意键退出..."
  read -n 1
fi
=======
python3 -m pip list

echo "========== 正在启动机器人 =========="
# 运行机器人
python3 bot.py
>>>>>>> 1a714346f1f021a5f2d3a69b4f5bbc7eafac125d
