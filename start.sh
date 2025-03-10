#!/bin/bash

echo "========== NovelAI Discord Bot 启动脚本 =========="

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

# 显示已安装的包
echo "已安装的Python包:"
python3 -m pip list

echo "========== 正在启动机器人 =========="
# 运行机器人
python3 bot.py