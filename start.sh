#!/bin/bash

# 首先更新pip本身
python3 -m pip install --upgrade pip

# 直接安装discord.py（指定版本）
python3 -m pip install -U discord.py==2.3.2

# 安装其他依赖
python3 -m pip install requests
python3 -m pip install flask

# 显示已安装的包
python3 -m pip list

# 运行机器人
python3 bot.py
