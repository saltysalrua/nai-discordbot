#!/bin/bash

# 确保使用Python 3
# 检查Python 3是否可用
if command -v python3 &> /dev/null
then
    echo "使用Python 3..."
    # 使用pip3安装依赖
    python3 -m pip install -r requirements.txt
    # 运行机器人
    python3 bot.py
else
    echo "Python 3不可用，尝试使用python命令..."
    # 检查python是否为Python 3
    PYTHON_VERSION=$(python --version 2>&1)
    if [[ $PYTHON_VERSION == *"Python 3"* ]]; then
        echo "Python命令是Python 3，使用中..."
        python -m pip install -r requirements.txt
        python bot.py
    else
        echo "错误: 需要Python 3才能运行此机器人。"
        echo "当前Python版本: $PYTHON_VERSION"
        exit 1
    fi
fi
