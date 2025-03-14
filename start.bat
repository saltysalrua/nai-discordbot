@echo off
setlocal enabledelayedexpansion

echo ========== NovelAI Discord Bot 启动脚本 v2.1.0 (Windows) ==========

REM 首先更新pip本身
echo 正在更新pip...
python -m pip install --upgrade pip

REM 安装所有依赖
echo 正在安装依赖...
python -m pip install -r requirements.txt

REM 检查配置文件是否存在
if not exist "config.txt" (
  echo 警告: 未找到config.txt配置文件
  
  if exist "config.example.txt" (
    echo 提示: 你可以复制config.example.txt为config.txt并进行编辑
    set /p answer=是否要现在创建config.txt? (y/n): 
    
    if /i "!answer!"=="y" (
      copy config.example.txt config.txt
      echo 已创建config.txt，请编辑并设置你的Discord token
      echo 按任意键继续...
      pause > nul
    )
  ) else (
    echo 注意: 请确保设置了DISCORD_TOKEN环境变量，或创建config.txt文件
  )
) else (
  echo 已找到config.txt配置文件
)

REM 检查数据目录
set DATA_DIR=data
if not exist "%DATA_DIR%" (
  echo 创建数据目录...
  mkdir "%DATA_DIR%"
)

REM 检查API密钥文件
if exist "api_keys.json" (
  echo 发现API密钥存储文件
  REM Windows没有chmod命令，但可以设置文件为只读提高安全性
  attrib +r api_keys.json
  echo 已设置api_keys.json为只读，以提高安全性
)

REM 显示已安装的包
echo 已安装的Python包:
python -m pip list | findstr "discord requests aiohttp gitpython"

echo ========== 正在启动机器人 ==========
REM 运行机器人
python bot.py

REM 检查退出代码
if %ERRORLEVEL% neq 0 (
  echo 机器人异常退出，退出代码: %ERRORLEVEL%
  echo 查看上方日志以获取更多信息
  echo 按任意键退出...
  pause > nul
)

endlocal
