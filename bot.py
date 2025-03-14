import os
import io
import json
import zipfile
import discord
from discord import app_commands, ui
import requests
import asyncio
import datetime
import traceback
import random
import time
import copy
import shutil
import sys
from typing import Dict, Optional, List, Union, Literal, Tuple, Any

# ===== 全局变量 =====
# API密钥和模板存储
api_keys = {}
prompt_templates = {}
# 使用跟踪
key_usage_counter = {}
key_last_used = {}
# 图像历史和生成队列
recent_generations = {}
generation_queues = {}
# 协作会话
relay_sessions = {}
# 用户批量任务状态
batch_tasks = {}

# 记录机器人启动时间和版本
BOT_START_TIME = datetime.datetime.now()
VERSION = "2.1.0"

# ===== 配置管理 =====
def read_config_file(file_path="config.txt"):
    """读取配置文件"""
    config = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
    except Exception as e:
        print(f"读取配置文件时出错: {str(e)}")
    return config

# 读取配置
config = read_config_file()

# 从配置文件加载设置
DISCORD_TOKEN = config.get('DISCORD_TOKEN') or os.getenv("DISCORD_TOKEN")
DEFAULT_MODEL = config.get('DEFAULT_MODEL', 'nai-diffusion-3')
DEFAULT_SIZE = (
    int(config.get('DEFAULT_SIZE_WIDTH', '832')), 
    int(config.get('DEFAULT_SIZE_HEIGHT', '1216'))
)
DEFAULT_STEPS = int(config.get('DEFAULT_STEPS', '28'))
DEFAULT_SCALE = float(config.get('DEFAULT_SCALE', '6.5'))
DEFAULT_SAMPLER = config.get('DEFAULT_SAMPLER', 'k_euler_ancestral')
DEFAULT_NOISE_SCHEDULE = config.get('DEFAULT_NOISE_SCHEDULE', 'native')
DEFAULT_CFG_RESCALE = float(config.get('DEFAULT_CFG_RESCALE', '0.1'))
DEFAULT_NEG_PROMPT = config.get('DEFAULT_NEG_PROMPT', 'lowres, {bad}, error, fewer, extra, missing, worst quality, jpeg artifacts, bad quality, watermark, unfinished, displeasing, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract], bad anatomy, bad hands')
BOT_ADMIN_IDS = config.get('BOT_ADMIN_IDS', "").split(",")
GITHUB_REPO = config.get('GITHUB_REPO', '')

# Discord机器人设置
intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# NovelAI API设置
NAI_API_URL = "https://image.novelai.net/ai/generate-image"

# 可用的选项
AVAILABLE_MODELS = [
    "nai-diffusion-4-full",
    "nai-diffusion-4-curated",
    "nai-diffusion-3",
    "nai-diffusion-3-furry"
]

MODEL_DESCRIPTIONS = {
    "nai-diffusion-4-full": "最新完整模型",
    "nai-diffusion-4-curated": "V4精选版",
    "nai-diffusion-3": "V3模型 (推荐，更稳定)",
    "nai-diffusion-3-furry": "毛绒特定模型"
}

AVAILABLE_SIZES = [
    "704x1472",
    "832x1216",
    "1024x1024",
    "1216x832",
    "1472x704"
]

AVAILABLE_SAMPLERS = [
    "k_euler",
    "k_euler_ancestral",
    "k_dpmpp_2s_ancestral",
    "k_dpmpp_2m_sde",
    "k_dpmpp_sde",
    "k_dpmpp_2m"
]

SAMPLER_DESCRIPTIONS = {
    "k_euler": "简单快速",
    "k_euler_ancestral": "默认推荐",
    "k_dpmpp_2s_ancestral": "高质量",
    "k_dpmpp_2m_sde": "细节丰富",
    "k_dpmpp_sde": "高级细节",
    "k_dpmpp_2m": "良好平衡"
}

AVAILABLE_NOISE_SCHEDULES = [
    "native",
    "karras",
    "exponential",
    "polyexponential",
]

AVAILABLE_NOISE_SCHEDULES_V4 = [
    "karras",
    "exponential",
    "polyexponential",
]

# ===== 文件存储功能 =====
def save_data_to_file(data, filename, key_field="expires_at"):
    """通用数据保存函数"""
    if not data:
        return
        
    # 处理日期字段序列化
    serializable_dict = {}
    for item_id, item_data in data.items():
        serializable_data = item_data.copy()
        if key_field in serializable_data and serializable_data[key_field]:
            if isinstance(serializable_data[key_field], datetime.datetime):
                serializable_data[key_field] = serializable_data[key_field].isoformat()
        serializable_dict[item_id] = serializable_data
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(serializable_dict, f, ensure_ascii=False, indent=2)
        print(f"已保存 {len(data)} 条数据到 {filename}")
    except Exception as e:
        print(f"保存数据到 {filename} 时出错: {str(e)}")

def load_data_from_file(filename, key_field="expires_at"):
    """通用数据加载函数"""
    if not os.path.exists(filename):
        print(f"未找到文件: {filename}")
        return {}
    
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 将字符串日期转换回datetime对象
        for item_id, item_data in data.items():
            if key_field in item_data and item_data[key_field]:
                item_data[key_field] = datetime.datetime.fromisoformat(item_data[key_field])
        
        print(f"已成功加载 {len(data)} 条数据从 {filename}")
        return data
    except Exception as e:
        print(f"加载数据从 {filename} 时出错: {str(e)}")
        return {}

def save_api_keys_to_file():
    """将标记为持久化的API密钥保存到文件"""
    # 只保存标记为持久化的密钥
    keys_to_save = {
        user_id: data.copy() 
        for user_id, data in api_keys.items() 
        if data.get("persist", False)
    }
    save_data_to_file(keys_to_save, "api_keys.json")

def load_api_keys_from_file():
    """从文件加载API密钥"""
    return load_data_from_file("api_keys.json")

def save_templates_to_file():
    """将提示词模板保存到文件"""
    save_data_to_file(prompt_templates, "prompt_templates.json", key_field="created_at")

def load_templates_from_file():
    """从文件加载提示词模板"""
    return load_data_from_file("prompt_templates.json", key_field="created_at")

# ===== API请求处理 =====
async def send_novelai_request(api_key, payload, interaction, retry_count=0):
    """使用改进的错误处理逻辑发送NovelAI API请求"""
    max_retries = 1
    
    # 验证API密钥格式
    if not api_key.startswith("pst-") or len(api_key) < 15:
        await interaction.followup.send("❌ API密钥格式无效。NovelAI的API密钥通常以'pst-'开头。", ephemeral=True)
        return None
    
    # 优化参数设置
    model = payload.get("model", "")
    input_prompt = payload.get("input", "")
    parameters = payload.get("parameters", {})
    optimized_parameters = parameters.copy()
    
    # v4模型特殊处理
    if model.startswith("nai-diffusion-4"):
        optimized_parameters["sm"] = False
        optimized_parameters["sm_dyn"] = False
        if optimized_parameters.get("noise_schedule") == "native":
            optimized_parameters["noise_schedule"] = "karras"
        # v4特定参数
        optimized_parameters["params_version"] = 3
        optimized_parameters["use_coords"] = True
        
        # v4格式化提示词和负面提示词
        negative_prompt = optimized_parameters.get("negative_prompt", "")
        
        # 构建v4提示词结构
        v4_prompt = {
            "caption": {
                "base_caption": input_prompt,
                "char_captions": []
            },
            "use_coords": True,
            "use_order": True
        }
        
        # 构建v4负面提示词结构
        v4_negative_prompt = {
            "caption": {
                "base_caption": negative_prompt,
                "char_captions": []
            },
            "use_coords": True,
            "use_order": True
        }
        
        # 将这些添加到参数中
        optimized_parameters["v4_prompt"] = v4_prompt
        optimized_parameters["v4_negative_prompt"] = v4_negative_prompt
        
        # v4不需要这些参数
        for key in ["dynamic_thresholding", "deliberate_euler_ancestral_bug", "prefer_brownian"]:
            if key in optimized_parameters:
                del optimized_parameters[key]
    
    # 确保宽高是64的倍数
    if "width" in optimized_parameters:
        optimized_parameters["width"] = (optimized_parameters["width"] // 64) * 64
    if "height" in optimized_parameters:
        optimized_parameters["height"] = (optimized_parameters["height"] // 64) * 64
    
    # 更新优化后的参数
    payload["parameters"] = optimized_parameters
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Referer": "https://novelai.net",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0"
    }
    
    try:
        response = await client.loop.run_in_executor(
            None, 
            lambda: requests.post(
                NAI_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
        )
        
        # 处理特定错误状态码
        if response.status_code == 402:
            await interaction.followup.send(
                "❌ 支付要求错误(402): 您的NovelAI订阅可能不支持此操作，或您的配额已用完。"
            )
            return None
            
        elif response.status_code == 401:
            await interaction.followup.send(
                "❌ 授权错误(401): API密钥无效或已过期。请使用`/apikey`命令重新设置有效的API密钥。"
            )
            return None
        
        elif response.status_code == 429:
            await interaction.followup.send(
                "❌ 请求频率限制(429): 您发送了太多请求。请等待一段时间后再试。"
            )
            return None
        
        elif response.status_code == 500:
            # 500错误可能是参数问题，尝试使用更简单的参数重试
            if retry_count < max_retries:
                await interaction.followup.send("⚠️ 服务器内部错误，正在使用更保守的参数重试...", ephemeral=True)
                
                # 构建更安全的参数配置
                safe_params = {
                    "width": 768,
                    "height": 1024,
                    "scale": 4.5,
                    "steps": 22,
                    "n_samples": 1,
                    "ucPreset": 0,
                    "qualityToggle": True,
                    "sampler": "k_euler",
                    "noise_schedule": "karras" if model.startswith("nai-diffusion-4") else "native",
                    "negative_prompt": parameters.get("negative_prompt", DEFAULT_NEG_PROMPT),
                    "sm": False if model.startswith("nai-diffusion-4") else True,
                    "sm_dyn": False if model.startswith("nai-diffusion-4") else True,
                    "cfg_rescale": 0,
                }
                
                # v4模型特定参数
                if model.startswith("nai-diffusion-4"):
                    safe_params["params_version"] = 3
                    safe_params["use_coords"] = True
                    
                    if parameters.get("legacy_uc") is not None:
                        safe_params["legacy_uc"] = parameters["legacy_uc"]

                    
                    # 添加v4提示词结构
                    safe_params["v4_prompt"] = {
                        "caption": {
                            "base_caption": input_prompt,
                            "char_captions": []
                        },
                        "use_coords": True,
                        "use_order": True
                    }
                    
                    safe_params["v4_negative_prompt"] = {
                        "caption": {
                            "base_caption": parameters.get("negative_prompt", DEFAULT_NEG_PROMPT),
                            "char_captions": []
                        },
                        "use_coords": True,
                        "use_order": True
                    }
                    
                    await interaction.followup.send(
                        "⚠️ v4模型可能目前遇到服务器问题，如果失败请尝试切换到nai-diffusion-3模型。",
                        ephemeral=True
                    )
                
                # 修改参数并重试
                payload["parameters"] = safe_params
                return await send_novelai_request(api_key, payload, interaction, retry_count + 1)
                
            # 重试失败，提供简洁错误信息
            error_message = "❌ NovelAI服务器内部错误(500)。请尝试切换模型或简化提示词。"
            await interaction.followup.send(error_message)
            return None
            
        elif response.status_code != 200:
            # 其他非200状态码
            await interaction.followup.send(f"❌ NovelAI API返回错误: 状态码 {response.status_code}")
            return None
        
        # 尝试解析ZIP文件
        try:
            # 保存响应内容
            response_content = response.content
            
            with zipfile.ZipFile(io.BytesIO(response_content)) as zip_file:
                zip_contents = zip_file.namelist()
                
                if "image_0.png" not in zip_contents:
                    await interaction.followup.send(f"❌ ZIP文件中找不到图像文件。")
                    return None
                    
                image_data = zip_file.read("image_0.png")
                
                # 添加生成历史记录
                user_id = str(interaction.user.id)
                if user_id not in recent_generations:
                    recent_generations[user_id] = []
                    
                # 创建生成记录
                generation_record = {
                    "timestamp": datetime.datetime.now(),
                    "payload": payload.copy(),  # 复制payload避免引用问题
                    "seed": optimized_parameters.get("seed", "随机")
                }
                
                # 限制每用户最多保存5条记录
                recent_generations[user_id].insert(0, generation_record)
                if len(recent_generations[user_id]) > 5:
                    recent_generations[user_id].pop()
                
                return image_data
                
        except zipfile.BadZipFile:
            # 如果ZIP解析失败，尝试直接将响应作为图像处理
            if len(response_content) > 8 and response_content[:8] == b'\x89PNG\r\n\x1a\n':
                return response_content
            
            # 如果不是PNG，再尝试看是否是JPEG
            if len(response_content) > 3 and response_content[:3] == b'\xff\xd8\xff':
                return response_content
            
            await interaction.followup.send("❌ 无法解析NovelAI API响应: 返回的既不是有效的ZIP文件也不是图像")
            return None
            
    except requests.exceptions.RequestException as e:
        # 网络请求异常
        await interaction.followup.send(f"❌ 连接NovelAI API失败: {str(e)}")
        return None
    except Exception as e:
        # 其他未预期的异常
        await interaction.followup.send(f"❌ 生成图像时出错: {str(e)}")
        return None

# 根据模型获取正确的默认参数
def get_model_default_params(model):
    """根据模型返回默认参数设置，增加对v4必要参数的支持"""
    # 基础默认参数
    params = {
        "width": DEFAULT_SIZE[0],
        "height": DEFAULT_SIZE[1],
        "scale": DEFAULT_SCALE,
        "sampler": DEFAULT_SAMPLER,
        "steps": DEFAULT_STEPS,
        "n_samples": 1,
        "ucPreset": 0,
        "qualityToggle": True,
        "negative_prompt": DEFAULT_NEG_PROMPT,
        "cfg_rescale": DEFAULT_CFG_RESCALE,
    }
    
    # 根据模型版本设置特定参数
    if model.startswith("nai-diffusion-4"):
        # v4模型参数
        params["sm"] = False  # 不支持SMEA
        params["sm_dyn"] = False  # 不支持动态SMEA
        params["noise_schedule"] = "karras"  # v4推荐使用karras
        
        # v4特定参数
        params["params_version"] = 3  # v4需要此参数
        params["use_coords"] = True  # 使用坐标系统
    else:
        # v3模型参数
        params["sm"] = True  # 支持SMEA
        params["sm_dyn"] = True  # 支持动态SMEA
        params["noise_schedule"] = DEFAULT_NOISE_SCHEDULE
    
    return params

# ===== 辅助功能 =====
# 获取当前服务器中的API密钥共享数量
def get_guild_shared_keys_info(guild_id):
    """获取当前服务器中的API密钥共享信息"""
    shared_keys = []
    
    for user_id, key_data in api_keys.items():
        if guild_id in key_data.get("shared_guilds", []):
            provider_name = key_data.get("provider_name", "未知用户")
            expires_at = key_data.get("expires_at")
            expiry_text = "永不过期" if expires_at is None else f"{expires_at.strftime('%Y-%m-%d %H:%M:%S')}"
            
            shared_keys.append({
                "user_id": user_id,
                "provider_name": provider_name,
                "expires_at": expiry_text
            })
    
    return shared_keys

# 智能选择共享密钥
async def select_optimal_key(shared_keys):
    """智能选择最佳API密钥"""
    now = datetime.datetime.now()
    
    # 如果只有一个共享密钥，直接使用
    if len(shared_keys) == 1:
        shared_user_id, key_data = shared_keys[0]
        
        # 更新使用记录
        if shared_user_id not in key_usage_counter:
            key_usage_counter[shared_user_id] = 0
        key_usage_counter[shared_user_id] += 1
        key_last_used[shared_user_id] = now
        
        return shared_keys[0]
    
    # 评分因素：使用频率、上次使用时间
    scored_keys = []
    for shared_user_id, key_data in shared_keys:
        # 使用次数评分
        usage_count = key_usage_counter.get(shared_user_id, 0)
        usage_score = max(0, 10 - min(usage_count, 10))  # 使用次数越少分数越高，最高10分
        
        # 时间评分 - 越久未使用分数越高
        last_used = key_last_used.get(shared_user_id, now - datetime.timedelta(days=1))
        time_diff = (now - last_used).total_seconds()
        time_score = min(10, time_diff / 60)  # 每分钟1分，最高10分
        
        # 综合评分
        total_score = usage_score + time_score
        scored_keys.append((shared_user_id, key_data, total_score))
    
    # 按评分排序，选择最高分
    scored_keys.sort(key=lambda x: x[2], reverse=True)
    selected = (scored_keys[0][0], scored_keys[0][1])
    
    # 更新使用记录
    shared_user_id = selected[0]
    if shared_user_id not in key_usage_counter:
        key_usage_counter[shared_user_id] = 0
    key_usage_counter[shared_user_id] += 1
    key_last_used[shared_user_id] = now
    
    return selected

# 辅助函数：获取API密钥
async def get_api_key(interaction: discord.Interaction) -> tuple[Optional[str], Optional[str]]:
    """获取用户的API密钥，或请求他们注册一个。返回 (api_key, provider_info)"""
    user_id = str(interaction.user.id)
    guild_id = interaction.guild_id
    
    # 检查用户是否已注册密钥
    if user_id in api_keys:
        user_key = api_keys[user_id]
        
        # 检查密钥是否已过期
        if "expires_at" in user_key and user_key["expires_at"] is not None and user_key["expires_at"] < datetime.datetime.now():
            await interaction.followup.send(
                "❌ 你的API密钥已过期，请使用 `/apikey` 命令重新注册。", 
                ephemeral=True
            )
            del api_keys[user_id]
            return None, None
        
        # 检查是否可以在此服务器使用
        if guild_id and not user_key.get("shared_guilds"):
            await interaction.followup.send(
                "⚠️ 你的API密钥设置为私人使用，但这是一个服务器频道。请使用 `/addsharing` 更新设置允许在此服务器共享使用。", 
                ephemeral=True
            )
            return None, None
            
        if guild_id and guild_id not in user_key.get("shared_guilds", []):
            await interaction.followup.send(
                "⚠️ 你的API密钥未设置为在此服务器共享。请使用 `/addsharing` 更新设置允许在此服务器共享使用。", 
                ephemeral=True
            )
            return None, None
            
        return user_key["key"], "自己的密钥"  # 使用自己的密钥
    
    # 用户没有注册密钥，查找服务器共享密钥
    if guild_id:
        shared_keys = []
        
        for shared_user_id, key_data in api_keys.items():
            if guild_id in key_data.get("shared_guilds", []):
                # 检查密钥是否过期
                if "expires_at" in key_data and key_data["expires_at"] is not None and key_data["expires_at"] < datetime.datetime.now():
                    continue
                
                shared_keys.append((shared_user_id, key_data))
        
        if shared_keys:
            # 选择最优的共享密钥
            selected_key = await select_optimal_key(shared_keys)
            if selected_key:
                shared_user_id, key_data = selected_key
                provider_name = key_data.get("provider_name", "未知用户")
                
                return key_data["key"], f"{provider_name} 共享的密钥"  # 使用共享密钥，显示提供者信息
    
    # 显示错误消息-没有可用密钥
    msg = "⚠️ 你需要先注册你的NovelAI API密钥才能使用此功能。"
    if guild_id:
        shared_keys_info = get_guild_shared_keys_info(guild_id)
        if shared_keys_info:
            msg += f"\n\n当前服务器有 {len(shared_keys_info)} 个共享的API密钥，但这些密钥可能已过期或不可用。"
        msg += "\n请使用 `/apikey [你的密钥]` 命令注册" + ("，或联系密钥提供者更新共享设置。" if shared_keys_info else "。")
    else:
        msg += "请使用 `/apikey [你的密钥]` 命令注册。"
    
    await interaction.followup.send(msg, ephemeral=True)
    return None, None

# ===== 后台任务 =====
# 定期保存任务
async def periodic_save_keys():
    """定期保存标记为持久化的API密钥和提示词模板"""
    while True:
        await asyncio.sleep(60 * 15)  # 每15分钟保存一次
        save_api_keys_to_file()
        save_templates_to_file()
        print(f"[{datetime.datetime.now()}] 已执行定期保存")

# 添加网络连接检查函数
async def check_internet_connection():
    """检查互联网连接，使用多个可靠站点进行测试"""
    test_sites = [
        "https://www.google.com",
        "https://www.cloudflare.com",
        "https://www.amazon.com"
    ]
    
    for site in test_sites:
        try:
            response = await client.loop.run_in_executor(
                None, 
                lambda: requests.get(site, timeout=5)
            )
            if response.status_code == 200:
                return True
        except:
            pass
    
    return False

# 改进 API 密钥有效性检查
async def check_api_key_validity(api_key, max_retries=2, retry_delay=3):
    """检查API密钥是否有效，带有重试机制"""
    test_payload = {
        "input": "test",
        "model": "nai-diffusion-3",
        "action": "generate",
        "parameters": {
            "width": 64,  # 使用最小尺寸
            "height": 64,
            "scale": 1.0,
            "sampler": "k_euler",
            "steps": 1,  # 使用最小步数减少服务器负担
            "n_samples": 1,
            "qualityToggle": False
        }
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Referer": "https://novelai.net",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0"
    }
    
    # 多次尝试
    for attempt in range(max_retries):
        try:
            response = await client.loop.run_in_executor(
                None, 
                lambda: requests.post(
                    NAI_API_URL,
                    headers=headers,
                    json=test_payload,
                    timeout=10
                )
            )
            
            # 检查响应状态码
            if response.status_code in (401, 402):
                # 这些是确定的无效密钥响应
                return False
                
            if response.status_code == 200:
                # 确定有效
                return True
                
            # 其他状态码可能是临时性问题，继续重试
                
        except requests.exceptions.RequestException:
            # 连接错误，可能是临时网络问题，等待后重试
            pass
            
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay)
    
    # 如果所有尝试都失败，检查互联网连接
    internet_available = await check_internet_connection()
    if not internet_available:
        print("⚠️ 互联网连接不可用，API密钥验证被跳过")
        return True  # 假定密钥有效，因为无法确定
        
    # 所有尝试都失败，但互联网可用，密钥可能确实无效
    return False

# 改进密钥验证定时任务
async def hourly_validate_keys():
    """每小时检查API密钥有效性，带有网络检查和智能重试"""
    while True:
        await asyncio.sleep(3600)  # 每小时检查一次
        print(f"[{datetime.datetime.now()}] 开始执行API密钥有效性检查...")
        
        # 首先检查互联网连接
        internet_available = await check_internet_connection()
        if not internet_available:
            print("⚠️ 互联网连接不可用，跳过API密钥验证")
            continue
        
        # 首先检查NovelAI站点是否可访问
        try:
            site_response = await client.loop.run_in_executor(
                None,
                lambda: requests.get("https://novelai.net/", timeout=10)
            )
            
            if site_response.status_code != 200:
                print(f"⚠️ NovelAI网站返回状态码 {site_response.status_code}，延迟密钥验证")
                continue
                
        except requests.exceptions.RequestException:
            print("⚠️ 无法连接到NovelAI网站，延迟密钥验证")
            continue
        
        invalid_keys = []
        checked_count = 0
        
        for user_id, key_data in list(api_keys.items()):
            # 先检查是否已过期
            if "expires_at" in key_data and key_data["expires_at"] and key_data["expires_at"] < datetime.datetime.now():
                invalid_keys.append(user_id)
                continue
            
            # 检查API密钥有效性 - 使用改进的检查函数
            is_valid = await check_api_key_validity(key_data["key"], max_retries=2)
            checked_count += 1
            
            if not is_valid:
                invalid_keys.append(user_id)
            
            # 每检查几个密钥暂停一下，避免过快请求
            if checked_count % 5 == 0:
                await asyncio.sleep(2)
        
        # 移除无效密钥
        for user_id in invalid_keys:
            if user_id in api_keys:  # 再次检查，因为可能在循环过程中被修改
                # 如果是持久化密钥，从文件中也删除
                was_persistent = api_keys[user_id].get("persist", False)
                del api_keys[user_id]
                
                if was_persistent:
                    try:
                        # 直接读取文件内容
                        if os.path.exists("api_keys.json"):
                            with open("api_keys.json", "r", encoding="utf-8") as f:
                                file_keys = json.load(f)
                            
                            # 如果用户 ID 在文件中存在，删除它
                            if user_id in file_keys:
                                del file_keys[user_id]
                            
                            # 写回文件
                            with open("api_keys.json", "w", encoding="utf-8") as f:
                                json.dump(file_keys, f, ensure_ascii=False, indent=2)
                                
                            print(f"已从 api_keys.json 文件中删除无效用户 {user_id} 的密钥")
                    except Exception as e:
                        print(f"从文件中删除无效密钥时出错: {str(e)}")
        
        print(f"[{datetime.datetime.now()}] API密钥检查完成，检查了 {checked_count} 个密钥，移除了 {len(invalid_keys)} 个无效密钥")

# 定期检查过期密钥
async def check_expired_keys():
    """定期检查并清理过期的API密钥"""
    while True:
        await asyncio.sleep(60 * 5)  # 每5分钟检查一次
        
        # 获取当前时间
        now = datetime.datetime.now()
        
        # 找出过期的密钥
        expired_keys = [
            user_id for user_id, data in api_keys.items()
            if "expires_at" in data and data["expires_at"] is not None and data["expires_at"] < now
        ]
        
        # 删除过期的密钥
        for user_id in expired_keys:
            del api_keys[user_id]
            
        if expired_keys:
            print(f"已清理 {len(expired_keys)} 个过期的API密钥")
            
            # 如果有持久化的密钥过期，更新存储
            if any(user_id in api_keys and api_keys[user_id].get("persist", False) for user_id in expired_keys):
                save_api_keys_to_file()

# 队列处理器
async def queue_processor():
    """持续处理所有队列中的请求"""
    while True:
        processed = False
        
        # 处理所有活跃队列
        for queue_id, queue_data in list(generation_queues.items()):
            if queue_data["queue"] and not queue_data["processing"]:
                # 标记为处理中
                queue_data["processing"] = True
                
                try:
                    # 处理队列头部的请求
                    request = queue_data["queue"][0]
                    await process_queued_request(request)
                    processed = True
                except Exception as e:
                    print(f"处理队列请求时出错: {str(e)}")
                    try:
                        interaction = request.get("interaction")
                        await interaction.followup.send(f"❌ 队列处理失败: {str(e)}", ephemeral=True)
                    except:
                        pass
                finally:
                    # 移除已处理的请求
                    queue_data["queue"].pop(0)
                    queue_data["processing"] = False
                    queue_data["last_processed"] = datetime.datetime.now()
                
                # 避免过快处理所有请求
                break  
        
        # 调整等待时间，避免无限循环消耗资源
        if not processed:
            await asyncio.sleep(1)
        else:
            await asyncio.sleep(3)  # 请求间隔，避免API限制

async def process_queued_request(request):
    """处理队列中的单个请求"""
    interaction = request.get("interaction")
    api_key = request.get("api_key") 
    payload = request.get("payload")
    provider_info = request.get("provider_info")
    is_batch = request.get("is_batch", False)
    batch_index = request.get("batch_index", 0)
    batch_total = request.get("batch_total", 1)
    
    # 复用现有的API请求处理函数
    image_data = await send_novelai_request(api_key, payload, interaction)
    if image_data is None:
        raise Exception("图像生成失败")
    
    # 创建文件并发送 - 复用现有模式
    file = discord.File(io.BytesIO(image_data), filename=f"queued_image_{int(time.time())}.png")
    
    title = "批量生成" if is_batch else "队列生成"
    if is_batch:
        title += f" ({batch_index+1}/{batch_total})"
        
    embed = discord.Embed(title=title, color=0xf75c7e)
    embed.add_field(name="提示词", value=payload.get("input", "")[:1024], inline=False)
    embed.add_field(name="模型", value=payload.get("model", DEFAULT_MODEL), inline=True)
    
    if provider_info:
        embed.add_field(name="🔑 API密钥", value=provider_info, inline=True)
        
    embed.set_image(url=f"attachment://queued_image_{int(time.time())}.png")
    embed.set_footer(text=f"由 {interaction.user.display_name} 生成")
    
    await interaction.followup.send(file=file, embed=embed)

# 协作会话清理
async def cleanup_expired_sessions():
    """定期清理过期的协作会话"""
    while True:
        await asyncio.sleep(60)  # 每分钟检查一次
        
        now = datetime.datetime.now()
        
        # 清理过期的接力会话
        expired_relays = []
        for session_id, session in relay_sessions.items():
            if session["expires_at"] < now:
                expired_relays.append(session_id)
                
                # 发送过期通知
                try:
                    channel = client.get_channel(int(session["channel_id"]))
                    if channel:
                        await channel.send(f"⏰ 接力生成会话 `{session_id}` 已过期。")
                except:
                    pass
        
        # 删除过期会话
        for session_id in expired_relays:
            del relay_sessions[session_id]
            
        if expired_relays:
            print(f"已清理 {len(expired_relays)} 个过期的接力会话")

# 创建接力生成的按钮视图
class RelayButtons(discord.ui.View):
    def __init__(self, session_id, expires_at):
        # 计算超时时间
        timeout = (expires_at - datetime.datetime.now()).total_seconds()
        super().__init__(timeout=timeout)
        self.session_id = session_id
        
    @discord.ui.button(label="添加内容", style=discord.ButtonStyle.primary, emoji="➕")
    async def add_content_button(self, interaction, button):
        # 使用全局处理函数
        await handle_relay_add_content(interaction, self.session_id)
        
    @discord.ui.button(label="完成接力", style=discord.ButtonStyle.success, emoji="✅")
    async def complete_relay_button(self, interaction, button):
        # 使用全局处理函数
        await handle_relay_complete(interaction, self.session_id)

# ===== 批量任务管理 =====
async def process_batch_task(task_id, user_id):
    """处理批量任务队列"""
    if task_id not in batch_tasks or user_id not in batch_tasks[task_id]:
        return
        
    task = batch_tasks[task_id][user_id]
    
    # 如果任务已被取消，不处理
    if task["status"] == "cancelled":
        return
        
    # 更新状态为处理中
    task["status"] = "processing"
    task["current"] = 0
    task["total"] = len(task["requests"])
    
    # 处理所有请求
    for i, request in enumerate(task["requests"]):
        # 如果任务被取消，提前退出
        if task_id not in batch_tasks or user_id not in batch_tasks[task_id] or batch_tasks[task_id][user_id]["status"] == "cancelled":
            break
            
        # 更新当前进度
        task["current"] = i + 1
        
        try:
            # 处理当前请求
            await process_queued_request(request)
            
            # 添加延迟以避免过快发送
            await asyncio.sleep(3)
        except Exception as e:
            print(f"批量任务 {task_id} 处理请求 {i+1}/{len(task['requests'])} 时出错: {str(e)}")
            
            # 尝试发送错误通知
            try:
                interaction = request.get("interaction")
                await interaction.followup.send(f"❌ 批量生成 {i+1}/{len(task['requests'])} 失败: {str(e)}", ephemeral=True)
            except:
                pass
    
    # 完成任务，更新状态
    if task_id in batch_tasks and user_id in batch_tasks[task_id]:
        batch_tasks[task_id][user_id]["status"] = "completed"
        batch_tasks[task_id][user_id]["completed_at"] = datetime.datetime.now()
        
        # 发送完成通知
        interaction = task["requests"][0].get("interaction")
        if interaction:
            await interaction.followup.send(f"✅ 批量任务 `{task_id}` 已完成，成功生成 {task['current']}/{task['total']} 张图像。", ephemeral=True)

# ===== 机器人初始化 =====
@client.event
async def on_ready():
    print(f'机器人已登录为 {client.user}')
    
    await tree.sync()  # 同步斜杠命令
    
    # 从文件加载API密钥
    global api_keys, prompt_templates
    loaded_keys = load_api_keys_from_file()
    if loaded_keys:
        api_keys.update(loaded_keys)
        print(f"已从文件加载 {len(loaded_keys)} 个API密钥")
        
    # 加载提示词模板
    loaded_templates = load_templates_from_file()
    if loaded_templates:
        prompt_templates = loaded_templates
        print(f"已加载 {len(loaded_templates)} 个提示词模板")
    
    # 初始化队列系统
    global generation_queues
    generation_queues = {}
    client.loop.create_task(queue_processor())
    print("队列系统已初始化")
    
    # 初始化协作系统
    global relay_sessions
    relay_sessions = {}
    client.loop.create_task(cleanup_expired_sessions())
    print("协作生成系统已初始化")
    
    # 初始化批量任务系统
    global batch_tasks
    batch_tasks = {}
    print("批量任务系统已初始化")
    
    # 启动各种后台任务
    client.loop.create_task(check_expired_keys())  # 密钥过期检查
    client.loop.create_task(periodic_save_keys())  # 定期保存
    client.loop.create_task(hourly_validate_keys())  # 密钥验证
    
    print(f"机器人 v{VERSION} 已完全初始化并准备就绪")

# ===== API密钥管理命令 =====
@tree.command(name="apikey", description="注册或管理你的NovelAI API密钥")
@app_commands.describe(
    key="你的NovelAI API密钥",
    sharing="设置密钥是否在此服务器共享",
    duration_hours="密钥有效时间(小时), 0表示永不过期",
    persist="是否在机器人重启后保存密钥（会进行存储）"
)
async def apikey_command(
    interaction: discord.Interaction, 
    key: str = None,
    sharing: Literal["私人使用", "服务器共享"] = "私人使用",
    duration_hours: int = 24,
    persist: Literal["是", "否"] = "否"
):
    user_id = str(interaction.user.id)
    
    # 检查是否是查看密钥信息请求
    if key is None:
        if user_id in api_keys:
            user_key = api_keys[user_id]
            
            # 检查密钥是否已过期
            if "expires_at" in user_key and user_key["expires_at"] is not None and user_key["expires_at"] < datetime.datetime.now():
                await interaction.response.send_message("你的API密钥已过期，请重新注册。", ephemeral=True)
                del api_keys[user_id]
                return
            
            # 构建密钥信息
            expiry = "永不过期" if "expires_at" not in user_key or user_key["expires_at"] is None else f"{user_key['expires_at'].strftime('%Y-%m-%d %H:%M:%S')}"
            
            # 查看共享信息
            if not user_key.get("shared_guilds"):
                sharing_info = "私人使用"
            else:
                sharing_info = f"共享的服务器: {len(user_key['shared_guilds'])}个"
                if interaction.guild_id and interaction.guild_id in user_key.get("shared_guilds", []):
                    sharing_info += " (包括当前服务器)"
            
            # 检查是否是持久化存储
            persist_info = "是" if user_key.get("persist", False) else "否"
            
            await interaction.response.send_message(
                f"你已注册API密钥:\n"
                f"• 密钥状态: 有效\n"
                f"• 共享设置: {sharing_info}\n"
                f"• 过期时间: {expiry}\n"
                f"• 持久化存储: {persist_info}", 
                ephemeral=True
            )
        else:
            # 如果在服务器中，显示当前服务器的共享密钥信息
            if interaction.guild_id:
                shared_keys = get_guild_shared_keys_info(interaction.guild_id)
                shared_info = f"当前服务器有 {len(shared_keys)} 个共享的API密钥。"
                if shared_keys:
                    providers = [key_info["provider_name"] for key_info in shared_keys]
                    shared_info += f" 提供者: {', '.join(providers)}"
                
                await interaction.response.send_message(
                    f"你还没有注册API密钥。请使用 `/apikey [你的密钥] [共享设置] [有效时间]` 来注册。\n\n{shared_info}",
                    ephemeral=True
                )
            else:
                await interaction.response.send_message(
                    "你还没有注册API密钥。请使用 `/apikey [你的密钥] [共享设置] [有效时间]` 来注册。",
                    ephemeral=True
                )
        return
    
    # 验证API密钥格式
    if not key.startswith("pst-") or len(key) < 15:
        await interaction.response.send_message(
            "❌ API密钥格式无效。NovelAI的API密钥应以'pst-'开头并包含足够长度。",
            ephemeral=True
        )
        return
        
    # 为用户注册新密钥
    guild_id = interaction.guild_id if interaction.guild_id and sharing == "服务器共享" else None
    
    # 设置过期时间
    expires_at = None
    if duration_hours > 0:
        expires_at = datetime.datetime.now() + datetime.timedelta(hours=duration_hours)
    
    # 保存密钥信息
    api_keys[user_id] = {
        "key": key,
        "shared_guilds": [guild_id] if guild_id else [],
        "expires_at": expires_at,
        "provider_name": interaction.user.display_name,  # 记录提供者名称
        "persist": persist == "是"  # 添加是否持久化的标志
    }
    
    # 构建确认信息
    expiry_text = "永不过期" if expires_at is None else f"{duration_hours}小时后过期 ({expires_at.strftime('%Y-%m-%d %H:%M:%S')})"
    sharing_text = "仅限你个人使用" if not guild_id else f"在此服务器共享使用"
    
    # 如果用户选择了持久化存储
    if persist == "是":
        # 告知用户关于存储的信息
        storage_info = (
            "⚠️ **关于密钥存储的重要信息**\n"
            "• 你的API密钥将被存储在机器人所在的服务器上\n"
            "• 这样在机器人重启后你的密钥设置仍然有效\n"
            "• 你可以随时使用`/deletekey`命令删除你的密钥\n"
            "• 密钥仍会按照设定的有效期自动失效"
        )
        
        # 保存密钥数据
        save_api_keys_to_file()
        
        await interaction.response.send_message(
            f"✅ API密钥已成功注册！\n"
            f"• 密钥: ||{key[:5]}...{key[-4:]}||\n"
            f"• 共享设置: {sharing_text}\n"
            f"• 有效期: {expiry_text}\n"
            f"• 持久存储: 已启用\n\n{storage_info}",
            ephemeral=True
        )
    else:
        # 如果用户选择不持久化，则使用原来的消息格式
        await interaction.response.send_message(
            f"✅ API密钥已成功注册！\n"
            f"• 密钥: ||{key[:5]}...{key[-4:]}||\n"
            f"• 共享设置: {sharing_text}\n"
            f"• 有效期: {expiry_text}\n"
            f"• 持久存储: 未启用（机器人重启后将失效）",
            ephemeral=True
        )

@tree.command(name="deletekey", description="删除你注册的NovelAI API密钥")
async def deletekey_command(interaction: discord.Interaction):
    user_id = str(interaction.user.id)
    
    if user_id in api_keys:
        was_persistent = api_keys[user_id].get("persist", False)
        del api_keys[user_id]
        
        # 如果是持久化密钥，需要从文件中读取所有密钥，删除此用户的密钥，然后重新写入
        if was_persistent:
            try:
                # 直接读取文件内容而不是通过 load_api_keys_from_file 函数
                if os.path.exists("api_keys.json"):
                    with open("api_keys.json", "r", encoding="utf-8") as f:
                        file_keys = json.load(f)
                    
                    # 如果用户 ID 在文件中存在，删除它
                    if user_id in file_keys:
                        del file_keys[user_id]
                    
                    # 写回文件
                    with open("api_keys.json", "w", encoding="utf-8") as f:
                        json.dump(file_keys, f, ensure_ascii=False, indent=2)
                    
                    print(f"已从 api_keys.json 文件中删除用户 {user_id} 的密钥")
            except Exception as e:
                print(f"从文件中删除密钥时出错: {str(e)}")
                # 即使出错，我们也继续响应给用户
        
        await interaction.response.send_message(
            "✅ 你的API密钥已从机器人中删除。" + 
            ("所有持久化存储的数据也已清除。" if was_persistent else ""), 
            ephemeral=True
        )
    else:
        await interaction.response.send_message("你没有注册API密钥。", ephemeral=True)

@tree.command(name="addsharing", description="将你的API密钥添加到当前服务器共享列表")
async def addsharing_command(interaction: discord.Interaction):
    user_id = str(interaction.user.id)
    guild_id = interaction.guild_id
    
    if not guild_id:
        await interaction.response.send_message("此命令只能在服务器中使用。", ephemeral=True)
        return
    
    if user_id not in api_keys:
        await interaction.response.send_message("你没有注册API密钥。请先使用 `/apikey` 命令注册。", ephemeral=True)
        return
    
    user_key = api_keys[user_id]
    
    # 检查密钥是否已过期
    if "expires_at" in user_key and user_key["expires_at"] is not None and user_key["expires_at"] < datetime.datetime.now():
        await interaction.response.send_message("你的API密钥已过期，请重新注册。", ephemeral=True)
        del api_keys[user_id]
        return
    
    # 如果服务器已在共享列表中
    if guild_id in user_key.get("shared_guilds", []):
        await interaction.response.send_message("你的API密钥已在此服务器共享。", ephemeral=True)
        return
    
    # 添加服务器到共享列表
    if "shared_guilds" not in user_key:
        user_key["shared_guilds"] = []
    
    user_key["shared_guilds"].append(guild_id)
    
    # 如果是持久化存储的密钥，保存更新
    if user_key.get("persist", False):
        save_api_keys_to_file()
        
    await interaction.response.send_message("✅ 你的API密钥现在已在此服务器共享。", ephemeral=True)

@tree.command(name="removesharing", description="从当前服务器共享列表中移除你的API密钥")
async def removesharing_command(interaction: discord.Interaction):
    user_id = str(interaction.user.id)
    guild_id = interaction.guild_id
    
    if not guild_id:
        await interaction.response.send_message("此命令只能在服务器中使用。", ephemeral=True)
        return
    
    if user_id not in api_keys:
        await interaction.response.send_message("你没有注册API密钥。", ephemeral=True)
        return
    
    user_key = api_keys[user_id]
    
    # 如果服务器不在共享列表中
    if guild_id not in user_key.get("shared_guilds", []):
        await interaction.response.send_message("你的API密钥未在此服务器共享。", ephemeral=True)
        return
    
    # 从共享列表中移除服务器
    user_key["shared_guilds"].remove(guild_id)
    
    # 如果是持久化存储的密钥，保存更新
    if user_key.get("persist", False):
        save_api_keys_to_file()
        
    await interaction.response.send_message("✅ 你的API密钥已从此服务器共享列表中移除。", ephemeral=True)

@tree.command(name="sharedkeys", description="显示当前服务器中共享的API密钥信息")
async def sharedkeys_command(interaction: discord.Interaction):
    if not interaction.guild_id:
        await interaction.response.send_message("此命令只能在服务器中使用。", ephemeral=True)
        return
    
    shared_keys = get_guild_shared_keys_info(interaction.guild_id)
    
    if not shared_keys:
        await interaction.response.send_message("当前服务器没有共享的API密钥。", ephemeral=True)
        return
    
    embed = discord.Embed(
        title=f"服务器共享API密钥 ({len(shared_keys)}个)",
        description="以下用户提供了API密钥在此服务器共享使用：",
        color=0xf75c7e
    )
    
    for i, key_info in enumerate(shared_keys, 1):
        embed.add_field(
            name=f"密钥 #{i}",
            value=f"提供者: {key_info['provider_name']}\n过期时间: {key_info['expires_at']}",
            inline=True
        )
    
    await interaction.response.send_message(embed=embed, ephemeral=True)

# ===== 提示词模板管理命令 =====
@tree.command(name="savetemplate", description="保存当前提示词为模板")
@app_commands.describe(
    name="模板名称",
    prompt="提示词内容",
    sharing="设置模板是否在此服务器共享",
    tags="标签，用逗号分隔（例如: 风景,动漫）",
    save_params="是否保存高级参数设置"
)
async def savetemplate_command(
    interaction: discord.Interaction, 
    name: str, 
    prompt: str, 
    sharing: Literal["私人使用", "服务器共享"] = "私人使用",
    tags: str = "",
    save_params: bool = False
):
    user_id = str(interaction.user.id)
    template_id = f"{user_id}_{int(time.time())}"
    guild_id = interaction.guild_id if interaction.guild_id and sharing == "服务器共享" else None
    
    # 保存模板信息
    template_data = {
        "name": name,
        "prompt": prompt,
        "creator_id": user_id,
        "creator_name": interaction.user.display_name,
        "shared_guilds": [guild_id] if guild_id else [],
        "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
        "created_at": datetime.datetime.now()
    }
    
    # 如果选择保存高级参数
    if save_params:
        # 获取用户最近的生成记录来提取参数
        has_recent_params = False
        if user_id in recent_generations and recent_generations[user_id]:
            latest_record = recent_generations[user_id][0]
            if "payload" in latest_record:
                params = latest_record["payload"].get("parameters", {})
                if params:
                    model = latest_record["payload"].get("model", DEFAULT_MODEL)
                    template_data["model"] = model
                    template_data["params"] = {
                        "width": params.get("width", DEFAULT_SIZE[0]),
                        "height": params.get("height", DEFAULT_SIZE[1]),
                        "scale": params.get("scale", DEFAULT_SCALE),
                        "sampler": params.get("sampler", DEFAULT_SAMPLER),
                        "steps": params.get("steps", DEFAULT_STEPS),
                        "noise_schedule": params.get("noise_schedule", DEFAULT_NOISE_SCHEDULE),
                        "cfg_rescale": params.get("cfg_rescale", DEFAULT_CFG_RESCALE),
                        "sm": params.get("sm", True),
                        "sm_dyn": params.get("sm_dyn", True),
                        "negative_prompt": params.get("negative_prompt", DEFAULT_NEG_PROMPT)
                    }
                    has_recent_params = True
        
        if not has_recent_params:
            # 使用默认参数
            template_data["model"] = DEFAULT_MODEL
            template_data["params"] = {
                "width": DEFAULT_SIZE[0],
                "height": DEFAULT_SIZE[1],
                "scale": DEFAULT_SCALE,
                "sampler": DEFAULT_SAMPLER,
                "steps": DEFAULT_STEPS,
                "noise_schedule": DEFAULT_NOISE_SCHEDULE,
                "cfg_rescale": DEFAULT_CFG_RESCALE,
                "sm": True,
                "sm_dyn": True,
                "negative_prompt": DEFAULT_NEG_PROMPT
            }
    
    prompt_templates[template_id] = template_data
    
    # 保存模板
    save_templates_to_file()
    
    # 构建确认信息
    sharing_text = "仅限你个人使用" if not guild_id else f"在此服务器共享使用"
    tags_text = tags if tags else "无"
    
    # 添加参数信息
    params_text = "已保存（包含当前生成设置）" if save_params else "未保存（仅保存提示词）"
    
    await interaction.response.send_message(
        f"✅ 提示词模板 \"{name}\" 已保存！\n"
        f"• 提示词: {prompt[:50]}{'...' if len(prompt) > 50 else ''}\n"
        f"• 共享设置: {sharing_text}\n"
        f"• 标签: {tags_text}\n"
        f"• 高级参数: {params_text}\n"
        f"• 模板ID: {template_id}\n\n"
        f"使用 `/usetemplate {template_id}` 来基于此模板生成图像，\n"
        f"或在其他生成命令中使用 `template_id={template_id}` 参数引用此模板。",
        ephemeral=True
    )

@tree.command(name="listtemplates", description="查看可用的提示词模板")
@app_commands.describe(
    filter_tags="按标签筛选（用逗号分隔）",
    show_all="是否显示所有共享模板"
)
async def listtemplates_command(
    interaction: discord.Interaction, 
    filter_tags: str = "",
    show_all: bool = False
):
    user_id = str(interaction.user.id)
    guild_id = interaction.guild_id
    
    # 处理筛选标签
    tags_filter = [tag.strip().lower() for tag in filter_tags.split(",") if tag.strip()]
    
    # 收集符合条件的模板
    available_templates = []
    
    for template_id, template in prompt_templates.items():
        # 判断用户是否有权访问此模板
        is_creator = template.get("creator_id") == user_id
        is_guild_shared = guild_id in template.get("shared_guilds", [])
        
        if is_creator or is_guild_shared or show_all:
            # 如果有标签筛选，则检查标签
            if tags_filter:
                template_tags = [tag.lower() for tag in template.get("tags", [])]
                if not any(tag in template_tags for tag in tags_filter):
                    continue
            
            # 收集模板信息
            template_info = {
                "id": template_id,
                "name": template.get("name", "未命名模板"),
                "creator": template.get("creator_name", "未知创建者"),
                "tags": template.get("tags", []),
                "has_params": "params" in template,
                "is_creator": is_creator,
                "is_shared": is_guild_shared
            }
            available_templates.append(template_info)
    
    if not available_templates:
        await interaction.response.send_message(
            f"没有找到符合条件的提示词模板。" +
            (f"尝试使用不同的标签筛选或选择「显示所有模板」。" if tags_filter else "尝试使用 `/savetemplate` 创建新模板。"),
            ephemeral=True
        )
        return
    
    # 创建嵌入消息
    embed = discord.Embed(
        title=f"提示词模板 ({len(available_templates)}个)",
        description=f"以下是你可以访问的提示词模板：" +
                   (f"\n筛选标签: {filter_tags}" if filter_tags else ""),
        color=0x3498db
    )
    
    # 最多显示20个模板
    if len(available_templates) > 20:
        embed.set_footer(text=f"共找到 {len(available_templates)} 个模板，仅显示前20个")
        available_templates = available_templates[:20]
    
    # 添加每个模板的信息
    for i, template in enumerate(available_templates, 1):
        tags_display = ", ".join(template["tags"]) if template["tags"] else "无标签"
        source_display = "✓ 你创建的" if template["is_creator"] else "👥 服务器共享" if template["is_shared"] else "🌐 全局共享"
        params_display = "🔧 包含参数设置" if template["has_params"] else "📝 仅含提示词"
        
        embed.add_field(
            name=f"{i}. {template['name']}",
            value=f"ID: `{template['id']}`\n创建者: {template['creator']}\n标签: {tags_display}\n{source_display}\n{params_display}",
            inline=i % 2 == 1  # 交替布局
        )
    
    # 显示用法信息
    embed.add_field(
        name="使用方法",
        value=(
            "• 单独使用: `/usetemplate [模板ID]`\n"
            "• 与高级生成结合: `/naigen template_id=[模板ID] [其他参数]`\n"
            "• 与批量生成结合: `/naibatch template_id=[模板ID] [变量定义]`"
        ),
        inline=False
    )
    
    await interaction.response.send_message(embed=embed, ephemeral=True)

@tree.command(name="usetemplate", description="使用提示词模板生成图像")
@app_commands.describe(
    template_id="模板ID（从 /listtemplates 获取）",
    model="选择模型（可覆盖模板中的模型设置）",
    override_prompt="额外添加到提示词的内容（可选）",
    use_params="是否使用模板中保存的参数设置"
)
@app_commands.choices(
    model=[
        app_commands.Choice(name=f"{model} - {MODEL_DESCRIPTIONS[model]}", value=model)
        for model in AVAILABLE_MODELS
    ]
)
async def usetemplate_command(
    interaction: discord.Interaction, 
    template_id: str,
    model: str = None,
    override_prompt: str = "",
    use_params: bool = True
):
    await interaction.response.defer(thinking=True)
    
    # 获取API密钥
    api_key, provider_info = await get_api_key(interaction)
    if not api_key:
        return
    
    # 查找模板
    if template_id not in prompt_templates:
        await interaction.followup.send("❌ 未找到指定的模板。请使用 `/listtemplates` 查看可用模板。", ephemeral=True)
        return
    
    template = prompt_templates[template_id]
    user_id = str(interaction.user.id)
    guild_id = interaction.guild_id
    
    # 检查访问权限
    is_creator = template.get("creator_id") == user_id
    is_guild_shared = guild_id in template.get("shared_guilds", [])
    
    if not (is_creator or is_guild_shared):
        await interaction.followup.send("❌ 你没有权限使用此模板。它可能是私人模板或未在此服务器共享。", ephemeral=True)
        return
    
    # 获取模板提示词
    prompt = template.get("prompt", "")
    if not prompt:
        await interaction.followup.send("❌ 此模板不包含有效的提示词。", ephemeral=True)
        return
    
    # 添加额外提示词
    if override_prompt:
        prompt = f"{prompt}, {override_prompt}"
    
    # 准备参数
    selected_model = model if model else template.get("model", DEFAULT_MODEL)
    
    # 获取参数 - 根据用户选择使用模板参数或默认参数
    model_params = None
    if use_params and "params" in template:
        model_params = template["params"].copy()
        # 确保参数兼容选中的模型
        if model and model.startswith("nai-diffusion-4") and not selected_model.startswith("nai-diffusion-4"):
            # 调整参数以适应v4模型
            model_params["sm"] = False
            model_params["sm_dyn"] = False
            if model_params.get("noise_schedule") == "native":
                model_params["noise_schedule"] = "karras"
        elif model and not model.startswith("nai-diffusion-4") and selected_model.startswith("nai-diffusion-4"):
            # 调整参数以适应v3模型
            model_params["sm"] = True
            model_params["sm_dyn"] = True
    else:
        model_params = get_model_default_params(selected_model)
    
    # 准备API请求
    payload = {
        "input": prompt,
        "model": selected_model,
        "action": "generate",
        "parameters": model_params
    }
    
    # 使用统一的API请求处理函数
    image_data = await send_novelai_request(api_key, payload, interaction)
    if image_data is None:
        return  # 如果API请求失败，直接返回
    
    # 创建文件对象并发送
    file = discord.File(io.BytesIO(image_data), filename="template_generated.png")
    
    # 创建嵌入消息
    embed = discord.Embed(title=f"模板生成: {template.get('name')}", color=0x3498db)
    embed.add_field(name="提示词", value=prompt[:1024], inline=False)
    embed.add_field(name="模型", value=selected_model, inline=True)
    
    # 显示关键参数
    param_text = []
    if model_params:
        param_text.append(f"尺寸: {model_params.get('width', DEFAULT_SIZE[0])}x{model_params.get('height', DEFAULT_SIZE[1])}")
        param_text.append(f"采样器: {model_params.get('sampler', DEFAULT_SAMPLER)}")
    embed.add_field(name="参数", value="\n".join(param_text) if param_text else "使用默认参数", inline=True)
    
    embed.add_field(name="模板创建者", value=template.get("creator_name", "未知"), inline=True)
    
    if template.get("tags"):
        embed.add_field(name="标签", value=", ".join(template.get("tags")), inline=True)
    
    # 显示API密钥提供者信息
    if provider_info:
        embed.add_field(name="🔑 API密钥", value=provider_info, inline=True)
        
    embed.set_image(url="attachment://template_generated.png")
    embed.set_footer(text=f"由 {interaction.user.display_name} 使用模板生成")
    
    await interaction.followup.send(file=file, embed=embed)

@tree.command(name="deletetemplate", description="删除你创建的提示词模板")
@app_commands.describe(
    template_id="要删除的模板ID"
)
async def deletetemplate_command(interaction: discord.Interaction, template_id: str):
    user_id = str(interaction.user.id)
    
    # 检查模板是否存在
    if template_id not in prompt_templates:
        await interaction.response.send_message("❌ 未找到指定的模板。", ephemeral=True)
        return
    
    # 检查是否是创建者
    template = prompt_templates[template_id]
    if template.get("creator_id") != user_id:
        await interaction.response.send_message("❌ 你不是此模板的创建者，无法删除。", ephemeral=True)
        return
    
    # 删除模板
    template_name = template.get("name", "未命名模板")
    del prompt_templates[template_id]
    
    # 保存更新
    save_templates_to_file()
    
    await interaction.response.send_message(f"✅ 已删除模板 \"{template_name}\"。", ephemeral=True)

@tree.command(name="updatetemplate", description="更新现有模板的参数")
@app_commands.describe(
    template_id="要更新的模板ID",
    new_name="新的模板名称（可选）",
    new_prompt="新的提示词（可选）",
    new_tags="新的标签（用逗号分隔）（可选）",
    update_params="是否更新为最近一次生成的参数"
)
async def updatetemplate_command(
    interaction: discord.Interaction, 
    template_id: str,
    new_name: str = None,
    new_prompt: str = None,
    new_tags: str = None,
    update_params: bool = False
):
    user_id = str(interaction.user.id)
    
    # 检查模板是否存在
    if template_id not in prompt_templates:
        await interaction.response.send_message("❌ 未找到指定的模板。", ephemeral=True)
        return
    
    # 检查是否是创建者
    template = prompt_templates[template_id]
    if template.get("creator_id") != user_id:
        await interaction.response.send_message("❌ 你不是此模板的创建者，无法更新。", ephemeral=True)
        return
    
    # 更新模板
    if new_name:
        template["name"] = new_name
    
    if new_prompt:
        template["prompt"] = new_prompt
    
    if new_tags:
        template["tags"] = [tag.strip() for tag in new_tags.split(",") if tag.strip()]
    
    # 更新参数
    if update_params:
        # 检查是否有最近的生成记录
        if user_id in recent_generations and recent_generations[user_id]:
            latest_record = recent_generations[user_id][0]
            if "payload" in latest_record:
                model = latest_record["payload"].get("model", DEFAULT_MODEL)
                params = latest_record["payload"].get("parameters", {})
                
                template["model"] = model
                template["params"] = {
                    "width": params.get("width", DEFAULT_SIZE[0]),
                    "height": params.get("height", DEFAULT_SIZE[1]),
                    "scale": params.get("scale", DEFAULT_SCALE),
                    "sampler": params.get("sampler", DEFAULT_SAMPLER),
                    "steps": params.get("steps", DEFAULT_STEPS),
                    "noise_schedule": params.get("noise_schedule", DEFAULT_NOISE_SCHEDULE),
                    "cfg_rescale": params.get("cfg_rescale", DEFAULT_CFG_RESCALE),
                    "sm": params.get("sm", True),
                    "sm_dyn": params.get("sm_dyn", True),
                    "negative_prompt": params.get("negative_prompt", DEFAULT_NEG_PROMPT)
                }
        else:
            await interaction.response.send_message("⚠️ 没有找到最近的生成记录，参数未更新。", ephemeral=True)
            return
    
    # 保存更新
    save_templates_to_file()
    
    # 构建更新摘要
    update_summary = []
    if new_name:
        update_summary.append(f"• 名称: {new_name}")
    if new_prompt:
        update_summary.append(f"• 提示词: {new_prompt[:50]}..." if len(new_prompt) > 50 else f"• 提示词: {new_prompt}")
    if new_tags:
        update_summary.append(f"• 标签: {new_tags}")
    if update_params:
        update_summary.append("• 参数: 已更新为最近一次生成的参数")
    
    await interaction.response.send_message(
        f"✅ 模板 \"{template['name']}\" 已更新！\n\n" + "\n".join(update_summary),
        ephemeral=True
    )

# ===== 图像生成命令 =====
@tree.command(name="nai", description="使用NovelAI生成图像")
@app_commands.describe(
    prompt="图像生成提示词",
    model="模型选择",
    template_id="要使用的模板ID（可选）"
)
@app_commands.choices(
    model=[
        app_commands.Choice(name=f"{model} - {MODEL_DESCRIPTIONS[model]}", value=model)
        for model in AVAILABLE_MODELS
    ]
)
async def nai_command(
    interaction: discord.Interaction, 
    prompt: str = None,
    model: str = None,
    template_id: str = None
):
    await interaction.response.defer(thinking=True)
    
    try:
        # 获取API密钥
        api_key, provider_info = await get_api_key(interaction)
        if not api_key:
            return
        
        # 处理模板
        if template_id:
            if template_id not in prompt_templates:
                await interaction.followup.send("❌ 未找到指定的模板。请使用 `/listtemplates` 查看可用模板。", ephemeral=True)
                return
                
            template = prompt_templates[template_id]
            user_id = str(interaction.user.id)
            guild_id = interaction.guild_id
            
            # 检查访问权限
            is_creator = template.get("creator_id") == user_id
            is_guild_shared = guild_id in template.get("shared_guilds", [])
            
            if not (is_creator or is_guild_shared):
                await interaction.followup.send("❌ 你没有权限使用此模板。", ephemeral=True)
                return
                
            # 如果未提供提示词，使用模板提示词
            if not prompt:
                prompt = template.get("prompt", "")
            else:
                # 如果提供了提示词，与模板提示词组合
                base_prompt = template.get("prompt", "")
                prompt = f"{base_prompt}, {prompt}"
                
            # 如果未指定模型，使用模板模型
            if not model and "model" in template:
                model = template["model"]
        
        # 确保有提示词
        if not prompt:
            await interaction.followup.send("❌ 必须提供提示词或有效的模板。", ephemeral=True)
            return
        
        # 验证并设置模型
        selected_model = model if model in AVAILABLE_MODELS else DEFAULT_MODEL
        
        # 获取适合模型的参数
        model_params = None
        if template_id and template_id in prompt_templates and "params" in prompt_templates[template_id]:
            # 使用模板参数
            model_params = prompt_templates[template_id]["params"].copy()
            
            # 调整参数以适应选中的模型
            if model and selected_model.startswith("nai-diffusion-4"):
                model_params["sm"] = False
                model_params["sm_dyn"] = False
                if model_params.get("noise_schedule") == "native":
                    model_params["noise_schedule"] = "karras"
        else:
            # 使用默认参数
            model_params = get_model_default_params(selected_model)
        
        # 准备API请求
        payload = {
            "input": prompt,
            "model": selected_model,
            "action": "generate",
            "parameters": model_params
        }
        
        # 使用统一的API请求处理函数
        image_data = await send_novelai_request(api_key, payload, interaction)
        if image_data is None:
            return  # 如果API请求失败，直接返回
        
        # 创建文件对象并发送
        file = discord.File(io.BytesIO(image_data), filename="generated_image.png")
        
        # 创建基本嵌入消息
        embed = discord.Embed(title="NovelAI 生成图像", color=0xf75c7e)
        embed.add_field(name="提示词", value=prompt[:1024], inline=False)
        embed.add_field(name="模型", value=selected_model, inline=True)
        
        # 如果使用模板，显示模板信息
        if template_id and template_id in prompt_templates:
            template_name = prompt_templates[template_id].get("name", "未命名模板")
            embed.add_field(name="使用模板", value=template_name, inline=True)
        
        # 显示参数
        param_text = []
        if model_params:
            param_text.append(f"尺寸: {model_params.get('width', DEFAULT_SIZE[0])}x{model_params.get('height', DEFAULT_SIZE[1])}")
            param_text.append(f"采样器: {model_params.get('sampler', DEFAULT_SAMPLER)}")
        if param_text:
            embed.add_field(name="参数", value="\n".join(param_text), inline=True)
        
        # 如果使用的是共享密钥，显示提供者信息
        if provider_info:
            if provider_info == "自己的密钥":
                embed.add_field(name="🔑 API密钥", value="使用自己的密钥", inline=True)
            else:
                embed.add_field(name="🔑 API密钥", value=provider_info, inline=True)
            
        embed.set_image(url="attachment://generated_image.png")
        embed.set_footer(text=f"由 {interaction.user.display_name} 生成")
        
        await interaction.followup.send(file=file, embed=embed)
        
    except Exception as e:
        print(f"生成图像时出错: {str(e)}")
        print(traceback.format_exc())
        await interaction.followup.send(f"❌ 生成图像时出错: {str(e)}")

@tree.command(name="naigen", description="使用NovelAI生成图像 (高级选项)")
@app_commands.describe(
    prompt="图像生成提示词",
    model="选择模型",
    size="图像尺寸 (宽x高)",
    steps="采样步数 (1-28)",
    scale="CFG比例 (1-10)",
    sampler="采样器",
    noise_schedule="噪声调度",
    negative_prompt="负面提示词",
    smea="启用SMEA (仅v3模型)",
    dynamic_smea="启用动态SMEA (仅v3模型)",
    cfg_rescale="CFG重缩放 (0-1)",
    seed="随机种子 (留空为随机)",
    variety_plus="启用Variety+功能",
    legacy_uc="启用legacy_uc功能 (仅v4模型)",
    template_id="要使用的模板ID (可选，可与其他参数结合)"
)
@app_commands.choices(
    model=[
        app_commands.Choice(name=f"{model} - {MODEL_DESCRIPTIONS[model]}", value=model)
        for model in AVAILABLE_MODELS
    ],
    size=[
        app_commands.Choice(name=size, value=size)
        for size in AVAILABLE_SIZES
    ],
    sampler=[
        app_commands.Choice(name=f"{sampler} - {SAMPLER_DESCRIPTIONS[sampler]}", value=sampler)
        for sampler in AVAILABLE_SAMPLERS
    ],
    noise_schedule=[
        app_commands.Choice(name=schedule, value=schedule)
        for schedule in AVAILABLE_NOISE_SCHEDULES
    ]
)
async def naigen_command(
    interaction: discord.Interaction, 
    prompt: str = None,
    model: str = None,
    size: str = None,
    steps: int = None,
    scale: float = None,
    sampler: str = None,
    noise_schedule: str = None,
    negative_prompt: str = None,
    smea: bool = None,
    dynamic_smea: bool = None,
    cfg_rescale: float = None,
    seed: str = None,
    variety_plus: bool = None,
    legacy_uc: bool = None,
    template_id: str = None
):
    await interaction.response.defer(thinking=True)
    
    try:
        # 获取API密钥
        api_key, provider_info = await get_api_key(interaction)
        if not api_key:
            return
        
        # 处理模板
        template_params = {}
        template_model = None
        template_prompt = None
        
        if template_id:
            if template_id not in prompt_templates:
                await interaction.followup.send("❌ 未找到指定的模板。请使用 `/listtemplates` 查看可用模板。", ephemeral=True)
                return
                
            template = prompt_templates[template_id]
            user_id = str(interaction.user.id)
            guild_id = interaction.guild_id
            
            # 检查访问权限
            is_creator = template.get("creator_id") == user_id
            is_guild_shared = guild_id in template.get("shared_guilds", [])
            
            if not (is_creator or is_guild_shared):
                await interaction.followup.send("❌ 你没有权限使用此模板。", ephemeral=True)
                return
                
            # 获取模板参数
            if "params" in template:
                template_params = template["params"]
            
            # 获取模板模型
            if "model" in template:
                template_model = template["model"]
                
            # 获取模板提示词
            template_prompt = template.get("prompt", "")
            
            # 如果未提供提示词，使用模板提示词
            if not prompt:
                prompt = template_prompt
            else:
                # 如果提供了提示词，与模板提示词组合
                prompt = f"{template_prompt}, {prompt}"
        
        # 确保有提示词
        if not prompt:
            await interaction.followup.send("❌ 必须提供提示词或有效的模板。", ephemeral=True)
            return
        
        # 用用户提供的参数覆盖模板参数
        # 选择模型的优先级：用户指定 > 模板指定 > 默认
        selected_model = model if model else template_model if template_model else DEFAULT_MODEL
        
        # 处理尺寸
        width, height = DEFAULT_SIZE
        if size:
            try:
                width, height = map(int, size.split('x'))
            except:
                pass
        elif "width" in template_params and "height" in template_params:
            width = template_params["width"]
            height = template_params["height"]
        
        # 确保步数在合理范围内 - 限制最大28步
        if steps is not None:
            steps = max(1, min(28, steps))
        elif "steps" in template_params:
            steps = template_params["steps"]
        else:
            steps = DEFAULT_STEPS
        
        # 确保CFG比例在合理范围内
        if scale is not None:
            scale = max(1.0, min(10.0, scale))
        elif "scale" in template_params:
            scale = template_params["scale"]
        else:
            scale = DEFAULT_SCALE
        
        # 确保CFG重缩放在合理范围内
        if cfg_rescale is not None:
            cfg_rescale = max(0.0, min(1.0, cfg_rescale))
        elif "cfg_rescale" in template_params:
            cfg_rescale = template_params["cfg_rescale"]
        else:
            cfg_rescale = DEFAULT_CFG_RESCALE
        
        # 处理采样器
        if not sampler:
            sampler = template_params.get("sampler", DEFAULT_SAMPLER)
        
        # 处理噪声调度，为v4模型自动调整
        if not noise_schedule:
            if "noise_schedule" in template_params:
                noise_schedule = template_params["noise_schedule"]
            else:
                noise_schedule = "karras" if selected_model.startswith("nai-diffusion-4") else DEFAULT_NOISE_SCHEDULE
        elif noise_schedule == "native" and selected_model.startswith("nai-diffusion-4"):
            noise_schedule = "karras"  # v4不支持native，自动切换为karras
        
        # 处理负面提示词
        if not negative_prompt:
            negative_prompt = template_params.get("negative_prompt", DEFAULT_NEG_PROMPT)
        
        # 处理SMEA设置
        if smea is None:
            if selected_model.startswith("nai-diffusion-4"):
                smea = False
            else:
                smea = template_params.get("sm", True)
                
        if dynamic_smea is None:
            if selected_model.startswith("nai-diffusion-4"):
                dynamic_smea = False
            else:
                dynamic_smea = template_params.get("sm_dyn", True)
        
        # 处理随机种子
        random_seed = True
        seed_value = 0
        if seed:
            try:
                seed_value = int(seed)
                random_seed = False
            except:
                pass
        
        # 处理Variety+参数，计算跳过CFG阀值
        skip_cfg_above_sigma = None
        if variety_plus:
            # 根据图像大小计算合适的阀值
            w = width / 8
            h = height / 8
            v = pow(4.0 * w * h / 63232, 0.5)
            skip_cfg_above_sigma = 19.0 * v
        
        # 构建参数
        model_params = {
            "width": width,
            "height": height,
            "scale": scale,
            "sampler": sampler,
            "steps": steps,
            "n_samples": 1,
            "ucPreset": 0,
            "qualityToggle": True,
            "negative_prompt": negative_prompt,
            "cfg_rescale": cfg_rescale,
            "noise_schedule": noise_schedule,
            "sm": smea,
            "sm_dyn": dynamic_smea,
            "seed": random.randint(0, 2**32-1) if random_seed else seed_value
        }
        
        # 添加Variety+相关参数
        if variety_plus and skip_cfg_above_sigma is not None:
            model_params["skip_cfg_above_sigma"] = skip_cfg_above_sigma
        
        # 添加v4特定参数
        if selected_model.startswith("nai-diffusion-4"):
            model_params["params_version"] = 3
            model_params["use_coords"] = True
            
            if legacy_uc:
               model_params["legacy_uc"] = True
        
        # 准备API请求
        payload = {
            "input": prompt,
            "model": selected_model,
            "action": "generate",
            "parameters": model_params
        }
        
        # 使用统一的API请求处理函数
        image_data = await send_novelai_request(api_key, payload, interaction)
        if image_data is None:
            return  # 如果API请求失败，直接返回
        
        # 创建文件对象并发送
        file = discord.File(io.BytesIO(image_data), filename="generated_image.png")
        
        # 创建嵌入消息
        embed = discord.Embed(title="NovelAI 高级生成", color=0xf75c7e)
        embed.add_field(name="提示词", value=prompt[:1024], inline=False)
        embed.add_field(name="模型", value=selected_model, inline=True)
        embed.add_field(name="尺寸", value=f"{width}x{height}", inline=True)
        
        # 显示种子值和Variety+状态
        seed_display = seed_value if not random_seed else "随机"
        embed.add_field(name="种子", value=f"{seed_display}", inline=True)
        
        if variety_plus:
            embed.add_field(name="Variety+", value="已启用", inline=True)
        
        if legacy_uc and selected_model.startswith("nai-diffusion-4"):
            embed.add_field(name="Legacy UC", value="已启用", inline=True)

        # 如果使用模板，显示模板信息
        if template_id and template_id in prompt_templates:
            template_name = prompt_templates[template_id].get("name", "未命名模板")
            embed.add_field(name="使用模板", value=template_name, inline=True)
        
        # 如果使用的是共享密钥，显示提供者信息
        if provider_info:
            if provider_info == "自己的密钥":
                embed.add_field(name="🔑 API密钥", value="使用自己的密钥", inline=True)
            else:
                embed.add_field(name="🔑 API密钥", value=provider_info, inline=True)
            
        embed.set_image(url="attachment://generated_image.png")
        embed.set_footer(text=f"由 {interaction.user.display_name} 生成")
        
        await interaction.followup.send(file=file, embed=embed)
        
    except Exception as e:
        print(f"高级生成出错: {str(e)}")
        print(traceback.format_exc())
        await interaction.followup.send(f"❌ 生成图像时出错: {str(e)}")

@tree.command(name="naivariation", description="基于最近生成的图像创建变体")
@app_commands.describe(
    index="要变化的图像索引(1为最近生成的)",
    variation_type="变化类型",
    additional_prompt="额外提示词(仅提示词增强模式使用)"
)
async def naivariation_command(
    interaction: discord.Interaction, 
    index: int = 1,
    variation_type: Literal["轻微调整", "提示词增强"] = "轻微调整",
    additional_prompt: str = ""
):
    await interaction.response.defer(thinking=True)
    
    user_id = str(interaction.user.id)
    if user_id not in recent_generations or not recent_generations[user_id]:
        await interaction.followup.send("❌ 没有找到最近的生成记录!", ephemeral=True)
        return
        
    if index < 1 or index > len(recent_generations[user_id]):
        await interaction.followup.send(f"❌ 索引超出范围，你只有 {len(recent_generations[user_id])} 条生成记录", ephemeral=True)
        return
    
    # 复制原始生成参数
    original_record = recent_generations[user_id][index-1]
    new_payload = copy.deepcopy(original_record["payload"])
    
    if variation_type == "轻微调整":
        # 微调参数但保持原始种子
        params = new_payload["parameters"]
        params["scale"] = min(10, params.get("scale", DEFAULT_SCALE) * random.uniform(0.9, 1.1))
        params["steps"] = min(28, params.get("steps", DEFAULT_STEPS) + random.randint(-2, 2))
    else:  # 提示词增强
        if not additional_prompt:
            await interaction.followup.send("❌ 提示词增强模式需要提供额外提示词", ephemeral=True)
            return
            
        # 添加新提示词内容
        original_prompt = new_payload.get("input", "")
        new_payload["input"] = f"{original_prompt}, {additional_prompt}"
        
        # 对v4模型更新提示词结构
        if "parameters" in new_payload and "v4_prompt" in new_payload["parameters"]:
            v4_prompt = new_payload["parameters"]["v4_prompt"]
            if "caption" in v4_prompt:
                v4_prompt["caption"]["base_caption"] = f"{original_prompt}, {additional_prompt}"
    
    # 复用现有的API请求代码
    api_key, provider_info = await get_api_key(interaction)
    if not api_key:
        return
        
    image_data = await send_novelai_request(api_key, new_payload, interaction)
    if image_data is None:
        return
    
    # 创建文件和嵌入消息
    file = discord.File(io.BytesIO(image_data), filename="variation.png")
    
    embed = discord.Embed(title=f"图像变体 - {variation_type}", color=0xf75c7e)
    embed.add_field(name="原始提示词", value=original_record["payload"].get("input", "")[:1024], inline=False)
    
    if variation_type == "提示词增强":
        embed.add_field(name="添加的内容", value=additional_prompt, inline=False)
        
    embed.add_field(name="模型", value=new_payload.get("model", DEFAULT_MODEL), inline=True)
    embed.add_field(name="种子", value=str(original_record["seed"]), inline=True)
    
    if provider_info:
        embed.add_field(name="🔑 API密钥", value=provider_info, inline=True)
        
    embed.set_image(url="attachment://variation.png")
    embed.set_footer(text=f"由 {interaction.user.display_name} 生成 | 变体")
    
    await interaction.followup.send(file=file, embed=embed)

# ===== 批量生成命令 =====
@tree.command(name="naibatch", description="提交批量图像生成请求")
@app_commands.describe(
    prompt="图像提示词模板，使用 {var1} {var2} 语法表示变量",
    variations="变量值列表，格式: var1=值1,值2,值3|var2=值4,值5,值6",
    param_variations="参数变化，格式: model=模型1,模型2|size=832x1216,1024x1024",
    model="默认使用的模型（如不在param_variations中指定）",
    template_id="要作为基础的模板ID（可选）"
)
@app_commands.choices(
    model=[
        app_commands.Choice(name=f"{model} - {MODEL_DESCRIPTIONS[model]}", value=model)
        for model in AVAILABLE_MODELS
    ]
)
async def naibatch_command(
    interaction: discord.Interaction, 
    prompt: str = None,
    variations: str = "",
    param_variations: str = "",
    model: str = None,
    template_id: str = None
):
    # 复用API密钥获取和参数验证逻辑
    await interaction.response.defer(thinking=True)
    
    api_key, provider_info = await get_api_key(interaction)
    if not api_key:
        return
    
    # 处理模板
    template_params = {}
    template_model = None
    template_prompt = None
    
    if template_id:
        if template_id not in prompt_templates:
            await interaction.followup.send("❌ 未找到指定的模板。请使用 `/listtemplates` 查看可用模板。", ephemeral=True)
            return
            
        template = prompt_templates[template_id]
        user_id = str(interaction.user.id)
        guild_id = interaction.guild_id
        
        # 检查访问权限
        is_creator = template.get("creator_id") == user_id
        is_guild_shared = guild_id in template.get("shared_guilds", [])
        
        if not (is_creator or is_guild_shared):
            await interaction.followup.send("❌ 你没有权限使用此模板。", ephemeral=True)
            return
            
        # 获取模板参数
        if "params" in template:
            template_params = template["params"]
        
        # 获取模板模型
        if "model" in template:
            template_model = template["model"]
            
        # 获取模板提示词
        template_prompt = template.get("prompt", "")
        
        # 如果未提供提示词，使用模板提示词
        if not prompt:
            prompt = template_prompt
        elif template_prompt:
            # 如果提供了提示词，与模板提示词组合
            prompt = f"{template_prompt}, {prompt}"
    
    # 确保有提示词
    if not prompt:
        await interaction.followup.send("❌ 必须提供提示词或有效的模板。", ephemeral=True)
        return
    
    # 选择模型的优先级：用户指定 > 模板指定 > 默认
    selected_model = model if model else template_model if template_model else DEFAULT_MODEL
        
    try:
        # 解析变量定义
        var_definitions = {}
        for part in variations.split('|'):
            if '=' not in part:
                continue
                
            var_name, var_values = part.split('=', 1)
            var_name = var_name.strip()
            var_values = [v.strip() for v in var_values.split(',')]
            var_definitions[var_name] = var_values
        
        # 解析参数变化
        param_var_definitions = {}
        if param_variations:
            for part in param_variations.split('|'):
                if '=' not in part:
                    continue
                    
                param_name, param_values = part.split('=', 1)
                param_name = param_name.strip().lower()
                param_values = [v.strip() for v in param_values.split(',')]
                param_var_definitions[param_name] = param_values
        
        # 生成所有可能的组合
        import itertools
        
        # 提示词变量组合
        prompt_vars_to_combine = []
        prompt_var_names = []
        
        for var_name, values in var_definitions.items():
            prompt_vars_to_combine.append(values)
            prompt_var_names.append(var_name)
            
        prompt_combinations = list(itertools.product(*prompt_vars_to_combine)) if prompt_vars_to_combine else [tuple()]
        
        # 参数变量组合
        param_vars_to_combine = []
        param_var_names = []
        
        for param_name, values in param_var_definitions.items():
            param_vars_to_combine.append(values)
            param_var_names.append(param_name)
            
        param_combinations = list(itertools.product(*param_vars_to_combine)) if param_vars_to_combine else [tuple()]
        
        # 计算总组合数
        total_combinations = len(prompt_combinations) * len(param_combinations)
        
        if total_combinations > 20:
            await interaction.followup.send(f"⚠️ 你定义了 {total_combinations} 个组合，超过最大限制(20个)。只处理前20个。", ephemeral=True)
            # 限制组合数，优先保持提示词变量的多样性
            if len(prompt_combinations) > 20:
                prompt_combinations = prompt_combinations[:20]
                total_combinations = len(prompt_combinations)
            else:
                max_param_combinations = 20 // len(prompt_combinations)
                param_combinations = param_combinations[:max_param_combinations]
                total_combinations = len(prompt_combinations) * len(param_combinations)
        
        # 创建批量任务ID
        task_id = f"batch_{int(time.time())}"
        user_id = str(interaction.user.id)
        
        if task_id not in batch_tasks:
            batch_tasks[task_id] = {}
            
        # 准备批处理队列
        batch_requests = []
        
        # 生成所有组合的请求
        for prompt_idx, prompt_combo in enumerate(prompt_combinations):
            # 创建当前组合的提示词
            current_prompt = prompt
            for j, var_name in enumerate(prompt_var_names):
                if j < len(prompt_combo):  # 确保索引有效
                    current_prompt = current_prompt.replace(f"{{{var_name}}}", prompt_combo[j])
            
            for param_idx, param_combo in enumerate(param_combinations):
                # 基础参数 - 使用模板参数或默认参数
                base_params = template_params.copy() if template_params else get_model_default_params(selected_model)
                
                # 应用参数变化
                current_model = selected_model
                for k, param_name in enumerate(param_var_names):
                    if k < len(param_combo):  # 确保索引有效
                        param_value = param_combo[k]
                        
                        # 特殊处理尺寸参数
                        if param_name == "size" and "x" in param_value:
                            try:
                                width, height = map(int, param_value.split("x"))
                                base_params["width"] = width
                                base_params["height"] = height
                            except:
                                pass
                        # 特殊处理模型参数
                        elif param_name == "model":
                            if param_value in AVAILABLE_MODELS:
                                current_model = param_value
                                # 调整参数以适应特定模型
                                if current_model.startswith("nai-diffusion-4"):
                                    base_params["sm"] = False
                                    base_params["sm_dyn"] = False
                                    if base_params.get("noise_schedule") == "native":
                                        base_params["noise_schedule"] = "karras"
                                    # 添加v4特定参数
                                    base_params["params_version"] = 3
                                    base_params["use_coords"] = True
                        # 特殊处理步数参数
                        elif param_name == "steps":
                            try:
                                steps = int(param_value)
                                base_params["steps"] = max(1, min(28, steps))
                            except:
                                pass
                        # 特殊处理缩放参数
                        elif param_name == "scale":
                            try:
                                scale = float(param_value)
                                base_params["scale"] = max(1.0, min(10.0, scale))
                            except:
                                pass
                        # 特殊处理采样器参数
                        elif param_name == "sampler" and param_value in AVAILABLE_SAMPLERS:
                            base_params["sampler"] = param_value
                        # 特殊处理噪声调度参数
                        elif param_name == "noise_schedule" and param_value in AVAILABLE_NOISE_SCHEDULES:
                            if param_value != "native" or not current_model.startswith("nai-diffusion-4"):
                                base_params["noise_schedule"] = param_value
                            else:
                                base_params["noise_schedule"] = "karras"  # v4不支持native
                
                # 准备API请求
                payload = {
                    "input": current_prompt,
                    "model": current_model,
                    "action": "generate",
                    "parameters": base_params
                }
                
                # 创建批处理请求
                batch_request = {
                    "interaction": interaction,
                    "api_key": api_key,
                    "payload": payload,
                    "provider_info": provider_info,
                    "is_batch": True,
                    "batch_index": len(batch_requests),
                    "batch_total": total_combinations
                }
                
                batch_requests.append(batch_request)
        
        # 保存批量任务信息
        batch_tasks[task_id][user_id] = {
            "requests": batch_requests,
            "created_at": datetime.datetime.now(),
            "status": "pending",
            "current": 0,
            "total": len(batch_requests)
        }
        
        # 启动处理任务
        client.loop.create_task(process_batch_task(task_id, user_id))
        
        await interaction.followup.send(
            f"✅ 已创建批量生成任务 `{task_id}`\n"
            f"• 提示词模板: {prompt}\n"
            f"• 提示词变量组合数: {len(prompt_combinations)}\n"
            f"• 参数变量组合数: {len(param_combinations)}\n"
            f"• 总生成图像数: {total_combinations}\n"
            f"• 状态: 队列处理中\n\n"
            f"使用 `/batchstatus {task_id}` 查看任务进度。",
            ephemeral=True
        )
        
    except Exception as e:
        print(f"批量生成时出错: {str(e)}")
        print(traceback.format_exc())
        await interaction.followup.send(f"❌ 批量生成时出错: {str(e)}")

@tree.command(name="batchstatus", description="查看批量生成任务的状态")
@app_commands.describe(
    task_id="要查询的任务ID (可选，留空查看所有任务)"
)
async def batchstatus_command(interaction: discord.Interaction, task_id: str = None):
    user_id = str(interaction.user.id)
    
    # 获取用户的任务
    user_tasks = {}
    for t_id, tasks in batch_tasks.items():
        if user_id in tasks:
            user_tasks[t_id] = tasks[user_id]
    
    if not user_tasks:
        await interaction.response.send_message("你没有正在进行的批量任务。", ephemeral=True)
        return
    
    # 如果指定了任务ID
    if task_id:
        if task_id not in batch_tasks or user_id not in batch_tasks[task_id]:
            await interaction.response.send_message(f"未找到指定的任务 `{task_id}`。", ephemeral=True)
            return
        
        task = batch_tasks[task_id][user_id]
        
        # 创建任务状态消息
        status_text = "进行中" if task["status"] == "processing" else \
                    "等待中" if task["status"] == "pending" else \
                    "已完成" if task["status"] == "completed" else \
                    "已取消"
        
        progress = f"{task['current']}/{task['total']}"
        
        embed = discord.Embed(
            title=f"批量任务 {task_id} 状态",
            description=f"任务状态: {status_text}",
            color=0x3498db
        )
        
        embed.add_field(name="进度", value=progress, inline=True)
        embed.add_field(name="创建时间", value=task["created_at"].strftime("%Y-%m-%d %H:%M:%S"), inline=True)
        
        if task["status"] == "completed" and "completed_at" in task:
            duration = task["completed_at"] - task["created_at"]
            minutes, seconds = divmod(duration.seconds, 60)
            embed.add_field(name="完成时间", value=task["completed_at"].strftime("%Y-%m-%d %H:%M:%S"), inline=True)
            embed.add_field(name="耗时", value=f"{minutes}分{seconds}秒", inline=True)
        
        # 添加操作说明
        if task["status"] in ["processing", "pending"]:
            embed.add_field(
                name="操作",
                value="使用 `/cancelbatch " + task_id + "` 取消此任务",
                inline=False
            )
        
        await interaction.response.send_message(embed=embed, ephemeral=True)
    else:
        # 显示所有任务的摘要
        embed = discord.Embed(
            title="批量任务列表",
            description=f"你有 {len(user_tasks)} 个批量任务",
            color=0x3498db
        )
        
        for t_id, task in user_tasks.items():
            status_text = "进行中" if task["status"] == "processing" else \
                        "等待中" if task["status"] == "pending" else \
                        "已完成" if task["status"] == "completed" else \
                        "已取消"
                        
            progress = f"{task['current']}/{task['total']}"
            
            embed.add_field(
                name=f"任务 {t_id}",
                value=f"状态: {status_text}\n进度: {progress}\n创建: {task['created_at'].strftime('%m-%d %H:%M')}",
                inline=True
            )
        
        # 添加使用说明
        embed.add_field(
            name="查看详情",
            value="使用 `/batchstatus [任务ID]` 查看任务详细状态",
            inline=False
        )
        
        await interaction.response.send_message(embed=embed, ephemeral=True)

@tree.command(name="cancelbatch", description="取消正在进行的批量生成任务")
@app_commands.describe(
    task_id="要取消的任务ID"
)
async def cancelbatch_command(interaction: discord.Interaction, task_id: str):
    user_id = str(interaction.user.id)
    
    if task_id not in batch_tasks or user_id not in batch_tasks[task_id]:
        await interaction.response.send_message(f"未找到指定的任务 `{task_id}`。", ephemeral=True)
        return
    
    task = batch_tasks[task_id][user_id]
    
    # 检查任务是否已经完成或已取消
    if task["status"] in ["completed", "cancelled"]:
        await interaction.response.send_message(f"任务 `{task_id}` 已经 {task['status']}，无法取消。", ephemeral=True)
        return
    
    # 取消任务
    task["status"] = "cancelled"
    
    await interaction.response.send_message(f"✅ 已取消任务 `{task_id}`。", ephemeral=True)

# ===== 协作生成命令 =====
@tree.command(name="relay", description="开始接力生成图像的协作会话")
@app_commands.describe(
    initial_prompt="初始提示词",
    max_participants="最大参与人数",
    duration_minutes="会话持续时间(分钟)"
)
async def relay_command(
    interaction: discord.Interaction, 
    initial_prompt: str, 
    max_participants: int = 5, 
    duration_minutes: int = 60
):
    await interaction.response.defer()
    
    # 检查是否在服务器中
    guild_id = interaction.guild_id
    if not guild_id:
        await interaction.followup.send("❌ 此命令只能在服务器中使用。", ephemeral=True)
        return
        
    # 获取API密钥 - 复用现有函数
    api_key, provider_info = await get_api_key(interaction)
    if not api_key:
        return
        
    # 创建会话
    session_id = f"relay_{guild_id}_{int(time.time())}"
    expires_at = datetime.datetime.now() + datetime.timedelta(minutes=duration_minutes)
    
    relay_sessions[session_id] = {
        "guild_id": str(guild_id),
        "channel_id": str(interaction.channel_id),
        "creator_id": str(interaction.user.id),
        "participants": [str(interaction.user.id)],
        "participant_names": [interaction.user.display_name],
        "current_prompt": initial_prompt,
        "max_participants": max_participants,
        "expires_at": expires_at,
        "is_completed": False,
        "api_key": api_key,
        "provider_info": provider_info,
        "message_id": None  # 将在发送后更新
    }
    
    # 创建嵌入消息
    embed = discord.Embed(
        title="🏆 图像生成接力",
        description="多人协作完成一幅生成图像！",
        color=0x3498db
    )
    
    embed.add_field(name="💭 当前提示词", value=initial_prompt, inline=False)
    embed.add_field(name="👥 已参与", value=f"1/{max_participants}: {interaction.user.display_name}", inline=True)
    embed.add_field(name="⏰ 截止时间", value=f"<t:{int(expires_at.timestamp())}:R>", inline=True)
    
    # 使用共享视图类
    view = RelayButtons(session_id, expires_at)
    message = await interaction.followup.send(embed=embed, view=view)
    
    # 保存消息ID以便后续更新
    relay_sessions[session_id]["message_id"] = message.id

async def handle_relay_add_content(interaction, session_id):
    """处理添加内容到接力会话的请求"""
    # 注意：不要在这里使用defer，因为我们要发送模态窗口
    
    if session_id not in relay_sessions:
        await interaction.response.send_message("❌ 此接力会话已不存在或已过期。", ephemeral=True)
        return
        
    session = relay_sessions[session_id]
    
    # 检查会话是否已完成
    if session["is_completed"]:
        await interaction.response.send_message("❌ 此接力会话已完成。", ephemeral=True)
        return
        
    # 检查是否已达到最大参与人数
    user_id = str(interaction.user.id)
    if len(session["participants"]) >= session["max_participants"] and user_id not in session["participants"]:
        await interaction.response.send_message(f"❌ 此接力会话已达到最大参与人数 ({session['max_participants']})。", ephemeral=True)
        return
        
    # 显示输入对话框
    class AddContentModal(discord.ui.Modal, title="添加接力内容"):
        content = discord.ui.TextInput(
            label="添加到提示词", 
            placeholder="输入你想要添加到提示词的内容...", 
            min_length=1, 
            max_length=200,
            style=discord.TextStyle.paragraph
        )

        async def on_submit(self, modal_interaction):
            await modal_interaction.response.defer(ephemeral=True)
            
            try:
                # 更新会话内容
                new_content = self.content.value.strip()
                current_prompt = session["current_prompt"]
                
                # 添加新内容
                updated_prompt = f"{current_prompt}, {new_content}"
                session["current_prompt"] = updated_prompt
                
                # 添加参与者（如果是新参与者）
                if user_id not in session["participants"]:
                    session["participants"].append(user_id)
                    session["participant_names"].append(interaction.user.display_name)
                
                # 发送新的更新消息，而不是尝试编辑原始消息
                try:
                    channel = client.get_channel(int(session["channel_id"]))
                    if channel:
                        # 创建新的嵌入消息
                        embed = discord.Embed(
                            title="🔄 接力生成更新",
                            description=f"**{interaction.user.display_name}** 添加了新内容",
                            color=0x9B59B6
                        )
                        
                        embed.add_field(name="💭 当前提示词", value=updated_prompt, inline=False)
                        embed.add_field(
                            name="👥 参与情况", 
                            value=f"{len(session['participant_names'])}/{session['max_participants']} 名参与者", 
                            inline=True
                        )
                        embed.add_field(
                            name="⏰ 截止时间", 
                            value=f"<t:{int(session['expires_at'].timestamp())}:R>", 
                            inline=True
                        )
                        
                        # 创建新的按钮视图
                        view = RelayButtons(session_id, session["expires_at"])
                        
                        # 发送新消息
                        await channel.send(embed=embed, view=view)
                except Exception as update_error:
                    print(f"发送更新消息时出错: {update_error}")
                
                await modal_interaction.followup.send(
                    f"✅ 你已成功添加内容: \"{new_content}\"\n当前提示词: {updated_prompt}", 
                    ephemeral=True
                )
            except Exception as e:
                await modal_interaction.followup.send(f"❌ 添加内容时出错: {str(e)}", ephemeral=True)
    
    # 发送模态窗口
    await interaction.response.send_modal(AddContentModal())

# 2. 完成接力功能
async def handle_relay_complete(interaction, session_id):
    """完成接力会话并生成最终图像"""
    await interaction.response.defer()
    
    if session_id not in relay_sessions:
        await interaction.followup.send("❌ 此接力会话已不存在或已过期。", ephemeral=True)
        return
        
    session = relay_sessions[session_id]
    
    # 检查会话是否已完成
    if session["is_completed"]:
        await interaction.followup.send("❌ 此接力会话已完成。", ephemeral=True)
        return
        
    # 检查是否是参与者
    user_id = str(interaction.user.id)
    if user_id not in session["participants"]:
        await interaction.followup.send("❌ 只有参与者可以完成接力会话。", ephemeral=True)
        return
    
    # 标记会话为已完成
    session["is_completed"] = True
    
    # 获取最终提示词
    final_prompt = session["current_prompt"]
    
    # 使用API生成最终图像
    api_key = session["api_key"]
    provider_info = session["provider_info"]
    
    # 增强负面提示词以避免不适当内容
    stronger_negative_prompt = DEFAULT_NEG_PROMPT + ", "
    
    # 获取适合模型的参数
    selected_model = DEFAULT_MODEL
    model_params = get_model_default_params(selected_model)
    model_params["negative_prompt"] = stronger_negative_prompt
    
    # 准备API请求
    payload = {
        "input": final_prompt,
        "model": selected_model,
        "action": "generate",
        "parameters": model_params
    }
    
    try:
        # 生成图像
        image_data = await send_novelai_request(api_key, payload, interaction)
        if image_data is None:
            await interaction.followup.send("❌ 生成最终图像失败。请稍后重试。", ephemeral=False)
            return
        
        # 创建文件对象并发送
        file = discord.File(io.BytesIO(image_data), filename="relay_final.png")
        
        # 创建嵌入消息
        embed = discord.Embed(
            title="🎉 接力生成完成!",
            description=f"由 {len(session['participants'])} 名成员共同创作",
            color=0x2ecc71
        )
        
        embed.add_field(name="📝 最终提示词", value=final_prompt, inline=False)
        embed.add_field(name="👥 参与者", value=", ".join(session["participant_names"]), inline=False)
        embed.add_field(name="🎨 模型", value=selected_model, inline=True)
        
        if provider_info:
            embed.add_field(name="🔑 API密钥", value=provider_info, inline=True)
            
        embed.set_image(url="attachment://relay_final.png")
        embed.set_footer(text=f"接力会话完成 • 由 {interaction.user.display_name} 确认完成")
        
        try:
            await interaction.followup.send(file=file, embed=embed)
        except discord.errors.HTTPException as http_error:
            if "error code: 20009" in str(http_error):
                # 处理不适当内容错误
                await interaction.followup.send(
                    "❌ Discord检测到生成的图像可能包含不适当内容，无法发送。\n"
                    "请尝试使用不同的提示词或添加更多的负面提示词。\n"
                    f"最终提示词为: {final_prompt}",
                    ephemeral=False
                )
            else:
                await interaction.followup.send(f"❌ 发送图像时出错: {http_error}", ephemeral=False)
    except Exception as e:
        await interaction.followup.send(f"❌ 完成接力过程中出错: {str(e)}", ephemeral=False)
    finally:
        # 删除会话数据以释放内存
        if session_id in relay_sessions:
            del relay_sessions[session_id]

# ===== 状态和信息命令 =====
@tree.command(name="checkapi", description="检查NovelAI API的可用性状态")
async def checkapi_command(interaction: discord.Interaction):
    await interaction.response.defer()
    
    try:
        # 检查NovelAI网站连通性
        site_response = await client.loop.run_in_executor(
            None,
            lambda: requests.get("https://novelai.net/", timeout=10)
        )
        
        if site_response.status_code == 200:
            site_status = "✅ NovelAI网站可以访问，API可能正常工作。"
        else:
            site_status = f"⚠️ NovelAI网站返回了状态码 {site_response.status_code}，API可能存在问题。"
    
    except requests.exceptions.RequestException as e:
        site_status = f"❌ 无法连接到NovelAI网站: {str(e)}"
    
    embed = discord.Embed(
        title="NovelAI API 状态检查",
        color=0xf75c7e
    )
    
    embed.add_field(name="当前状态", value=site_status, inline=False)
    embed.add_field(name="已知问题", 
                   value="• v4模型可能返回500内部服务器错误\n• 如果遇到v4模型的500错误，建议尝试使用v3模型代替。", 
                   inline=False)
    
    await interaction.followup.send(embed=embed)

@tree.command(name="botstatus", description="检查机器人的当前状态和性能")
async def botstatus_command(interaction: discord.Interaction):
    # 延迟响应，告诉Discord我们需要更多时间
    await interaction.response.defer()
    
    # 收集状态信息
    total_keys = len(api_keys)
    shared_keys_count = len([1 for key_data in api_keys.values() if key_data.get("shared_guilds")])
    persistent_keys = len([1 for key_data in api_keys.values() if key_data.get("persist", False)])
    
    # 计算即将过期的密钥
    soon_expire = 0
    for key_data in api_keys.values():
        if "expires_at" in key_data and key_data["expires_at"]:
            time_left = (key_data["expires_at"] - datetime.datetime.now()).total_seconds()
            if 0 < time_left < 24*3600:  # 24小时内过期
                soon_expire += 1
    
    # 计算机器人运行时间 - 使用全局启动时间变量
    current_time = datetime.datetime.now()
    uptime = current_time - BOT_START_TIME
    
    days = uptime.days
    hours, remainder = divmod(uptime.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{days}天 {hours}小时 {minutes}分钟 {seconds}秒"
    
    # 收集模板数据
    templates_count = len(prompt_templates)
    
    # 收集队列数据
    active_queues = len([q for q in generation_queues.values() if q["queue"]])
    total_queued = sum(len(q["queue"]) for q in generation_queues.values())
    
    # 收集批量任务数据
    active_batch_tasks = 0
    pending_batch_tasks = 0
    for task_group in batch_tasks.values():
        for task in task_group.values():
            if task["status"] == "processing":
                active_batch_tasks += 1
            elif task["status"] == "pending":
                pending_batch_tasks += 1
    
    # 收集协作会话数据
    active_relays = len([s for s in relay_sessions.values() if not s["is_completed"]])
    
    # 构建状态嵌入消息
    embed = discord.Embed(
        title="📊 NovelAI Bot 状态",
        description="机器人当前运行状态和性能信息",
        color=0x3498db
    )
    
    embed.add_field(name="🤖 运行状态", value="✅ 正常运行中", inline=False)
    embed.add_field(name="🔑 API密钥统计", 
                   value=f"总数: {total_keys}\n共享密钥: {shared_keys_count}\n持久化密钥: {persistent_keys}\n即将过期: {soon_expire}", 
                   inline=True)
    embed.add_field(name="🗂️ 模板统计", 
                   value=f"总数: {templates_count}", 
                   inline=True)
    embed.add_field(name="📋 队列统计", 
                   value=f"活跃队列: {active_queues}\n等待任务: {total_queued}", 
                   inline=True)
    embed.add_field(name="📊 批量任务", 
                   value=f"活跃任务: {active_batch_tasks}\n等待任务: {pending_batch_tasks}", 
                   inline=True)
    embed.add_field(name="👥 协作会话", 
                   value=f"活跃接力: {active_relays}", 
                   inline=True)
    embed.add_field(name="📡 Discord连接", 
                   value=f"延迟: {round(client.latency * 1000, 2)}ms", 
                   inline=True)
    embed.add_field(name="⏱️ 运行时间", 
                   value=f"{uptime_str}", 
                   inline=True)
    
    # NovelAI API状态检查结果
    try:
        # 简单检查NovelAI网站连通性
        site_response = await client.loop.run_in_executor(
            None,
            lambda: requests.get("https://novelai.net/", timeout=5)
        )
        
        if site_response.status_code == 200:
            api_status = "✅ 可用"
        else:
            api_status = f"⚠️ 状态码: {site_response.status_code}"
    
    except requests.exceptions.RequestException:
        api_status = "❌ 连接失败"
    
    embed.add_field(name="🌐 NovelAI API", value=api_status, inline=False)
    
    # 添加版本信息和时间戳
    embed.set_footer(text=f"Bot版本: {VERSION} • {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    await interaction.followup.send(embed=embed)

# ===== GitHub 热更新功能 =====
@tree.command(name="update", description="从GitHub更新机器人代码")
@app_commands.describe(
    branch="要拉取的分支名称",
    force="是否强制更新，覆盖本地修改"
)
async def update_command(interaction: discord.Interaction, branch: str = "main", force: bool = False):
    # 检查权限(只允许机器人管理员使用)
    user_id = str(interaction.user.id)
    
    if not BOT_ADMIN_IDS or user_id not in BOT_ADMIN_IDS:
        await interaction.response.send_message("❌ 你没有权限执行更新操作。", ephemeral=True)
        return
    
    await interaction.response.defer(thinking=True)
    
    try:
        # 检查git依赖
        try:
            import git
        except ImportError:
            await interaction.followup.send("❌ 未安装git模块。请先运行 `pip install gitpython`。")
            return
            
        # 检查是否是git仓库
        try:
            repo = git.Repo('.')
        except git.exc.InvalidGitRepositoryError:
            await interaction.followup.send("❌ 当前目录不是git仓库。")
            return
            
        # 获取当前版本
        current_commit = repo.head.commit
        current_version = current_commit.hexsha[:7]
        
        # 检查远程分支
        try:
            origin = repo.remotes.origin
            origin.fetch()
            remote_branch = origin.refs[branch]
        except Exception as e:
            await interaction.followup.send(f"❌ 获取远程分支时出错: {str(e)}")
            return
            
        # 获取远程版本
        remote_commit = remote_branch.commit
        remote_version = remote_commit.hexsha[:7]
        
        # 检查是否有更新
        if current_commit.hexsha == remote_commit.hexsha:
            await interaction.followup.send(f"✅ 已是最新版本 ({current_version})，无需更新。")
            return
            
        # 显示更新信息
        commits_between = list(repo.iter_commits(f"{current_commit.hexsha}..{remote_commit.hexsha}"))
        update_info = "\n".join([f"• {commit.message.split('\n')[0]}" for commit in commits_between[:5]])
        
        if len(commits_between) > 5:
            update_info += f"\n• ...以及另外 {len(commits_between) - 5} 条提交"
            
        # 备份当前状态
        backup_path = f"backup_{int(time.time())}"
        os.makedirs(backup_path, exist_ok=True)
        
        # 备份所有Python文件
        for root, dirs, files in os.walk("."):
            # 跳过备份目录
            if root.startswith(f"./{backup_path}"):
                continue
                
            # 跳过Git目录
            if ".git" in root:
                continue
                
            # 创建对应的备份目录结构
            backup_dir = os.path.join(backup_path, root[2:])  # 去掉开头的 ./
            os.makedirs(backup_dir, exist_ok=True)
            
            # 复制所有Python文件
            for file in files:
                if file.endswith(".py"):
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(backup_dir, file)
                    shutil.copy2(src_file, dst_file)
        
        # 执行git操作
        if force:
            # 强制更新，丢弃本地修改
            repo.git.reset('--hard', remote_branch.name)
        else:
            # 尝试合并更新
            try:
                repo.git.pull('origin', branch)
            except git.GitCommandError as e:
                await interaction.followup.send(
                    f"❌ 拉取更新失败: {str(e)}\n\n"
                    f"您可能有本地修改冲突。尝试使用 `--force` 参数进行强制更新。"
                )
                return
                
        # 检查依赖更新
        try:
            if os.path.exists("requirements.txt"):
                os.system("pip install -r requirements.txt")
        except Exception as e:
            await interaction.followup.send(f"⚠️ 更新依赖时出现问题: {str(e)}")
            
        # 发送成功消息
        success_message = (
            f"✅ 更新成功!\n\n"
            f"从 {current_version} 更新到 {remote_version}\n\n"
            f"更新内容:\n{update_info}\n\n"
            f"已在 {backup_path} 创建备份。\n"
            f"将在10秒后重启机器人..."
        )
        
        await interaction.followup.send(success_message)
        
        # 保存所有状态
        save_api_keys_to_file()
        save_templates_to_file()
        
        # 延迟后重启
        await asyncio.sleep(10)
        
        # 重启程序
        os.execv(sys.executable, ['python'] + sys.argv)
        
    except Exception as e:
        await interaction.followup.send(f"❌ 更新过程中出错: {str(e)}\n{traceback.format_exc()}")

# ===== 预览批量生成 =====
@tree.command(name="previewbatch", description="预览批量生成的组合而不实际生成图像")
@app_commands.describe(
    prompt="图像提示词模板，使用 {var1} {var2} 语法表示变量",
    variations="变量值列表，格式: var1=值1,值2,值3|var2=值4,值5,值6",
    param_variations="参数变化，格式: model=模型1,模型2|size=832x1216,1024x1024"
)
async def previewbatch_command(
    interaction: discord.Interaction, 
    prompt: str,
    variations: str = "",
    param_variations: str = ""
):
    await interaction.response.defer(thinking=True)
    
    try:
        # 解析变量定义
        var_definitions = {}
        for part in variations.split('|'):
            if '=' not in part:
                continue
                
            var_name, var_values = part.split('=', 1)
            var_name = var_name.strip()
            var_values = [v.strip() for v in var_values.split(',')]
            var_definitions[var_name] = var_values
        
        # 解析参数变化
        param_var_definitions = {}
        if param_variations:
            for part in param_variations.split('|'):
                if '=' not in part:
                    continue
                    
                param_name, param_values = part.split('=', 1)
                param_name = param_name.strip().lower()
                param_values = [v.strip() for v in param_values.split(',')]
                param_var_definitions[param_name] = param_values
        
        # 生成所有可能的组合
        import itertools
        
        # 提示词变量组合
        prompt_vars_to_combine = []
        prompt_var_names = []
        
        for var_name, values in var_definitions.items():
            prompt_vars_to_combine.append(values)
            prompt_var_names.append(var_name)
            
        prompt_combinations = list(itertools.product(*prompt_vars_to_combine)) if prompt_vars_to_combine else [tuple()]
        
        # 参数变量组合
        param_vars_to_combine = []
        param_var_names = []
        
        for param_name, values in param_var_definitions.items():
            param_vars_to_combine.append(values)
            param_var_names.append(param_name)
            
        param_combinations = list(itertools.product(*param_vars_to_combine)) if param_vars_to_combine else [tuple()]
        
        # 计算总组合数
        total_combinations = len(prompt_combinations) * len(param_combinations)
        
        # 预览所有组合
        combinations_preview = []
        count = 0
        
        # 最多预览50个组合
        for prompt_combo in prompt_combinations:
            # 创建当前组合的提示词
            current_prompt = prompt
            for j, var_name in enumerate(prompt_var_names):
                if j < len(prompt_combo):  # 确保索引有效
                    current_prompt = current_prompt.replace(f"{{{var_name}}}", prompt_combo[j])
            
            for param_combo in param_combinations:
                # 当前组合的参数
                current_params = {}
                for k, param_name in enumerate(param_var_names):
                    if k < len(param_combo):  # 确保索引有效
                        current_params[param_name] = param_combo[k]
                
                # 添加到预览列表
                combinations_preview.append({
                    "prompt": current_prompt,
                    "params": current_params
                })
                
                count += 1
                if count >= 50:
                    break
                    
            if count >= 50:
                break
        
        # 生成预览嵌入消息
        embed = discord.Embed(
            title="批量生成预览",
            description=f"模板: {prompt}",
            color=0x3498db
        )
        
        embed.add_field(name="提示词变量", value=", ".join([f"{k}={len(v)}个值" for k, v in var_definitions.items()]) or "无", inline=True)
        embed.add_field(name="参数变量", value=", ".join([f"{k}={len(v)}个值" for k, v in param_var_definitions.items()]) or "无", inline=True)
        embed.add_field(name="总组合数", value=f"{total_combinations}个" + (" (仅预览前50个)" if total_combinations > 50 else ""), inline=False)
        
        # 添加组合预览示例
        preview_text = ""
        for i, combo in enumerate(combinations_preview[:10], 1):
            param_text = ", ".join([f"{k}={v}" for k, v in combo["params"].items()]) if combo["params"] else "默认参数"
            preview_text += f"{i}. 提示词: {combo['prompt'][:50]}{'...' if len(combo['prompt']) > 50 else ''}\n   参数: {param_text}\n\n"
        
        if combinations_preview:
            embed.add_field(name="组合示例", value=preview_text, inline=False)
        
        # 添加使用说明
        embed.add_field(
            name="生成指令",
            value=f"使用 `/naibatch` 命令并传入相同参数来开始批量生成。",
            inline=False
        )
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        print(f"预览批量生成时出错: {str(e)}")
        print(traceback.format_exc())
        await interaction.followup.send(f"❌ 预览批量生成时出错: {str(e)}")

# ===== 帮助命令 =====
@tree.command(name="help", description="显示帮助信息")
async def help_command(interaction: discord.Interaction):
    embed = discord.Embed(
        title="NovelAI 机器人帮助", 
        description="这个机器人使用NovelAI API生成图像。以下是可用的命令：", 
        color=0xf75c7e
    )
    
    embed.add_field(
        name="🖼️ 图像生成命令",
        value=(
            "• `/nai [prompt] [model] [template_id]` - 使用基础设置快速生成图像\n"
            "• `/naigen [prompt] [options...] [template_id]` - 使用高级设置生成图像\n"
            "• `/naivariation [index] [type]` - 基于最近生成的图像创建变体\n"
            "• `/naibatch [prompt] [variations] [param_variations]` - 批量生成多个变体图像\n"
            "• `/previewbatch [prompt] [variations]` - 预览批量生成而不实际生成图像\n"
            "• `/batchstatus [task_id]` - 查看批量生成任务状态\n"
            "• `/cancelbatch [task_id]` - 取消批量生成任务\n"
            "• `/relay [prompt]` - 开始一个接力生成协作会话"
        ),
        inline=False
    )
    
    embed.add_field(
        name="📝 提示词模板",
        value=(
            "• `/savetemplate [name] [prompt] [save_params]` - 保存提示词模板\n"
            "• `/listtemplates [filter_tags]` - 查看可用的提示词模板\n"
            "• `/usetemplate [id] [override_prompt]` - 使用模板生成图像\n"
            "• `/updatetemplate [id] [new_params]` - 更新现有模板\n"
            "• `/deletetemplate [id]` - 删除你创建的模板"
        ),
        inline=False
    )
    
    embed.add_field(
        name="🔑 API密钥管理",
        value=(
            "• `/apikey [key] [sharing]` - 注册或管理API密钥\n"
            "• `/sharedkeys` - 查看服务器共享的API密钥\n"
            "• `/addsharing` - 在当前服务器共享你的密钥\n"
            "• `/removesharing` - 停止在当前服务器共享\n"
            "• `/deletekey` - 删除你注册的API密钥"
        ),
        inline=False
    )
    
    embed.add_field(
        name="🔧 状态检查",
        value=(
            "• `/checkapi` - 检查NovelAI API状态\n"
            "• `/botstatus` - 查看机器人运行状态和性能"
        ),
        inline=False
    )
    
    embed.add_field(
        name="⭐ 新功能与改进",
        value=(
            "• **模板增强**: 模板现在可以保存完整参数并与其他命令结合使用\n"
            "• **批量生成扩展**: 支持同时变化提示词和生成参数\n"
            "• **接力生成改进**: 修复内容添加后的消息更新问题\n"
            "• **预览功能**: 可以预览批量生成的组合而不实际生成图像"
        ),
        inline=False
    )
    
    embed.add_field(
        name="ℹ️ 关于版本",
        value=f"版本: v{VERSION}\n"
              f"有关最新更新和详细用法，请访问GitHub仓库。",
        inline=False
    )
    
    await interaction.response.send_message(embed=embed)

# ===== 主函数 =====
if __name__ == "__main__":
    # 使用配置文件中的令牌，如果没有则尝试从环境变量获取
    TOKEN = DISCORD_TOKEN
    if not TOKEN:
        print("错误: 未设置DISCORD_TOKEN，请在config.txt文件或环境变量中配置")
        exit(1)
    
    # 显示已加载的配置
    print(f"已加载配置:")
    print(f"- 默认模型: {DEFAULT_MODEL}")
    print(f"- 默认尺寸: {DEFAULT_SIZE}")
    print(f"- 默认步数: {DEFAULT_STEPS}")
    
    # 运行Discord机器人
    client.run(TOKEN)