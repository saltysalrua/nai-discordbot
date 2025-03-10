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
from typing import Dict, Optional, List, Union, Literal

# 全局变量，记录每个密钥的使用情况
key_usage_counter = {}
key_last_used = {}

# 记录机器人启动时间的全局变量
BOT_START_TIME = datetime.datetime.now()

# 读取配置文件函数
def read_config_file(file_path="config.txt"):
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

# 从配置文件加载设置，如果找不到则使用默认值
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

# Discord机器人设置
intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# NovelAI API设置
NAI_API_URL = "https://image.novelai.net/ai/generate-image"

# 用户API密钥存储
# 结构: {user_id: {"key": api_key, "shared_guilds": [guild_ids], "expires_at": datetime, "provider_name": "用户名", "persist": bool}}
api_keys = {}

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

# 保存API密钥到文件
def save_api_keys_to_file():
    """将标记为持久化的API密钥保存到文件"""
    # 只保存标记为持久化的密钥
    keys_to_save = {
        user_id: data.copy() 
        for user_id, data in api_keys.items() 
        if data.get("persist", False)
    }
    
    # 如果没有需要保存的密钥，则不进行任何操作
    if not keys_to_save:
        return
    
    # 准备用于序列化的数据
    serializable_dict = {}
    for user_id, data in keys_to_save.items():
        serializable_data = data.copy()
        # 处理datetime对象
        if "expires_at" in serializable_data and serializable_data["expires_at"]:
            serializable_data["expires_at"] = serializable_data["expires_at"].isoformat()
        serializable_dict[user_id] = serializable_data
    
    try:
        # 保存数据到JSON文件
        with open("api_keys.json", "w", encoding="utf-8") as f:
            json.dump(serializable_dict, f, ensure_ascii=False, indent=2)
        
        print(f"已保存 {len(keys_to_save)} 个API密钥")
    except Exception as e:
        print(f"保存API密钥时出错: {str(e)}")

# 从文件加载API密钥
def load_api_keys_from_file():
    """从文件加载API密钥"""
    if not os.path.exists("api_keys.json"):
        print("未找到API密钥文件")
        return {}
    
    try:
        # 读取JSON数据
        with open("api_keys.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 将字符串日期转换回datetime对象
        import datetime
        for user_id, key_data in data.items():
            if "expires_at" in key_data and key_data["expires_at"]:
                key_data["expires_at"] = datetime.datetime.fromisoformat(key_data["expires_at"])
        
        print(f"已成功加载 {len(data)} 个API密钥")
        return data
    
    except Exception as e:
        print(f"加载API密钥时出错: {str(e)}")
        return {}

@client.event
async def on_ready():
    print(f'机器人已登录为 {client.user}')
    await tree.sync()  # 同步斜杠命令
    
    # 从文件加载API密钥
    global api_keys
    loaded_keys = load_api_keys_from_file()
    if loaded_keys:
        api_keys.update(loaded_keys)
        print(f"已从文件加载 {len(loaded_keys)} 个API密钥")
    
    # 启动密钥过期检查任务
    client.loop.create_task(check_expired_keys())
    # 启动定期保存任务
    client.loop.create_task(periodic_save_keys())
    # 启动每小时密钥验证任务
    client.loop.create_task(hourly_validate_keys())

# 改进的API请求处理函数
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

# 定期保存任务
async def periodic_save_keys():
    """定期保存标记为持久化的API密钥"""
    while True:
        await asyncio.sleep(60 * 15)  # 每15分钟保存一次
        save_api_keys_to_file()

# 检查API密钥有效性
async def check_api_key_validity(api_key):
    """检查API密钥是否有效"""
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
            return False
        return True
    except:
        # 连接错误也视为可能无效
        return False

# 每小时检查密钥有效性
async def hourly_validate_keys():
    """每小时检查API密钥有效性"""
    while True:
        await asyncio.sleep(3600)  # 每小时检查一次
        print(f"[{datetime.datetime.now()}] 开始执行API密钥有效性检查...")
        
        invalid_keys = []
        checked_count = 0
        
        for user_id, key_data in list(api_keys.items()):
            # 先检查是否已过期
            if "expires_at" in key_data and key_data["expires_at"] and key_data["expires_at"] < datetime.datetime.now():
                print(f"密钥已过期: {user_id}")
                invalid_keys.append(user_id)
                continue
            
            # 检查API密钥有效性
            is_valid = await check_api_key_validity(key_data["key"])
            checked_count += 1
            
            if not is_valid:
                print(f"密钥无效: {user_id}")
                invalid_keys.append(user_id)
            
            # 每检查几个密钥暂停一下，避免过快请求
            if checked_count % 5 == 0:
                await asyncio.sleep(2)
        
        # 移除无效密钥
        for user_id in invalid_keys:
            del api_keys[user_id]
        
        # 如果有删除持久化密钥，保存更新
        if any(user_id in api_keys and api_keys[user_id].get("persist", False) for user_id in invalid_keys):
            save_api_keys_to_file()
        
        print(f"[{datetime.datetime.now()}] API密钥检查完成，检查了 {checked_count} 个密钥，移除了 {len(invalid_keys)} 个无效密钥")

# 密钥管理命令
@tree.command(name="apikey", description="注册或管理你的NovelAI API密钥")
@app_commands.describe(
    key="你的NovelAI API密钥",
    sharing="设置密钥是否在此服务器共享",
    duration_hours="密钥有效时间(小时), 0表示永不过期",
    persist="是否在机器人重启后保存密钥（会进行加密存储）"
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
            "• 你的API密钥将被存储在机器人所在的服务器上（注意：不进行加密）\n"
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

# 删除密钥命令
@tree.command(name="deletekey", description="删除你注册的NovelAI API密钥")
async def deletekey_command(interaction: discord.Interaction):
    user_id = str(interaction.user.id)
    
    if user_id in api_keys:
        was_persistent = api_keys[user_id].get("persist", False)
        del api_keys[user_id]
        
        # 如果是持久化密钥，立即更新存储
        if was_persistent:
            save_api_keys_to_file()
        
        await interaction.response.send_message(
            "✅ 你的API密钥已从机器人中删除。" + 
            ("所有持久化存储的数据也已清除。" if was_persistent else ""), 
            ephemeral=True
        )
    else:
        await interaction.response.send_message("你没有注册API密钥。", ephemeral=True)

# 添加密钥到服务器共享命令
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

# 从服务器共享列表中移除密钥命令
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
    
    # 用户没有注册密钥，也没有可用的共享密钥
    if guild_id:
        shared_keys_info = get_guild_shared_keys_info(guild_id)
        if shared_keys_info:
            await interaction.followup.send(
                f"⚠️ 你需要先注册你的NovelAI API密钥才能使用此功能。\n\n"
                f"当前服务器有 {len(shared_keys_info)} 个共享的API密钥，但这些密钥可能已过期或不可用。\n"
                f"请使用 `/apikey [你的密钥]` 命令注册，或联系密钥提供者更新共享设置。", 
                ephemeral=True
            )
        else:
            await interaction.followup.send(
                "⚠️ 你需要先注册你的NovelAI API密钥才能使用此功能。请使用 `/apikey [你的密钥]` 命令注册。", 
                ephemeral=True
            )
    else:
        await interaction.followup.send(
            "⚠️ 你需要先注册你的NovelAI API密钥才能使用此功能。请使用 `/apikey [你的密钥]` 命令注册。", 
            ephemeral=True
        )
    
    return None, None

# 展示服务器共享密钥列表
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

# 添加NovelAI API状态检查命令
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

# 添加Bot状态检查命令
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
    embed.set_footer(text=f"Bot版本: 1.2.0 • {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 使用followup而不是直接响应，因为我们已经延迟了
    await interaction.followup.send(embed=embed)

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

# 新增高级生成命令
@tree.command(name="naigen", description="使用NovelAI生成图像 (高级选项)")
@app_commands.describe(
    prompt="图像生成提示词",
    model="选择模型",
    size="图像尺寸 (宽x高)",
    steps="采样步数 (1-28)",  # 更新描述
    scale="CFG比例 (1-10)",
    sampler="采样器",
    noise_schedule="噪声调度",
    negative_prompt="负面提示词",
    smea="启用SMEA (仅v3模型)",
    dynamic_smea="启用动态SMEA (仅v3模型)",
    cfg_rescale="CFG重缩放 (0-1)",
    seed="随机种子 (留空为随机)",
    variety_plus="启用Variety+功能"  # 新增选项
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
    prompt: str,
    model: str = DEFAULT_MODEL,
    size: str = None,
    steps: int = DEFAULT_STEPS,
    scale: float = DEFAULT_SCALE,
    sampler: str = DEFAULT_SAMPLER,
    noise_schedule: str = None,
    negative_prompt: str = None,
    smea: bool = True,
    dynamic_smea: bool = True,
    cfg_rescale: float = DEFAULT_CFG_RESCALE,
    seed: str = None,
    variety_plus: bool = False  # 新增参数
):
    await interaction.response.defer(thinking=True)
    
    try:
        # 获取API密钥
        api_key, provider_info = await get_api_key(interaction)
        if not api_key:
            return
        
        # 处理尺寸
        width, height = DEFAULT_SIZE
        if size:
            try:
                width, height = map(int, size.split('x'))
            except:
                pass
        
        # 确保步数在合理范围内 - 限制最大28步
        steps = max(1, min(28, steps))
        
        # 确保CFG比例在合理范围内
        scale = max(1.0, min(10.0, scale))
        
        # 确保CFG重缩放在合理范围内
        cfg_rescale = max(0.0, min(1.0, cfg_rescale))
        
        # 处理噪声调度，为v4模型自动调整
        if not noise_schedule:
            noise_schedule = "karras" if model.startswith("nai-diffusion-4") else DEFAULT_NOISE_SCHEDULE
        elif noise_schedule == "native" and model.startswith("nai-diffusion-4"):
            noise_schedule = "karras"  # v4不支持native，自动切换为karras
        
        # 处理SMEA设置
        if model.startswith("nai-diffusion-4"):
            smea = False
            dynamic_smea = False
        
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
            "negative_prompt": negative_prompt or DEFAULT_NEG_PROMPT,
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
        if model.startswith("nai-diffusion-4"):
            model_params["params_version"] = 3
            model_params["use_coords"] = True
        
        # 准备API请求
        payload = {
            "input": prompt,
            "model": model,
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
        embed.add_field(name="模型", value=model, inline=True)
        embed.add_field(name="尺寸", value=f"{width}x{height}", inline=True)
        
        # 显示种子值和Variety+状态
        seed_display = seed_value if not random_seed else "随机"
        embed.add_field(name="种子", value=f"{seed_display}", inline=True)
        
        if variety_plus:
            embed.add_field(name="Variety+", value="已启用", inline=True)
        
        # 如果使用的是共享密钥，显示提供者信息
        if provider_info:
            if provider_info == "自己的密钥":
                embed.add_field(name="🔑 API密钥", value="使用自己的密钥", inline=True)
            else:
                embed.add_field(name="🔑 API密钥", value=provider_info, inline=True)
            
        embed.set_image(url="attachment://generated_image.png")
        embed.set_footer(text=f"由 {interaction.user.display_name} 生成")
        
        # 不再显示参数细节，只含基本信息
        await interaction.followup.send(file=file, embed=embed)
        
    except Exception as e:
        print(f"高级生成出错: {str(e)}")
        print(traceback.format_exc())
        await interaction.followup.send(f"❌ 生成图像时出错: {str(e)}")

# 基础生成命令
@tree.command(name="nai", description="使用NovelAI生成图像")
@app_commands.describe(
    prompt="图像生成提示词",
    model="模型选择"
)
@app_commands.choices(
    model=[
        app_commands.Choice(name=f"{model} - {MODEL_DESCRIPTIONS[model]}", value=model)
        for model in AVAILABLE_MODELS
    ]
)
async def nai_command(
    interaction: discord.Interaction, 
    prompt: str,
    model: str = None
):
    await interaction.response.defer(thinking=True)
    
    try:
        # 获取API密钥
        api_key, provider_info = await get_api_key(interaction)
        if not api_key:
            return
        
        # 验证并设置模型
        selected_model = model if model in AVAILABLE_MODELS else DEFAULT_MODEL
        
        # 获取适合模型的参数
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

# 帮助命令
@tree.command(name="help", description="显示帮助信息")
async def help_command(interaction: discord.Interaction):
    embed = discord.Embed(
        title="NovelAI 机器人帮助", 
        description="这个机器人使用NovelAI API生成图像。以下是可用的命令：", 
        color=0xf75c7e
    )
    
    embed.add_field(
        name="/apikey [key] [sharing] [duration_hours] [persist]",
        value=(
            "注册你的NovelAI API密钥。\n"
            "- `key`: 你的API密钥\n"
            "- `sharing`: 设置为「服务器共享」或「私人使用」\n"
            "- `duration_hours`: 密钥有效期(小时)，0表示永不过期\n"
            "- `persist`: 是否在机器人重启后保存密钥（加密存储）"
        ),
        inline=False
    )
    
    embed.add_field(
        name="/nai [prompt] [model]",
        value=(
            "使用基础设置快速生成图像。\n"
            "- `prompt`: 图像提示词\n"
            "- `model`: (可选)模型名称"
        ),
        inline=False
    )
    
    embed.add_field(
        name="/naigen [prompt] [options...]",
        value=(
            "使用高级设置生成图像，提供更多参数控制。\n"
            "- 支持设置尺寸、步数、CFG比例、采样器等\n"
            "- 可以设置随机种子以重现相同结果\n"
            "- 支持启用Variety+功能增强创意多样性"
        ),
        inline=False
    )
    
    embed.add_field(
        name="密钥管理命令",
        value=(
            "- `/sharedkeys`: 查看服务器共享的API密钥\n"
            "- `/addsharing`: 在当前服务器共享你的密钥\n"
            "- `/removesharing`: 停止在当前服务器共享\n"
            "- `/deletekey`: 删除你注册的API密钥"
        ),
        inline=False
    )
    
    embed.add_field(
        name="状态检查命令",
        value=(
            "- `/checkapi`: 检查NovelAI API状态\n"
            "- `/botstatus`: 查看机器人运行状态和性能"
        ),
        inline=False
    )
    
    embed.add_field(
        name="模型兼容性说明",
        value=(
            "• v3模型 (nai-diffusion-3, nai-diffusion-3-furry): 支持SMEA和所有噪声调度\n"
            "• v4模型 (nai-diffusion-4-full, nai-diffusion-4-curated): 不支持SMEA，推荐使用karras噪声调度"
        ),
        inline=False
    )
    
    await interaction.response.send_message(embed=embed)

# 主函数
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