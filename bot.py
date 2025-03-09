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
import threading
import time
import aiohttp
from typing import Dict, Optional, List, Union, Literal, Tuple
from flask import Flask, render_template_string

# 创建Flask应用
app = Flask(__name__)

@app.route('/')
def home():
    # 返回一个简单的HTML页面
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>NovelAI Discord Bot</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                text-align: center;
            }
            .container {
                background-color: #f5f5f5;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #7289DA;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>NovelAI Discord Bot</h1>
            <p>Bot is running! This page helps keep the bot active on Glitch.</p>
            <p>To add the bot to your server, use the Discord Developer Portal.</p>
            <p>Current Status: ✅ Online</p>
        </div>
    </body>
    </html>
    """)

# 启动Flask网页服务器
def run_flask():
    # 设置主机为0.0.0.0以允许外部访问
    app.run(host='0.0.0.0', port=3000)

# Discord机器人设置
intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# NovelAI API设置
NAI_API_URL = "https://image.novelai.net/ai/generate-image"

# 用户API密钥存储
# 结构: {user_id: {"key": api_key, "shared_guilds": [guild_ids], "expires_at": datetime, "provider_name": "用户名"}}
api_keys = {}

# 默认参数
DEFAULT_MODEL = "nai-diffusion-3"  # 更改为v3模型作为默认，因为更稳定
DEFAULT_SIZE = (832, 1216)  # (width, height)
DEFAULT_STEPS = 28
DEFAULT_SCALE = 6.5
DEFAULT_SAMPLER = "k_euler_ancestral"
DEFAULT_NOISE_SCHEDULE = "native"
DEFAULT_CFG_RESCALE = 0.1
DEFAULT_NEG_PROMPT = "lowres, {bad}, error, fewer, extra, missing, worst quality, jpeg artifacts, bad quality, watermark, unfinished, displeasing, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract], bad anatomy, bad hands"

# 可用的选项
AVAILABLE_MODELS = [
    "nai-diffusion-4-full",
    "nai-diffusion-4-curated",
    "nai-diffusion-3",
    "nai-diffusion-3-furry"
]

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

@client.event
async def on_ready():
    print(f'机器人已登录为 {client.user}')
    await tree.sync()  # 同步斜杠命令
    
    # 启动密钥过期检查任务
    client.loop.create_task(check_expired_keys())

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
    parameters = payload.get("parameters", {})
    optimized_parameters = parameters.copy()
    
    # v4模型特殊处理
    if model.startswith("nai-diffusion-4"):
        optimized_parameters["sm"] = False
        optimized_parameters["sm_dyn"] = False
        if optimized_parameters.get("noise_schedule") == "native":
            optimized_parameters["noise_schedule"] = "karras"
        # v4特定参数
        optimized_parameters["params_version"] = 3  # v4需要此参数
        optimized_parameters["use_coords"] = True  # 使用坐标系统
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
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
        "Accept": "*/*"
    }
    
    try:
        # 记录请求开始时间用于计算时长
        start_time = time.time()
        
        # 记录请求参数（仅供调试）
        print(f"请求模型: {model}")
        print(f"请求参数: {json.dumps({k: v for k, v in optimized_parameters.items() if k != 'negative_prompt'})}")
        
        # 发送HTTP请求
        response = await client.loop.run_in_executor(
            None, 
            lambda: requests.post(
                NAI_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
        )
        
        # 计算请求时长
        request_time = time.time() - start_time
        print(f"API请求完成，耗时: {request_time:.2f}秒，状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        
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
                
                # 构建更安全的参数配置 - 使用极度保守的参数
                safe_params = {
                    "width": 768,        # 使用更小的尺寸
                    "height": 1024,
                    "scale": 4.5,        # 进一步降低CFG scale
                    "steps": 22,         # 减少步数
                    "n_samples": 1,
                    "ucPreset": 0,
                    "qualityToggle": True,
                    "sampler": "k_euler", # 改用更简单的采样器
                    "noise_schedule": "karras" if model.startswith("nai-diffusion-4") else "native",
                    "negative_prompt": parameters.get("negative_prompt", DEFAULT_NEG_PROMPT),
                    "sm": False if model.startswith("nai-diffusion-4") else True,
                    "sm_dyn": False if model.startswith("nai-diffusion-4") else True,
                    "cfg_rescale": 0,     # 禁用CFG rescale
                }
                
                # v4模型特定参数
                if model.startswith("nai-diffusion-4"):
                    safe_params["params_version"] = 3
                    safe_params["use_coords"] = True
                    
                    # 如果仍然是v4模型，建议用户尝试v3
                    await interaction.followup.send(
                        "⚠️ v4模型可能目前遇到服务器问题，如果失败请尝试切换到nai-diffusion-3模型。",
                        ephemeral=True
                    )
                
                # 修改参数并重试
                payload["parameters"] = safe_params
                return await send_novelai_request(api_key, payload, interaction, retry_count + 1)
                
            # 重试失败，提供详细诊断信息
            try:
                error_json = response.json()
                error_message = f"❌ NovelAI服务器内部错误(500)。\n\n**错误详情**: {error_json.get('message', '未知错误')}"
            except:
                error_message = f"❌ NovelAI服务器内部错误(500)。\n\n**响应内容**: {response.text[:200]}"
                
            error_message += "\n\n**可能的原因**:\n"
            error_message += "• 提示词中包含不支持的内容\n"
            error_message += "• 所选参数与模型不兼容\n"
            error_message += "• NovelAI服务器临时故障\n\n"
            error_message += "**建议**: \n"
            error_message += "• 尝试切换到nai-diffusion-3模型\n"
            error_message += "• 简化提示词，移除复杂或特殊的描述\n"
            error_message += "• 等待几分钟后再试\n"
            error_message += "• 检查NovelAI官方状态页或Discord服务器\n\n"
            error_message += "使用 `/checkapi` 命令检查API状态。"
            
            await interaction.followup.send(error_message)
            return None
            
        elif response.status_code != 200:
            # 其他非200状态码
            try:
                error_data = response.json()
                error_message = error_data.get('message', f'API错误: 状态码 {response.status_code}')
            except:
                error_message = f'API错误: 状态码 {response.status_code}, 响应内容: {response.text[:100]}...'
            
            await interaction.followup.send(f"❌ NovelAI API返回错误: {error_message}")
            return None
        
        # 检查Content-Type - 更宽松的验证，接受binary/octet-stream和其他二进制类型
        content_type = response.headers.get('Content-Type', '').lower()
        valid_types = ['application/zip', 'application/x-zip-compressed', 'binary/octet-stream', 'application/octet-stream']
        is_valid_type = any(t in content_type for t in valid_types)

        if not is_valid_type:
            # 只有当内容类型明确不是二进制时才报错
            if 'application/json' in content_type:
                try:
                    error_data = response.json()
                    error_message = error_data.get('message', '未知错误')
                    await interaction.followup.send(f"❌ NovelAI API返回错误: {error_message}")
                    return None
                except:
                    pass
            
            # 即使Content-Type不匹配，仍然尝试解析为ZIP
            print(f"警告: 内容类型意外: {content_type}，但仍将尝试解析为ZIP文件")
        
        # 尝试解析ZIP文件
        try:
            # 先保存响应内容用于调试
            response_content = response.content
            
            with zipfile.ZipFile(io.BytesIO(response_content)) as zip_file:
                zip_contents = zip_file.namelist()
                
                # 记录ZIP文件内容，便于调试
                print(f"ZIP文件内容: {zip_contents}")
                
                if "image_0.png" not in zip_contents:
                    await interaction.followup.send(f"❌ ZIP文件中找不到图像文件。内容: {zip_contents}")
                    return None
                    
                image_data = zip_file.read("image_0.png")
                return image_data
        except zipfile.BadZipFile:
            # 如果ZIP解析失败，尝试直接将响应作为图像处理
            print("ZIP解析失败，尝试直接以图像格式处理响应...")
            
            # 检查响应内容的前几个字节，看是否为PNG文件头
            if len(response_content) > 8 and response_content[:8] == b'\x89PNG\r\n\x1a\n':
                print("检测到PNG文件头，直接返回图像数据")
                return response_content
            
            # 如果不是PNG，再尝试看是否是JPEG
            if len(response_content) > 3 and response_content[:3] == b'\xff\xd8\xff':
                print("检测到JPEG文件头，直接返回图像数据")
                return response_content
            
            # 输出一些调试信息
            content_preview = response_content[:50].hex()
            print(f"响应不是ZIP也不是常见图像格式。前50字节: {content_preview}")
            
            await interaction.followup.send("❌ 无法解析NovelAI API响应: 返回的既不是有效的ZIP文件也不是图像")
            return None
    except requests.exceptions.RequestException as e:
        # 网络请求异常
        error_message = f'API请求失败: {str(e)}'
        print(f"请求异常: {error_message}")
        await interaction.followup.send(f"❌ 连接NovelAI API失败: {str(e)}")
        return None
    except Exception as e:
        # 其他未预期的异常
        error_message = f'处理API请求时出错: {str(e)}'
        print(f"未预期异常: {error_message}")
        print(traceback.format_exc())
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
    await interaction.response.defer(thinking=True)
    
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
    
    # 特别说明当前已知问题
    known_issues = (
        "**当前已知问题**:\n"
        "• v4-full和v4-curated模型可能返回500内部服务器错误\n"
        "• 有时API返回的是binary/octet-stream而不是zip格式\n\n"
        "如果遇到v4模型的500错误，建议尝试使用v3模型代替。"
    )
    
    # 收集潜在的解决方案
    advice = (
        "如果您遇到NovelAI API错误，可以尝试以下解决方案:\n\n"
        "1. **检查API密钥**\n"
        "   • 确保您的密钥是从NovelAI账户中正确复制的\n"
        "   • 密钥应以'pst-'开头\n"
        "   • 尝试重新注册密钥 `/apikey [你的密钥]`\n\n"
        "2. **检查订阅**\n"
        "   • 确保您的NovelAI订阅仍然有效\n"
        "   • 确认您的订阅计划支持所选的模型\n\n"
        "3. **调整生成参数**\n"
        "   • 使用更保守的图像尺寸（如832x1216）\n"
        "   • 减少步数（如20-28步）\n"
        "   • 对于v4模型：禁用SMEA，使用karras噪声调度\n"
        "   • 对于v3模型：可以使用SMEA和任何噪声调度\n\n"
        "4. **服务状态**\n"
        "   • NovelAI服务器可能暂时过载或维护\n"
        "   • 稍后再试或尝试更换模型"
    )
    
    # 创建包含信息的嵌入
    embed = discord.Embed(
        title="NovelAI API 状态检查",
        color=0xf75c7e
    )
    
    embed.add_field(name="当前状态", value=site_status, inline=False)
    embed.add_field(name="已知问题", value=known_issues, inline=False)
    embed.add_field(name="故障排除指南", value=advice, inline=False)
    
    await interaction.followup.send(embed=embed)

# 密钥管理命令
@tree.command(name="apikey", description="注册或管理你的NovelAI API密钥")
@app_commands.describe(
    key="你的NovelAI API密钥",
    sharing="设置密钥是否在此服务器共享",
    duration_hours="密钥有效时间(小时), 0表示永不过期"
)
async def apikey_command(
    interaction: discord.Interaction, 
    key: str = None,
    sharing: Literal["私人使用", "服务器共享"] = "私人使用",
    duration_hours: int = 24
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
            
            await interaction.response.send_message(
                f"你已注册API密钥:\n"
                f"• 密钥状态: 有效\n"
                f"• 共享设置: {sharing_info}\n"
                f"• 过期时间: {expiry}", 
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
                    f"你还没有注册API密钥。请使用 `/apikey [你的密钥] [共享设置] [有效时间]` 来注册。\n\n{shared_info}\n\n你可以使用他人共享的密钥，或注册自己的密钥。",
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
        "provider_name": interaction.user.display_name  # 记录提供者名称
    }
    
    # 构建确认信息
    expiry_text = "永不过期" if expires_at is None else f"{duration_hours}小时后过期 ({expires_at.strftime('%Y-%m-%d %H:%M:%S')})"
    sharing_text = "仅限你个人使用" if not guild_id else f"在此服务器共享使用"
    
    await interaction.response.send_message(
        f"✅ API密钥已成功注册！\n"
        f"• 密钥: ||{key[:5]}...{key[-4:]}||\n"
        f"• 共享设置: {sharing_text}\n"
        f"• 有效期: {expiry_text}\n\n"
        f"你可以随时使用 `/apikey` 查看当前密钥状态，或使用新参数重新注册来更新设置。",
        ephemeral=True
    )

# 删除密钥命令
@tree.command(name="deletekey", description="删除你注册的NovelAI API密钥")
async def deletekey_command(interaction: discord.Interaction):
    user_id = str(interaction.user.id)
    
    if user_id in api_keys:
        del api_keys[user_id]
        await interaction.response.send_message("✅ 你的API密钥已从机器人中删除。", ephemeral=True)
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
    await interaction.response.send_message("✅ 你的API密钥已从此服务器共享列表中移除。", ephemeral=True)

# 基础生成命令
@tree.command(name="nai", description="使用NovelAI生成图像")
@app_commands.describe(
    prompt="图像生成提示词",
    model="模型选择 (可选)"
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
            embed.add_field(name="API密钥提供者", value=provider_info, inline=True)
            
        embed.set_image(url="attachment://generated_image.png")
        embed.set_footer(text=f"由 {interaction.user.display_name} 生成 | 使用基础设置")
        
        # 添加高级生成按钮
        view = ui.View()
        view.add_item(AdvancedButton(prompt, selected_model))
        
        await interaction.followup.send(file=file, embed=embed, view=view)
        
    except Exception as e:
        print(f"生成图像时出错: {str(e)}")
        print(traceback.format_exc())
        await interaction.followup.send(f"❌ 生成图像时出错: {str(e)}")

# 高级生成按钮
class AdvancedButton(ui.Button):
    def __init__(self, prompt, model=DEFAULT_MODEL):
        super().__init__(style=discord.ButtonStyle.primary, label="高级设置", custom_id="advanced_settings")
        self.prompt = prompt
        self.model = model
        
    async def callback(self, interaction: discord.Interaction):
        # 打开高级设置模态窗口
        await interaction.response.send_modal(AdvancedSettingsModal(self.prompt, self.model))

# 高级设置模态窗口
class AdvancedSettingsModal(ui.Modal, title="NovelAI 高级设置"):
    def __init__(self, prompt, model=DEFAULT_MODEL):
        super().__init__()
        self.prompt.default = prompt
        self.model_input.default = model

    prompt = ui.TextInput(label="提示词", style=discord.TextStyle.paragraph, required=True)
    model_input = ui.TextInput(label="模型", default=DEFAULT_MODEL, required=False)
    size_input = ui.TextInput(label="尺寸 (宽x高)", default="832x1216", required=False)
    steps_input = ui.TextInput(label="步数", default="28", required=False)
    negative_input = ui.TextInput(label="负面提示词 (选填)", style=discord.TextStyle.paragraph, required=False)
    
    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True)
        
        try:
            # 获取API密钥
            api_key, provider_info = await get_api_key(interaction)
            if not api_key:
                return
                
            # 获取用户输入的值
            prompt = self.prompt.value
            model = self.model_input.value if self.model_input.value in AVAILABLE_MODELS else DEFAULT_MODEL
            
            # 解析尺寸
            try:
                width, height = map(int, self.size_input.value.lower().split('x'))
                # 确保尺寸是64的倍数
                width = (width // 64) * 64
                height = (height // 64) * 64
            except:
                width, height = DEFAULT_SIZE
            
            # 解析步数
            try:
                steps = int(self.steps_input.value)
                steps = max(1, min(steps, 50))  # 限制步数范围
            except:
                steps = DEFAULT_STEPS
            
            # 获取负面提示词
            negative = self.negative_input.value or DEFAULT_NEG_PROMPT
            
            # 获取适合模型的参数并应用自定义设置
            model_params = get_model_default_params(model)
            model_params["width"] = width
            model_params["height"] = height
            model_params["steps"] = steps
            model_params["negative_prompt"] = negative
            
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
            
            # 创建详细嵌入消息
            embed = discord.Embed(title="NovelAI 高级生成图像", color=0xf75c7e)
            embed.add_field(name="提示词", value=prompt[:1024], inline=False)
            embed.add_field(name="模型", value=model, inline=True)
            embed.add_field(name="尺寸", value=f"{width}x{height}", inline=True)
            embed.add_field(name="步数", value=str(steps), inline=True)
            
            # 如果使用的是共享密钥，显示提供者信息
            if provider_info:
                embed.add_field(name="API密钥提供者", value=provider_info, inline=True)
                
            if negative != DEFAULT_NEG_PROMPT:
                embed.add_field(name="负面提示词", value=negative[:1024], inline=False)
            embed.set_image(url="attachment://generated_image.png")
            embed.set_footer(text=f"由 {interaction.user.display_name} 生成 | 高级设置")
            
            # 添加重新生成按钮
            view = ui.View()
            view.add_item(AdvancedButton(prompt, model))
            # 添加完整设置按钮
            view.add_item(CompleteSettingsButton(prompt, model))
            
            await interaction.followup.send(file=file, embed=embed, view=view)
            
        except Exception as e:
            print(f"高级生成出错: {str(e)}")
            print(traceback.format_exc())
            await interaction.followup.send(f"❌ 生成图像时出错: {str(e)}")

# 完整设置按钮
class CompleteSettingsButton(ui.Button):
    def __init__(self, prompt, model=DEFAULT_MODEL):
        super().__init__(style=discord.ButtonStyle.secondary, label="完整参数", custom_id="complete_settings")
        self.prompt = prompt
        self.model = model
        
    async def callback(self, interaction: discord.Interaction):
        await interaction.response.send_message("打开完整设置面板", view=CompleteSettingsView(self.prompt, self.model), ephemeral=True)

# 完整设置面板
class CompleteSettingsView(ui.View):
    def __init__(self, prompt, model=DEFAULT_MODEL):
        super().__init__()
        self.prompt = prompt
        self.model = model
        self.width, self.height = DEFAULT_SIZE
        self.steps = DEFAULT_STEPS
        self.scale = DEFAULT_SCALE
        self.sampler = DEFAULT_SAMPLER
        
        # 根据模型类型设置参数
        self.noise_schedule = "karras" if model.startswith("nai-diffusion-4") else DEFAULT_NOISE_SCHEDULE
        self.negative_prompt = DEFAULT_NEG_PROMPT
        self.cfg_rescale = DEFAULT_CFG_RESCALE
        
        # 根据模型类型设置SMEA选项
        self.sm = False if model.startswith("nai-diffusion-4") else True
        self.sm_dyn = False if model.startswith("nai-diffusion-4") else True
        
        # 添加各种选择器
        self.add_item(ModelSelectMenu(model))
        self.add_item(SizeSelectMenu(f"{self.width}x{self.height}"))
        self.add_item(SamplerSelectMenu(self.sampler))
        # 注意v4模型使用不同的噪声调度选项列表
        if model.startswith("nai-diffusion-4"):
            self.add_item(NoiseScheduleSelectMenu(self.noise_schedule, is_v4=True))
        else:
            self.add_item(NoiseScheduleSelectMenu(self.noise_schedule, is_v4=False))
        
    @ui.button(label="生成图像", style=discord.ButtonStyle.success, row=4)
    async def generate_button(self, interaction: discord.Interaction, button: ui.Button):
        await interaction.response.defer(thinking=True)
        
        try:
            # 获取API密钥
            api_key, provider_info = await get_api_key(interaction)
            if not api_key:
                return
                
            # 获取基本参数
            model_params = get_model_default_params(self.model)
            
            # 应用自定义参数
            model_params["width"] = self.width
            model_params["height"] = self.height
            model_params["scale"] = self.scale
            model_params["sampler"] = self.sampler
            model_params["steps"] = self.steps
            model_params["negative_prompt"] = self.negative_prompt
            model_params["noise_schedule"] = self.noise_schedule
            
            # 仅在v3模型中应用SMEA设置
            if not self.model.startswith("nai-diffusion-4"):
                model_params["sm"] = self.sm
                model_params["sm_dyn"] = self.sm_dyn
                
            # 准备API请求
            payload = {
                "input": self.prompt,
                "model": self.model,
                "action": "generate",
                "parameters": model_params
            }
            
            # 使用统一的API请求处理函数
            image_data = await send_novelai_request(api_key, payload, interaction)
            if image_data is None:
                return  # 如果API请求失败，直接返回
            
            # 创建文件对象并发送
            file = discord.File(io.BytesIO(image_data), filename="generated_image.png")
            
            # 创建详细嵌入消息
            embed = discord.Embed(title="NovelAI 完整参数生成", color=0xf75c7e)
            embed.add_field(name="提示词", value=self.prompt[:1024], inline=False)
            embed.add_field(name="模型", value=self.model, inline=True)
            embed.add_field(name="尺寸", value=f"{self.width}x{self.height}", inline=True)
            embed.add_field(name="步数", value=str(self.steps), inline=True)
            embed.add_field(name="采样器", value=self.sampler, inline=True)
            embed.add_field(name="噪声调度", value=self.noise_schedule, inline=True)
            embed.add_field(name="CFG比例", value=str(self.scale), inline=True)
            
            # 如果使用的是共享密钥，显示提供者信息
            if provider_info:
                embed.add_field(name="API密钥提供者", value=provider_info, inline=True)
                
            # 如果是v3模型，显示SMEA设置
            if not self.model.startswith("nai-diffusion-4"):
                embed.add_field(name="SMEA", value="启用" if self.sm else "禁用", inline=True)
                embed.add_field(name="动态SMEA", value="启用" if self.sm_dyn else "禁用", inline=True)
                
            if self.negative_prompt != DEFAULT_NEG_PROMPT:
                embed.add_field(name="负面提示词", value=self.negative_prompt[:1024], inline=False)
            embed.set_image(url="attachment://generated_image.png")
            embed.set_footer(text=f"由 {interaction.user.display_name} 生成 | 完整参数设置")
            
            # 添加重新生成按钮
            view = ui.View()
            view.add_item(AdvancedButton(self.prompt, self.model))
            view.add_item(CompleteSettingsButton(self.prompt, self.model))
            
            await interaction.followup.send(file=file, embed=embed, view=view)
            
        except Exception as e:
            print(f"完整参数生成出错: {str(e)}")
            print(traceback.format_exc())
            await interaction.followup.send(f"❌ 生成图像时出错: {str(e)}")
    
    @ui.button(label="设置步数", style=discord.ButtonStyle.secondary, row=4)
    async def steps_button(self, interaction: discord.Interaction, button: ui.Button):
        await interaction.response.send_modal(StepsModal(self))
        
    @ui.button(label="设置CFG比例", style=discord.ButtonStyle.secondary, row=4)
    async def scale_button(self, interaction: discord.Interaction, button: ui.Button):
        await interaction.response.send_modal(ScaleModal(self))
        
    @ui.button(label="设置负面提示词", style=discord.ButtonStyle.secondary, row=5)
    async def negative_button(self, interaction: discord.Interaction, button: ui.Button):
        await interaction.response.send_modal(NegativePromptModal(self))
        
    @ui.button(label="SMEA设置", style=discord.ButtonStyle.secondary, row=5)
    async def smea_button(self, interaction: discord.Interaction, button: ui.Button):
        # 检查是否为v4模型
        if self.model.startswith("nai-diffusion-4"):
            await interaction.response.send_message(
                "⚠️ v4模型不支持SMEA和动态SMEA功能，这些选项已自动禁用。",
                ephemeral=True
            )
            return
            
        # 如果是v3模型，则切换SMEA设置
        self.sm = not self.sm
        self.sm_dyn = not self.sm_dyn
        
        await interaction.response.send_message(
            f"SMEA设置已更新:\n- SMEA: {'启用' if self.sm else '禁用'}\n- 动态SMEA: {'启用' if self.sm_dyn else '禁用'}",
            ephemeral=True
        )

# 步数设置模态窗口
class StepsModal(ui.Modal, title="设置采样步数"):
    def __init__(self, parent_view):
        super().__init__()
        self.parent_view = parent_view
        self.steps_input.default = str(parent_view.steps)
        
    steps_input = ui.TextInput(label="步数 (1-50)", required=True)
    
    async def on_submit(self, interaction: discord.Interaction):
        try:
            steps = int(self.steps_input.value)
            steps = max(1, min(steps, 50))  # 限制范围
            self.parent_view.steps = steps
            await interaction.response.send_message(f"步数已设置为: {steps}", ephemeral=True)
        except ValueError:
            await interaction.response.send_message("请输入有效的数字", ephemeral=True)

# CFG比例设置模态窗口
class ScaleModal(ui.Modal, title="设置CFG比例"):
    def __init__(self, parent_view):
        super().__init__()
        self.parent_view = parent_view
        self.scale_input.default = str(parent_view.scale)
        
    scale_input = ui.TextInput(label="CFG比例 (1-10)", required=True)
    
    async def on_submit(self, interaction: discord.Interaction):
        try:
            scale = float(self.scale_input.value)
            scale = max(1.0, min(scale, 10.0))  # 限制范围
            self.parent_view.scale = scale
            await interaction.response.send_message(f"CFG比例已设置为: {scale}", ephemeral=True)
        except ValueError:
            await interaction.response.send_message("请输入有效的数字", ephemeral=True)

# 负面提示词设置模态窗口
class NegativePromptModal(ui.Modal, title="设置负面提示词"):
    def __init__(self, parent_view):
        super().__init__()
        self.parent_view = parent_view
        self.negative_input.default = parent_view.negative_prompt
        
    negative_input = ui.TextInput(label="负面提示词", style=discord.TextStyle.paragraph, required=False)
    
    async def on_submit(self, interaction: discord.Interaction):
        self.parent_view.negative_prompt = self.negative_input.value or DEFAULT_NEG_PROMPT
        await interaction.response.send_message("负面提示词已更新", ephemeral=True)

# 模型选择菜单
class ModelSelectMenu(ui.Select):
    def __init__(self, default_model=DEFAULT_MODEL):
        options = [
            discord.SelectOption(
                label=model,
                description=get_model_description(model),
                default=(model == default_model)
            )
            for model in AVAILABLE_MODELS
        ]
        super().__init__(placeholder="选择模型", options=options)
    
    async def callback(self, interaction: discord.Interaction):
        view = self.view
        view.model = self.values[0]
        
        # 如果选择了v4模型，自动禁用SMEA并设置噪声调度为karras
        if view.model.startswith("nai-diffusion-4"):
            view.sm = False
            view.sm_dyn = False
            if view.noise_schedule == "native":
                view.noise_schedule = "karras"
            
            # 更新噪声调度菜单
            # 首先移除当前的噪声调度菜单
            for item in list(view.children):
                if isinstance(item, NoiseScheduleSelectMenu):
                    view.remove_item(item)
            
            # 添加新的v4特定噪声调度菜单
            view.add_item(NoiseScheduleSelectMenu(view.noise_schedule, is_v4=True))
            
            await interaction.response.send_message(
                f"模型已设置为: {self.values[0]}\n{get_model_full_description(self.values[0])}\n\n⚠️ 注意: v4模型不支持SMEA和native噪声调度，相关设置已自动调整。\n\n⚠️ 注意: v4模型当前可能存在服务器问题，如果遇到500错误，请尝试使用v3模型。",
                ephemeral=True
            )
        else:
            # 更新噪声调度菜单为v3版本
            for item in list(view.children):
                if isinstance(item, NoiseScheduleSelectMenu):
                    view.remove_item(item)
            
            # 添加v3支持的完整噪声调度选项
            view.add_item(NoiseScheduleSelectMenu(view.noise_schedule, is_v4=False))
            
            await interaction.response.send_message(
                f"模型已设置为: {self.values[0]}\n{get_model_full_description(self.values[0])}",
                ephemeral=True
            )

# 尺寸选择菜单
class SizeSelectMenu(ui.Select):
    def __init__(self, default_size="832x1216"):
        options = [
            discord.SelectOption(
                label=size,
                default=(size == default_size)
            )
            for size in AVAILABLE_SIZES
        ]
        super().__init__(placeholder="选择图像尺寸", options=options)
    
    async def callback(self, interaction: discord.Interaction):
        view = self.view
        width, height = map(int, self.values[0].split("x"))
        view.width = width
        view.height = height
        await interaction.response.send_message(f"图像尺寸已设置为: {self.values[0]}", ephemeral=True)

# 采样器选择菜单
class SamplerSelectMenu(ui.Select):
    def __init__(self, default_sampler=DEFAULT_SAMPLER):
        options = [
            discord.SelectOption(
                label=sampler,
                description=get_sampler_description(sampler),
                default=(sampler == default_sampler)
            )
            for sampler in AVAILABLE_SAMPLERS
        ]
        super().__init__(placeholder="选择采样器", options=options)
    
    async def callback(self, interaction: discord.Interaction):
        view = self.view
        view.sampler = self.values[0]
        await interaction.response.send_message(
            f"采样器已设置为: {self.values[0]}\n{get_sampler_full_description(self.values[0])}",
            ephemeral=True
        )

# 噪声调度选择菜单
class NoiseScheduleSelectMenu(ui.Select):
    def __init__(self, default_schedule=DEFAULT_NOISE_SCHEDULE, is_v4=False):
        # 根据是否为v4模型选择可用的噪声调度
        schedules = AVAILABLE_NOISE_SCHEDULES_V4 if is_v4 else AVAILABLE_NOISE_SCHEDULES
        
        options = [
            discord.SelectOption(
                label=schedule,
                description=get_noise_schedule_description(schedule, is_v4),
                default=(schedule == default_schedule)
            )
            for schedule in schedules
        ]
        super().__init__(placeholder="选择噪声调度", options=options)
        self.is_v4 = is_v4
    
    async def callback(self, interaction: discord.Interaction):
        view = self.view
        selected_schedule = self.values[0]
        
        # 如果选择了native但模型是v4，给出警告并设置为karras
        if selected_schedule == "native" and view.model.startswith("nai-diffusion-4"):
            view.noise_schedule = "karras"
            await interaction.response.send_message(
                "⚠️ v4模型不支持native噪声调度。已自动切换为karras噪声调度。",
                ephemeral=True
            )
        else:
            view.noise_schedule = selected_schedule
            await interaction.response.send_message(f"噪声调度已设置为: {selected_schedule}", ephemeral=True)

# 帮助命令
@tree.command(name="help", description="显示帮助信息")
async def help_command(interaction: discord.Interaction):
    embed = discord.Embed(
        title="NovelAI 机器人帮助", 
        description="这个机器人使用NovelAI API生成图像。以下是可用的命令：", 
        color=0xf75c7e
    )
    
    embed.add_field(
        name="/apikey [key] [sharing] [duration_hours]",
        value=(
            "注册你的NovelAI API密钥。\n"
            "- `key`: 你的API密钥\n"
            "- `sharing`: 设置为「服务器共享」或「私人使用」\n"
            "- `duration_hours`: 密钥有效期(小时)，0表示永不过期"
        ),
        inline=False
    )
    
    embed.add_field(
        name="/deletekey",
        value="删除你注册的API密钥",
        inline=False
    )
    
    embed.add_field(
        name="/addsharing",
        value="将你的API密钥添加到当前服务器共享列表",
        inline=False
    )
    
    embed.add_field(
        name="/removesharing",
        value="从当前服务器共享列表中移除你的API密钥",
        inline=False
    )
    
    embed.add_field(
        name="/sharedkeys",
        value="显示当前服务器中共享的API密钥信息",
        inline=False
    )
    
    embed.add_field(
        name="/checkapi",
        value="检查NovelAI API的可用性状态并提供故障排除建议",
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
        name="高级设置按钮",
        value="点击生成图像下方的「高级设置」按钮，可以打开高级设置面板来自定义基本参数。",
        inline=False
    )
    
    embed.add_field(
        name="完整参数按钮",
        value="点击「完整参数」按钮可以访问所有可配置的参数，如采样器、噪声调度、SMEA等。",
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
    
    embed.add_field(
        name="已知问题",
        value=(
            "• v4模型可能返回500内部服务器错误\n"
            "• v3模型可能返回binary/octet-stream而不是zip格式\n"
            "建议首选v3模型，更加稳定"
        ),
        inline=False
    )
    
    await interaction.response.send_message(embed=embed)

# 辅助函数
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
            
        return user_key["key"], None  # 使用自己的密钥，不显示提供者信息
    
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
            # 选择第一个可用的共享密钥
            shared_user_id, key_data = shared_keys[0]
            provider_name = key_data.get("provider_name", "未知用户")
            
            return key_data["key"], provider_name  # 使用共享密钥，显示提供者信息
    
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

def get_model_description(model):
    descriptions = {
        "nai-diffusion-4-full": "最新完整模型 (⚠️可能不稳定)",
        "nai-diffusion-4-curated": "V4精选版 (⚠️可能不稳定)",
        "nai-diffusion-3": "V3模型 (推荐，更稳定)",
        "nai-diffusion-3-furry": "毛绒特定模型 (稳定)"
    }
    return descriptions.get(model, "")

def get_model_full_description(model):
    descriptions = {
        "nai-diffusion-4-full": "最新的NovelAI完整模型，拥有最全面的训练数据和最佳的生成质量。适合生成各种风格的图像。⚠️ 注意：目前可能存在稳定性问题。",
        "nai-diffusion-4-curated": "V4的精选版本，使用了经过筛选的数据集。可能在某些特定风格上有更好的表现。⚠️ 注意：目前可能存在稳定性问题。",
        "nai-diffusion-3": "稳定的V3模型，速度稍快且API兼容性更好。推荐优先使用此模型。",
        "nai-diffusion-3-furry": "专为生成毛绒/兽人角色优化的模型。在这类内容上有最佳表现。API兼容性良好。"
    }
    return descriptions.get(model, "无描述可用")

def get_sampler_description(sampler):
    descriptions = {
        "k_euler": "简单快速",
        "k_euler_ancestral": "默认推荐",
        "k_dpmpp_2s_ancestral": "高质量",
        "k_dpmpp_2m_sde": "细节丰富",
        "k_dpmpp_sde": "高级细节",
        "k_dpmpp_2m": "良好平衡"
    }
    return descriptions.get(sampler, "")

def get_sampler_full_description(sampler):
    descriptions = {
        "k_euler": "最简单和最快的采样器。生成速度快但质量可能不如其他选项。",
        "k_euler_ancestral": "NovelAI默认推荐的采样器。在速度和质量之间取得了良好的平衡。",
        "k_dpmpp_2s_ancestral": "DPM++ 2S Ancestral - 提供高质量结果，适合大多数生成场景。",
        "k_dpmpp_2m_sde": "DPM++ 2M SDE - 生成丰富细节的图像，但可能需要更多步数才能发挥最佳效果。",
        "k_dpmpp_sde": "DPM++ SDE - 高级采样器，擅长复杂细节，但速度较慢。",
        "k_dpmpp_2m": "DPM++ 2M - 在质量和速度上取得良好平衡的采样器。"
    }
    return descriptions.get(sampler, "无描述可用")

def get_noise_schedule_description(schedule, is_v4=False):
    if is_v4 and schedule == "native":
        return "原生 (不适用于v4模型)"
        
    descriptions = {
        "native": "原生 (仅v3模型)",
        "karras": "Karras (推荐)",
        "exponential": "指数型",
        "polyexponential": "多项式指数型"
    }
    return descriptions.get(schedule, "")

# 主函数
if __name__ == "__main__":
    # 从环境变量获取令牌
    TOKEN = os.getenv("DISCORD_TOKEN")
    if not TOKEN:
        print("错误: 未设置DISCORD_TOKEN环境变量")
        exit(1)
    
    # 在一个新线程中启动Flask网页服务器
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True  # 设置为守护线程，这样当主程序退出时，这个线程也会退出
    flask_thread.start()
    
    # 运行Discord机器人
    client.run(TOKEN)