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
from typing import Dict, Optional, List, Union, Literal
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
# 结构: {user_id: {"key": api_key, "shared_guilds": [guild_ids], "expires_at": datetime}}
api_keys = {}

# 默认参数
DEFAULT_MODEL = "nai-diffusion-4-full"
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
    "polyexponential"
]

@client.event
async def on_ready():
    print(f'机器人已登录为 {client.user}')
    await tree.sync()  # 同步斜杠命令
    
    # 启动密钥过期检查任务
    client.loop.create_task(check_expired_keys())

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
            if "expires_at" in user_key and user_key["expires_at"] < datetime.datetime.now():
                await interaction.response.send_message("你的API密钥已过期，请重新注册。", ephemeral=True)
                del api_keys[user_id]
                return
            
            # 构建密钥信息
            expiry = "永不过期" if "expires_at" not in user_key else f"{user_key['expires_at'].strftime('%Y-%m-%d %H:%M:%S')}"
            sharing_info = "私人使用" if not user_key.get("shared_guilds") else f"共享的服务器: {len(user_key['shared_guilds'])}个"
            
            await interaction.response.send_message(
                f"你已注册API密钥:\n"
                f"• 密钥状态: 有效\n"
                f"• 共享设置: {sharing_info}\n"
                f"• 过期时间: {expiry}", 
                ephemeral=True
            )
        else:
            await interaction.response.send_message(
                "你还没有注册API密钥。请使用 `/apikey [你的密钥] [共享设置] [有效时间]` 来注册。",
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
        "expires_at": expires_at
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
        api_key = await get_api_key(interaction)
        if not api_key:
            return
        
        # 验证并设置模型
        selected_model = model if model in AVAILABLE_MODELS else DEFAULT_MODEL
        
        # 使用所有默认参数生成图像
        width, height = DEFAULT_SIZE
        
        # 准备API请求
        payload = {
            "input": prompt,
            "model": selected_model,
            "action": "generate",
            "parameters": {
                "width": width,
                "height": height,
                "scale": DEFAULT_SCALE,
                "sampler": DEFAULT_SAMPLER,
                "steps": DEFAULT_STEPS,
                "n_samples": 1,
                "ucPreset": 0,
                "qualityToggle": True,
                "sm": True,
                "sm_dyn": True,
                "negative_prompt": DEFAULT_NEG_PROMPT,
                "noise_schedule": DEFAULT_NOISE_SCHEDULE,
                "cfg_rescale": DEFAULT_CFG_RESCALE,
            }
        }
        
        # 发送API请求
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Referer": "https://novelai.net",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0"
        }
        
        response_data = await client.loop.run_in_executor(None, lambda: requests.post(
            NAI_API_URL,
            headers=headers,
            json=payload
        ).content)
        
        # 处理ZIP响应
        with zipfile.ZipFile(io.BytesIO(response_data)) as zip_file:
            image_data = zip_file.read("image_0.png")
        
        # 创建文件对象并发送
        file = discord.File(io.BytesIO(image_data), filename="generated_image.png")
        
        # 创建基本嵌入消息
        embed = discord.Embed(title="NovelAI 生成图像", color=0xf75c7e)
        embed.add_field(name="提示词", value=prompt[:1024], inline=False)
        embed.add_field(name="模型", value=selected_model, inline=True)
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
            api_key = await get_api_key(interaction)
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
            
            # 准备API请求
            payload = {
                "input": prompt,
                "model": model,
                "action": "generate",
                "parameters": {
                    "width": width,
                    "height": height,
                    "scale": DEFAULT_SCALE,
                    "sampler": DEFAULT_SAMPLER,
                    "steps": steps,
                    "n_samples": 1,
                    "ucPreset": 0,
                    "qualityToggle": True,
                    "sm": True,
                    "sm_dyn": True,
                    "negative_prompt": negative,
                    "noise_schedule": DEFAULT_NOISE_SCHEDULE,
                    "cfg_rescale": DEFAULT_CFG_RESCALE,
                }
            }
            
            # 发送API请求
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Referer": "https://novelai.net",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0"
            }
            
            response_data = await client.loop.run_in_executor(None, lambda: requests.post(
                NAI_API_URL,
                headers=headers,
                json=payload
            ).content)
            
            # 处理ZIP响应
            with zipfile.ZipFile(io.BytesIO(response_data)) as zip_file:
                image_data = zip_file.read("image_0.png")
            
            # 创建文件对象并发送
            file = discord.File(io.BytesIO(image_data), filename="generated_image.png")
            
            # 创建详细嵌入消息
            embed = discord.Embed(title="NovelAI 高级生成图像", color=0xf75c7e)
            embed.add_field(name="提示词", value=prompt[:1024], inline=False)
            embed.add_field(name="模型", value=model, inline=True)
            embed.add_field(name="尺寸", value=f"{width}x{height}", inline=True)
            embed.add_field(name="步数", value=str(steps), inline=True)
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
        self.noise_schedule = DEFAULT_NOISE_SCHEDULE
        self.negative_prompt = DEFAULT_NEG_PROMPT
        self.cfg_rescale = DEFAULT_CFG_RESCALE
        self.sm = True
        self.sm_dyn = True
        
        # 添加各种选择器
        self.add_item(ModelSelectMenu(model))
        self.add_item(SizeSelectMenu(f"{self.width}x{self.height}"))
        self.add_item(SamplerSelectMenu(self.sampler))
        self.add_item(NoiseScheduleSelectMenu(self.noise_schedule))
        
    @ui.button(label="生成图像", style=discord.ButtonStyle.success, row=4)
    async def generate_button(self, interaction: discord.Interaction, button: ui.Button):
        await interaction.response.defer(thinking=True)
        
        try:
            # 获取API密钥
            api_key = await get_api_key(interaction)
            if not api_key:
                return
                
            # 准备API请求
            payload = {
                "input": self.prompt,
                "model": self.model,
                "action": "generate",
                "parameters": {
                    "width": self.width,
                    "height": self.height,
                    "scale": self.scale,
                    "sampler": self.sampler,
                    "steps": self.steps,
                    "n_samples": 1,
                    "ucPreset": 0,
                    "qualityToggle": True,
                    "sm": self.sm,
                    "sm_dyn": self.sm_dyn,
                    "negative_prompt": self.negative_prompt,
                    "noise_schedule": self.noise_schedule,
                    "cfg_rescale": self.cfg_rescale,
                }
            }
            
            # 发送API请求
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Referer": "https://novelai.net",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0"
            }
            
            response_data = await client.loop.run_in_executor(None, lambda: requests.post(
                NAI_API_URL,
                headers=headers,
                json=payload
            ).content)
            
            # 处理ZIP响应
            with zipfile.ZipFile(io.BytesIO(response_data)) as zip_file:
                image_data = zip_file.read("image_0.png")
            
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
        # 切换SMEA设置
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
    def __init__(self, default_schedule=DEFAULT_NOISE_SCHEDULE):
        options = [
            discord.SelectOption(
                label=schedule,
                description=get_noise_schedule_description(schedule),
                default=(schedule == default_schedule)
            )
            for schedule in AVAILABLE_NOISE_SCHEDULES
        ]
        super().__init__(placeholder="选择噪声调度", options=options)
    
    async def callback(self, interaction: discord.Interaction):
        view = self.view
        view.noise_schedule = self.values[0]
        await interaction.response.send_message(f"噪声调度已设置为: {self.values[0]}", ephemeral=True)

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
        name="/help",
        value="显示此帮助信息。",
        inline=False
    )
    
    await interaction.response.send_message(embed=embed)

# 辅助函数
async def get_api_key(interaction: discord.Interaction) -> Optional[str]:
    """获取用户的API密钥，或请求他们注册一个"""
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
            return None
        
        # 检查是否可以在此服务器使用
        if guild_id and not user_key.get("shared_guilds"):
            await interaction.followup.send(
                "⚠️ 你的API密钥设置为私人使用，但这是一个服务器频道。请使用 `/apikey` 更新设置允许在此服务器共享使用。", 
                ephemeral=True
            )
            return None
            
        if guild_id and guild_id not in user_key.get("shared_guilds", []):
            await interaction.followup.send(
                "⚠️ 你的API密钥未设置为在此服务器共享。请使用 `/apikey` 更新设置允许在此服务器共享使用。", 
                ephemeral=True
            )
            return None
            
        return user_key["key"]
    
    # 用户没有注册密钥
    await interaction.followup.send(
        "⚠️ 你需要先注册你的NovelAI API密钥才能使用此功能。请使用 `/apikey [你的密钥]` 命令注册。", 
        ephemeral=True
    )
    return None

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
        "nai-diffusion-4-full": "最新完整模型",
        "nai-diffusion-4-curated": "V4 精选版",
        "nai-diffusion-3": "上一代模型",
        "nai-diffusion-3-furry": "毛绒特定模型"
    }
    return descriptions.get(model, "")

def get_model_full_description(model):
    descriptions = {
        "nai-diffusion-4-full": "最新的NovelAI完整模型，拥有最全面的训练数据和最佳的生成质量。适合生成各种风格的图像。",
        "nai-diffusion-4-curated": "V4的精选版本，使用了经过筛选的数据集。可能在某些特定风格上有更好的表现。",
        "nai-diffusion-3": "上一代模型，速度稍快但质量不如V4。如果你喜欢老版本的风格可以使用它。",
        "nai-diffusion-3-furry": "专为生成毛绒/兽人角色优化的模型。在这类内容上有最佳表现。"
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

def get_noise_schedule_description(schedule):
    descriptions = {
        "native": "原生",
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