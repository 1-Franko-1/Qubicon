import discord
import os
from discord.ext import commands

# Define your bot prefix and token
TOKEN = os.getenv("QB_TOKEN")
CHANNEL_ID = 1296220677682233375  # Replace with the ID of the specific channel you want to monitor

intents = discord.Intents.default()
intents.message_content = True  # Allows bot to access message content (for edited messages)

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}!')

@bot.event
async def on_message_edit(before, after):
    # Check if the message was edited in the specified channel
    if before.channel.id == CHANNEL_ID:
        # Respond to the edited message with "toodles"
        await after.channel.send("toodles", reference=after)

bot.run(TOKEN)