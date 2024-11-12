import discord
import moviepy.editor as mp
import speech_recognition as sr
import os
from discord.ext import commands
import random

# Load the bot token from environment variable
TOKEN = os.getenv("QB_TOKEN")

# Check if the token is not found
if TOKEN is None:
    raise ValueError("The QB_TOKEN environment variable is not set")

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')

@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return
    
    if message.attachments:
        for attachment in message.attachments:
            if attachment.filename.endswith(('.mp4', '.mov', '.avi')):  # Check if the attachment is a video file
                await message.channel.send('Processing your video...')

                # Download the video
                video_path = f'./downloads/{random.randint(100000000, 999999999)}-{attachment.filename}'
                await attachment.save(video_path)

                # Extract audio from the video
                audio_path = f'./tempfiles/audio-{random.randint(100000000, 999999999)}.wav'
                try:
                    with mp.VideoFileClip(video_path) as video_clip:
                        audio_clip = video_clip.audio
                        audio_clip.write_audiofile(audio_path)

                    # Use speech recognition to transcribe audio
                    recognizer = sr.Recognizer()
                    try:
                        with sr.AudioFile(audio_path) as source:
                            audio = recognizer.record(source)
                            transcription = recognizer.recognize_google(audio)
                            await message.channel.send(f'Transcription:\n{transcription}')
                    except Exception as e:
                        await message.channel.send(f'Error transcribing audio: {e}')
                except Exception as e:
                    await message.channel.send(f'Error processing video: {e}')
                finally:
                    # Clean up the files
                    if os.path.exists(video_path):
                        os.remove(video_path)
                    if os.path.exists(audio_path):
                        os.remove(audio_path)

    await bot.process_commands(message)

bot.run(TOKEN)
