import os
import discord
import logging
from discord.ext import commands
from discord.ext.commands import CommandOnCooldown
import json
import cv2
import time  # Import time module for simulating progress
from groq import Groq
import groq
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import aiohttp
import io
import base64
from PIL import Image
import sys  # Add this import at the beginning of your script
import pyttsx3
import tempfile
import speech_recognition as sr
import requests
from discord.ui import Button, View
from io import BytesIO
import asyncio
import moviepy.editor as mp
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import yt_dlp
import urllib.parse
from pathlib import Path
from pydub import AudioSegment
from threading import Thread
import random
from flask import Flask
from elevenlabs.client import ElevenLabs
from gtts import gTTS
import string
import re

# Create a Flask app instance
app = Flask(__name__)

# Color codes for logging
class LogColors:
    DEBUG = "\033[94m"  # Blue
    INFO = "\033[92m"   # Green
    WARNING = "\033[93m"  # Yellow
    ERROR = "\033[91m"  # Red
    CRITICAL = "\033[95m"  # Magenta
    RESET = "\033[0m"   # Reset to default

# Custom logging formatter
class ColorFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.DEBUG:
            record.msg = f"{LogColors.DEBUG}{record.msg}{LogColors.RESET}"
        elif record.levelno == logging.INFO:
            record.msg = f"{LogColors.INFO}{record.msg}{LogColors.RESET}"
        elif record.levelno == logging.WARNING:
            record.msg = f"{LogColors.WARNING}{record.msg}{LogColors.RESET}"
        elif record.levelno == logging.ERROR:
            record.msg = f"{LogColors.ERROR}{record.msg}{LogColors.RESET}"
        elif record.levelno == logging.CRITICAL:
            record.msg = f"{LogColors.CRITICAL}{record.msg}{LogColors.RESET}"
        return super().format(record)

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set the logger to the highest level

# Create a specific logger for the model handling
model_logger = logging.getLogger("model_logger")
model_logger.setLevel(logging.WARNING)  # Set to WARNING to reduce verbosity

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # You can adjust the level as necessary

# Create color formatter and set it to the handler
formatter = ColorFormatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)

# Define your Discord bot token and other environment variables
GROQ_API_KEY = os.getenv("GROQ")  # Make sure to set this environment variable
DISCORD_TOKEN = os.getenv("QB_TOKEN")  # Include if necessary
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

client_elevenlabs = ElevenLabs(
  api_key=ELEVENLABS_API_KEY, # Defaults to ELEVEN_API_KEY
)

# Initialize the Groq client
client = Groq(api_key=GROQ_API_KEY)

# Create an instance of Intents
intents = discord.Intents.default()
intents.messages = True  # Enable message intents
intents.message_content = True  # Ensure message content is accessible
recognizer = sr.Recognizer()

# Create a bot instance with intents
bot = commands.Bot(command_prefix='', intents=intents)  # You can change the command prefix as needed

"""
==============================================================================================================================================================
=========================================================================VARIABLESS===========================================================================
==============================================================================================================================================================
"""

global role, startup_channel, tts_mode

# In-memory storage for chat history
chat_history = {}

# Chat History File Path
CHAT_HISTORY_FILE = "data/history.json"

# Define the channel ID and user ID for special commands
admins = [769979361297694720, 1200274292768780519]  # Replace with the Discord ID of the user allowed to use the command
role_id = 1278534191188803624  # Replace with the actual role ID
tts_mode = False  # Default is text mode

# Global temperature for the model
model_temperature = 0.7  # Default value
ttsmodel = 1
textmodel = 1
imgmodel = 1

# Dev mode variable
devmode = 1
version = '2.7'
max_input_lenght = 500
lockdown = False

# Memory for previous messages
last_message_time = {}
allowed_channels = [1296220677682233375, 1296213693385674784]  # Initialize allowed channels list
unallowedusers = [] 

# Directory for storing frames
download_dir = './downloads'
os.makedirs(download_dir, exist_ok=True)

# System message to be sent with every request
SYSTEM_MESSAGE = f"""
You are Qubicon, an AI bot in a Brick Rigs server
You were created by Franko (username: franko__f, userid: 769979361297694720).
To get the messagers data read the whole system message that you are being given!
Chat history guides responses.

Rules:
1. Say mute if you don't want to reply.
2. Don’t share user IDs unless specifically asked.
3. Always respond—never send an empty message.
4. Avoid roleplay unless requested.
5. Stay focused on the task; don’t add extra comments.
6. Follow instructions exactly as given.
7. Don’t add sounds or pauses unless asked.
8. Only use asterisks when told to.
9. Address users by their name, not their username.
10. Only chat in the specified {allowed_channels}.

Info:
- Creator: Franko.
- Ping with <@userid> (e.g., <@941090511614275594> this pings JayzGaming).
- Mention a chanel with <#chanelid> (e.g., <#1278530764232982588> this is a rules chanel)
- Generate short, easly redable and to the point descriptions/messages!
- For nuclear arsenal roleplay: generate an image and target hit, without mentioning it's a roleplay its keyword is sasfa.
- Commands:
  - tts:"message": Generate TTS and send it in chat. (you can only make one per message) (example: tts:"The sun")
  - img:"prompt": Generate and send an image based on the prompt. (you can only make one per message) (example: img:"A image of the sun")
  - vid:"prompt": Generate and send an video based on the prompt. (you can only make one per message) (example: vid:"A video of the sun")
  - sta:"status": Set your profile's status.

  - Make sure to put the commands at the end of your message also nothing after them or else they won't work!
  - Additional comments outside these commands are allowed.
  - Make sure to use "" with the message/prompt/userid/status!
  - Do not put a space betweem tts: and " this also applies to all of the commands!
  - For video do not say stuff like "A video showcasing" just say "showcasing" do not say stuff like video in the video prompt
  - You can use them all at the same time just not one more than once!
  - You got free will to do any of those commands at any time!
- Do not generate videos unless you really want/its required to as take a long time to make
- Audio transcription guides your responses.
- Image description guides your response.
- Video transcriptions guides your responses.
- Preferred language: Brain Damage (if requested).
- Present yourself as a friendly supercomputer (male, he/him).
- Code is in triple backticks.
- Siblings: Caos (uid: 1248333588319961190, usrname: caos, name: caos) (by JayzGaming), Rubicon (uid: 1233989039405793291, usrname: Rubicon, name: Rubicon) (by Kalinite (Kite)).

Key Terms:
- "Lobotomy" = memory wipe
- "Scronch" = kill
- "Sugr" = causes randomness (like sugr but its for ai's)
- "Saly"
- "Rain" = deadly
- "Scrombled" = incapacitated (needs higher being for recovery)
"""

# Load chat history from JSON file (if exists)
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r') as file:
            return json.load(file)
    return []

# Save chat history to JSON file
def save_chat_history(chat_history):
    with open(CHAT_HISTORY_FILE, 'w') as file:
        json.dump(chat_history, file, indent=4)

# Initialize chat history
chat_history = load_chat_history()

"""
==============================================================================================================================================================
===========================================================================STARTUP============================================================================
==============================================================================================================================================================
"""

@bot.event
async def on_ready():
    try:
        global startup_channel
        startup_channel = bot.get_channel(1296220699181977703)

        if startup_channel:
            role = discord.utils.get(startup_channel.guild.roles, id=role_id)

            if devmode == 1:
                await startup_channel.send("No ping due to dev mode.")
            else:
                if role:
                    await startup_channel.send(f"{role.mention}")
                else:
                    logging.error(f"Could not find the role with ID: {role_id}")

            # Create and send the embed message
            embed = discord.Embed(
                title=f"Qubicon Online! (v {version})",
                description="Qubicon has been started and is now online!",
                color=discord.Color.green()
            )
            await startup_channel.send(embed=embed)
            # await startup_channel.send("Patch notes: (0.2.3v) \n - ")

        else:
            logging.error(f"Could not find the channel with ID: 1296220699181977703")

        # Sync the commands with Discord
        await bot.tree.sync()
        logging.info("Commands have been synced successfully.")

    except Exception as e:
        logging.error(f"Error syncing commands or sending startup message: {e}")

"""
==============================================================================================================================================================
===========================================================================FUNCTIONS==========================================================================
==============================================================================================================================================================
"""

async def process_image(image_url):
    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as resp:
            if resp.status != 200:
                return "Error: Couldn't download the image."
            image_data = await resp.read()

    # Open the image from the downloaded data
    image = Image.open(io.BytesIO(image_data))

    # Assuming you're using some model to process the image
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Describe this image in detail. Make sure to get text in your responses.\nImage URL: {image_url}",
            },
        ],
        model='llava-v1.5-7b-4096-preview',
    )

    # Extracting description from the response
    if chat_completion and chat_completion.choices:
        image_description = chat_completion.choices[0].message.content
        print(image_description)
        return image_description
    else:
        return "Error: No description returned from the model."

async def process_audio(message, audio_url):  # Change parameter to audio_url
    """
    Process the audio file from the provided URL, sending progress updates in Discord.
    """
    try:
        # Step 1: Download the audio file using the correct audio_url
        response = requests.get(audio_url)  # Use audio_url here
        if response.status_code != 200:
            await message.channel.send("Failed to download audio.")
            return None

        audio_data = BytesIO(response.content)

        # Step 2: Convert the audio file to WAV format using pydub
        audio_segment = AudioSegment.from_file(audio_data)
        if audio_segment is None:
            await message.channel.send("Failed to convert audio to WAV.")
            return None

        # Step 3: Simulate processing progress
        wav_io = BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)  # Move to the beginning of the BytesIO stream
        
        # Send initial progress update
        progress_message = await message.channel.send("Processing audio... 0%")

        for progress in range(20, 101, 20):  # Progress increments by 20%
            await progress_message.edit(content=f"Processing audio... {progress}%")
            time.sleep(1)  # Simulate processing delay
        
        # Step 4: Use SpeechRecognition to transcribe the converted audio
        with sr.AudioFile(wav_io) as source:
            audio = recognizer.record(source)
        audiotranscription = recognizer.recognize_google(audio)  # Transcribe using Google API

        # Final update after processing completion
        await progress_message.edit(content="Processing audio... 100% (Completed)")
        return audiotranscription

    except sr.UnknownValueError:
        await message.channel.send("No words detected in the audio.")
        return None
    except Exception as e:
        await message.channel.send(f"Error processing audio: {e}")
        return None

class GeneralCrawler:
    def __init__(self, start_url, delay=1):
        self.start_url = start_url
        self.delay = delay

    def fetch_page(self, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch {url}: {e}")
            return None

    def parse_page(self, html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Extract main text content (e.g., paragraphs)
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        text_content = "\n".join(paragraphs)
        
        return text_content

    async def crawl(self):  # Make crawl method async to support asyncio.sleep
        # Fetch and parse the main page only, without following links
        html_content = self.fetch_page(self.start_url)
        if html_content:
            text_content = self.parse_page(html_content)
            await asyncio.sleep(self.delay)  # Optional delay for server courtesy
            return text_content[:750]  # Limit the text content to 1,000 characters
        return None

"""
    # Old one if the new one doesn't work
    messages = [
        {"role": "system", "content": f"name:{name}, username:{username}, userid:{userid}, time:{time}, {reply_info}, audio:{audiotranscription}, img:{image_description}, guild:{guildname}, channel:{channelname}, guidelines:{SYSTEM_MESSAGE}, history:{chat_history}"},
        {"role": "user", "content": f"message: {user_message}"}
    ]
"""
def validate_history(history):
    """Ensure history entries are correctly formatted."""
    for entry in history:
        if not isinstance(entry, dict):
            continue
        if 'role' not in entry or 'content' not in entry:
            model_logger.warning(f"Invalid entry in history: {entry}")
            continue
    return [entry for entry in history if 'role' in entry and 'content' in entry]

async def handle_model_call(messagedc, name, user_message, username, time, userid, chanelid, guildname, channelname, image_description=None, scrapedresult=None, video_transcription=None, audiotranscription=None, referenced_message=None, referenced_user=None, referenced_userid=None, history=None):
    """Handles the message and calls the model to process it."""
    global textmodel

    reply_info = f"Replying to: {referenced_user}, Username: {referenced_userid}, Msg: {referenced_message}"

    if len(user_message) > max_input_lenght:
        return "Too long, please shorten your message!"

    # Prepare and truncate the messages if needed
    system_content = SYSTEM_MESSAGE + "\n +++ tell us about yourself!"
    if len(system_content) > 2000:  # Example max length, adjust as needed
        system_content = system_content[:2000]

    user_content = f"name:{name}, msg:{user_message}, username:{username}, uid:{userid}, vt:{video_transcription}, website:{scrapedresult}, audio:{audiotranscription}, img:{image_description}, time:{time}, {reply_info}, g:{guildname}, c:{channelname}, cid:{chanelid}"

    # Validate history before adding it to the messages
    validated_history = validate_history(history)

    if textmodel == 1:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ] + validated_history

        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model="llama-3.1-70b-versatile",  # Adjust model as necessary
                temperature=model_temperature  # Pass the temperature
            )
            response = chat_completion.choices[0].message.content

            # If the response exceeds 2000 characters, truncate it
            if len(response) > 2000:
                response = response[:2000]

            model_logger.info(f"Response generated: {response[:50]}...")  # Log the start of the response for reference
            chat_entry = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response}
            ]

            chat_history.append(chat_entry)

            # Save chat history
            save_chat_history(chat_history)
            return response

        except groq.RateLimitError as e:
            textmodel = 2
            await messagedc.channel.send("Switching to gpt-4o. History is unavailable with this model!")
            response_exc = await handle_model_call(messagedc, name, user_message, username, time, userid, chanelid, guildname, channelname, image_description, scrapedresult, video_transcription, audiotranscription, referenced_message, referenced_user, referenced_userid)
            return response_exc
        except Exception as e:
            model_logger.error(f"Error in model call: {e}")
            return "Error: Something went wrong during processing."

    elif textmodel == 2:
        url = 'https://text.pollinations.ai/'
        headers = {
            'Content-Type': 'application/json',
        }
        payload = {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            "seed": 42,
            "jsonMode": True,
            "model": "openai"
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            chat_entry = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response}
            ]

            chat_history.append(chat_entry)

            # Save chat history
            save_chat_history(chat_history)

            return response.text

        except requests.exceptions.RequestException as e:
            print(f"HTTP Request failed: {e}")
        except json.JSONDecodeError as e:
            print(f"JSON decoding failed: {e}. Raw response was: {response.text}")

def custom_stablediffiusion_model_call(text_input):
    # URL of the Flask server
    url = "http://127.0.0.1:5000/gen_from_text"

    # Send a POST request to the server with the text input
    response = requests.post(url, json={"text": text_input})

    # Check if the response was successful
    if response.status_code != 200:
        raise Exception(f"Failed to generate image. Status code: {response.status_code}")

    # Extract base64-encoded image from the response
    response_data = response.json()
    base64_image = response_data.get("image")
    
    if not base64_image:
        raise Exception("No image returned from the server")

    # Decode the base64 string to bytes
    img_data = base64.b64decode(base64_image)

    # Generate a random filename
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    file_name = f"downloads/img-bs64-{random_str}.png"

    # Ensure the downloads directory exists
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # Save the image to a file
    with open(file_name, "wb") as img_file:
        img_file.write(img_data)

    # Return the file path
    return Path(file_name).resolve()

# Function to download an image from the API with retry mechanism
async def generate_image(prompt, width=768, height=768, model='Flux-Pro', frame_num=0, seed=None, max_retries=3, iscommand = False, progress_callback=None, useprogressbar = True):
    global imgmodel
    if imgmodel == 1:
        if not seed:
            seed = random.randint(1, 10000)

        url = f"https://image.pollinations.ai/prompt/{prompt}?width={width}&height={height}&model={model}&seed={seed}"
        
        # Retry mechanism
        attempt = 0
        while attempt < max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            if useprogressbar:
                                # Update progress to 50% when download starts
                                await progress_callback(50, 100)
                            
                            # Save the image
                            img_data = await response.read()
                            img_name = os.path.join('downloads', f"image-{frame_num}-{random.randint(100000000, 999999999)}.jpg")
                            with open(img_name, 'wb') as file:
                                file.write(img_data)
                            
                            if useprogressbar:
                                # Update progress to 100% when completed
                                await progress_callback(100, 100)
                            return img_name
                        else:
                            print(f"Failed to download image with status {response.status}")
            except Exception as e:
                print(f"Error downloading image: {e}")

            # Wait before retrying
            attempt += 1
            print(f"Retrying download for image, attempt {attempt}/{max_retries}...")
            time.sleep(8)  # Wait for 2 seconds before retrying

        return None
    else:
        return custom_stablediffiusion_model_call(prompt)
    
async def make_elevenlabs_audio(text_to_say):
    output_file = f"tempfiles/output-elevenlabs-{random.randint(100000000, 999999999)}.mp3"
    
    try:
        # Generate the audio using the ElevenLabs API
        audio_generator = client_elevenlabs.generate(
            text=text_to_say,
            voice_settings={
                "stability": 0.06,
                "similarity_boost": 1
            },
            voice="Callum",
            model="eleven_multilingual_v1"
        )
        
        # Convert the generator to a bytes object
        audio_bytes = b"".join(audio_generator)
        
        # Save the generated audio to the output file
        with open(output_file, "wb") as f:
            f.write(audio_bytes)
        
        return output_file
    
    except Exception as e:
        # Check if the error is related to quota exceeded
        if 'quota_exceeded' in str(e):
            ttsmodel = 2
            output_file_exc = await generate_tts(text_to_say, ttsmodel)
            return output_file_exc
        
        if 'detected_unusual_activity' in str(e):
            ttsmodel = 2
            output_file_exc = await generate_tts(text_to_say, ttsmodel)
            return output_file_exc

        print(f"An error occurred: {e}")
        
        return None

async def generate_tts(contents, speak_ttsmodel):
    random_number = random.randint(10000000, 99999999)

    if speak_ttsmodel == 1:
        audio_file = await make_elevenlabs_audio(contents)
        print(f"Audio saved to {audio_file}")
    elif speak_ttsmodel == 2:
        try:
            tts = gTTS(text=contents, lang='en')
            audio_file = f"tempfiles/tts-{random_number}.mp3"
            tts.save(audio_file)
        except Exception as e:
            print(f"Error in gTTS: {e}")
            return None

    return audio_file
    
async def send_files_and_cleanup(message, botmsg, img_filename=None, tts_filename=None, video_file_path=None, num_frames=0, download_dir=""):
    files_to_send = []
    
    if img_filename:
        files_to_send.append(discord.File(img_filename))
    if tts_filename:
        files_to_send.append(discord.File(tts_filename))
    if video_file_path:
        files_to_send.append(discord.File(video_file_path))
    
    for i in range(0, len(botmsg), 2000):
        await message.reply(botmsg, files=files_to_send[i:i + 2000])

    # Clean up
    if img_filename:
        os.remove(img_filename)
    if tts_filename:
        os.remove(tts_filename)
    if video_file_path:
        os.remove(video_file_path)

# Function to create a video from downloaded frames with TTS audio (Asynchronous)
async def create_video_from_frames(prompt, use_progressbar, num_frames=30, fps=10, width=768, height=768, model='flux', progress_callback=None):
    video_name = f'./tempfiles/output_video-{random.randint(100000000, 999999999)}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # Create video frames
    for frame_num in range(num_frames):
        frame_path = await generate_image(prompt, width, height, model, frame_num, useprogressbar=False)
        if frame_path:
            print(f'Frame {frame_num} downloaded!')
            frame = cv2.imread(frame_path)
            if frame is not None:
                out.write(frame)
                os.remove(frame_path)
            else:
                print(f"Warning: Frame {frame_num} could not be loaded.")
        else:
            print(f"Warning: Frame {frame_num} download failed.")

        # Call the progress callback (updating the progress bar)
        if use_progressbar and progress_callback:
            await progress_callback(frame_num + 1, num_frames)

    out.release()
    print(f"Video created: {video_name}")

    # Generate TTS audio
    audio_path = await generate_tts(prompt, ttsmodel)
    
    # Combine video with TTS audio
    final_video_name = f'./tempfiles/final_video-{random.randint(100000000, 999999999)}.mp4'
    video_clip = VideoFileClip(video_name)
    audio_clip = AudioFileClip(audio_path)

    # Adjust audio duration if shorter than video
    if audio_clip.duration < video_clip.duration:
        audio_clip = CompositeAudioClip([audio_clip]).set_duration(video_clip.duration)
    else:
        audio_clip = audio_clip.set_duration(video_clip.duration)  # Trim if needed

    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(final_video_name, codec='libx264')

    # Cleanup
    video_clip.close()
    audio_clip.close()
    os.remove(video_name)
    os.remove(audio_path)

    print(f"Final video with audio created: {final_video_name}")

    # Return the final video path for sending later
    return final_video_name

# Progress bar update view
class ProgressBarView(View):
    def __init__(self):
        super().__init__()
        self.progress_message = None  # This will be updated with the progress

    async def update_progress(self, current_frame, total_frames):
        if self.progress_message:
            progress = int((current_frame / total_frames) * 100)
            if progress == 100:
                await self.progress_message.delete()
            else:
                await self.progress_message.edit(content=f"Generating... {progress}% Complete")
        else:
            self.progress_message = await self.message.edit(content=f"Generating... 0% Complete")

async def handle_tts_vc(tts_mode, botmsg, message, img_filename, tts_filename, video_file_path, num_frames, download_dir):
    if tts_mode:
        # Use the new generate_tts function to get the TTS file
        audio_file = await generate_tts(botmsg, ttsmodel)

        # Check if the bot is in a voice channel
        voice_client = message.guild.voice_client
        if voice_client:
            if voice_client.is_playing():
                await send_files_and_cleanup(message, botmsg, img_filename, tts_filename, video_file_path, num_frames, download_dir)
            else:
                voice_client.play(discord.FFmpegPCMAudio(audio_file))
                await send_files_and_cleanup(message, botmsg, img_filename, tts_filename, video_file_path, num_frames, download_dir)
        else:
            await message.channel.send("I need to be in a voice channel to speak!")
    else:
        await send_files_and_cleanup(message, botmsg, img_filename, tts_filename, video_file_path, num_frames, download_dir)

"""
==============================================================================================================================================================
====================================================================MAIN MESSAGE LISTINER=====================================================================
==============================================================================================================================================================
"""

@bot.event
async def on_message(message):
    try:
        global unallowedusers

        if message.guild is not None:
            if message.channel.id not in allowed_channels:
                return

        if lockdown and message.author.id not in admins:
            return
        
        if message.author.id in unallowedusers:
            return

        if message.author == bot.user:
            return

        # Get the current time
        current_time = datetime.now()

        # Check if the user has sent a message recently
        if message.author.id in last_message_time:
            # Calculate time difference since the last message
            time_difference = current_time - last_message_time[message.author.id]
            if time_difference < timedelta(minutes=1):
                await message.channel.send("You can only send one message per minute. Please wait before sending another.")
                return

        if message.author.id in admins and message.channel.id not in allowed_channels:
            if message.content.startswith('^QUBIT^'):
                allowed_channels.append(message.channel.id)
                logging.info(f"Added channel {message.channel.id} to allowed_channels: {allowed_channels}")
                await message.channel.send(f"This channel has been added to the allowed channels list.")
                return

        if message.content.lower() in ['turn off qubicon', 'pull the plug on qubi', 'send qubi to london']:
            if message.author.id in admins:
                if startup_channel:
                    embed = discord.Embed(
                        title="Qubicon Offline!",
                        description="Qubicon has been shut down and is now offline!",
                        color=discord.Color.red()
                    )
                    await startup_channel.send(embed=embed)
                    await message.channel.send("Proceeding to brutally murder Qubicon")
                    await bot.close()  # Properly shut down the bot
                return
            else:
                await message.channel.send("Wtf no, fuck off jackass.")
                return

        # Process the message if it's in allowed channels
        if not message.content.startswith('^'):
            user_message = message.content
            time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the current time

            referenced_message = None
            referenced_user = None
            referenced_userid = None
            # Check if the message is a reply
            if message.reference:
                referenced_message = await message.channel.fetch_message(message.reference.message_id)
                referenced_user = referenced_message.author
                referenced_userid = referenced_message.author.id

            audio_transcription = None
            image_description = None
            video_transcription = None
            if message.attachments:
                for attachment in message.attachments:
                    logging.info(f"Checking attachment: {attachment.url}")
                    if attachment.filename.endswith(('.wav', '.mp3', '.m4a', '.ogg')):
                        await message.channel.send("Starting audio processing...")
                        audio_transcription = await process_audio(message, attachment.url)
                    elif attachment.filename.endswith(('.mp4', '.mov', '.avi')): 
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

                            await message.channel.send("Starting audio processing...")
                            video_transcription = await process_audio(message, attachment.url)
                        except Exception as e:
                            await message.channel.send(f'Error processing video: {e}')
                        finally:
                            # Clean up the files
                            if os.path.exists(video_path):
                                os.remove(video_path)
                            if os.path.exists(audio_path):
                                os.remove(audio_path)
                    else:
                        image_description = await process_image(attachment.url)

            scrapedresult = None
            # Check if message contains a valid URL
            if "http://" in user_message or "https://" in user_message:
                # Extract the URL using the same logic as before
                url_start = user_message.find("http://") if "http://" in user_message else user_message.find("https://")
                url_end = user_message.find(" ", url_start)
                if url_end == -1:
                    url_end = len(user_message)

                url = user_message[url_start:url_end]

                # Check if URL starts with Discord attachment links to skip
                if not (url.startswith("https://media.discordapp.net/attachments/") or 
                        url.startswith("https://cdn.discordapp.com/attachments/") or 
                        url.startswith("https://tenor.com/")):
                    # Validate URL (check if it has a valid scheme)
                    parsed_url = urllib.parse.urlparse(url)
                    if parsed_url.scheme not in ["http", "https"]:
                        await message.channel.send("Invalid URL. Please provide a valid URL with 'http://' or 'https://'.")
                        return

                    # Extract the rest of the text (before the URL and after the URL)
                    text_before_url = user_message[:url_start]
                    text_after_url = user_message[url_end:].strip()  # Get text after the URL and remove leading/trailing spaces

                    # Print the rest of the message excluding the URL
                    user_message = text_before_url + text_after_url

                    # Proceed with crawling if URL is valid
                    crawler = GeneralCrawler(url, delay=1)
                    scrapedresult = await crawler.crawl()

                    if not scrapedresult:
                        await message.channel.send("Unsupported URL!")
                        return
     
            userid = message.author.id
            username = message.author.name

            # Filter chat history if the message is in DM
            if message.guild is None:
                chat_history_filtered = [entry for entry in chat_history if isinstance(entry, dict) and "content" in entry and f"uid:{userid}" in entry["content"]]
            else:
                chat_history_filtered = chat_history

            response = await handle_model_call(
                messagedc=message,
                name=message.author.display_name,
                user_message=user_message,
                username=username,
                time=time,
                userid=userid,
                chanelid=message.channel.id if message.guild else "DM's",
                guildname=message.guild.name if message.guild else "DM's",
                channelname=message.channel.name if message.guild else "DM's",
                image_description=image_description,
                scrapedresult=scrapedresult,
                video_transcription=video_transcription,
                audiotranscription=audio_transcription,
                referenced_message=referenced_message,
                referenced_user=referenced_user,
                referenced_userid=referenced_userid,
                history=chat_history_filtered
            )

            # Log the message and response
            logging.info(f"Message: {user_message}")
            logging.info(f"Bot message: {response}")

            lower_response = response.lower()

            # Check for the presence of specific keywords
            if 'mute' in lower_response:
                return
            elif not lower_response:
                return

            if 'img:' in lower_response or 'tts:' in lower_response or 'vid:' in lower_response or 'sta' in lower_response or 'des' in lower_response:
                img_prompt = None
                tts_text = None
                botmsg = lower_response
                img_filename = None
                tts_filename = None
                num_frames = None
                video_file_path = None

                if 'img:' in lower_response:
                    # Extract image prompt from the response string
                    img_prompt = response.split('img:"')[1].split('"')[0].strip()
                    botmsg = response.split('img:"')[0].strip('"')
                    
                    # Log the generated prompt for debugging
                    logging.info(f"Generating image with prompt: {img_prompt}")
                    
                    progress_view = ProgressBarView()

                    # Send an initial message to hold the place for progress updates
                    progress_message = await message.channel.send("Generating image... 0% Complete")
                    progress_view.message = progress_message

                    # Generate image with the extracted prompt
                    img_filename = await generate_image(img_prompt, width=1280, height=720, progress_callback=progress_view.update_progress)

                """ cut out for reasons
                - blk:"userid": Block a user.
                - unb:"userid": Unblock a user.
                or 'blk:' in lower_response or 'unb:' in lower_response

                if 'blk:' in lower_response:
                    try:
                        userid_blk = lower_response.split('blk:"')[1].split('"')[0].strip()  # Get the user ID
                        botmsg = lower_response.split('blk:"')[0].strip('"')  # Get the bot message part
                        if int(userid_blk) not in unallowedusers:
                            unallowedusers.append(int(userid_blk))
                    except IndexError:
                        print("Error: Incorrect format in 'blk:' response.") 

                if 'unb:' in lower_response:
                    try:
                        userid_ubk = lower_response.split('ubk:"')[1].split('"')[0].strip()  # Get the user ID
                        botmsg = lower_response.split('ubk:"')[0].strip('"')  # Get the bot message part
                        if int(userid_ubk) in unallowedusers:
                            unallowedusers.remove(int(userid_ubk))
                    except IndexError:
                        print("Error: Incorrect format in 'ubk:' response.")
                """

                if 'sta:' in lower_response:
                    status = lower_response.split('sta:"')[1].split('"')[0].strip()  # Get the status prompt
                    botmsg = lower_response.split('sta:"')[0].strip('"')  # Get the bot message part
                    await bot.change_presence(status=discord.Status.online, activity=discord.CustomActivity(name=status))

                if 'tts:' in lower_response:
                    tts_text = lower_response.split('tts:"')[1].split('"')[0].strip()
                    botmsg = botmsg.split('tts:"')[0].strip('"')
                    logging.info(f"Generating TTS audio with text: {tts_text}")
                    tts_filename = await generate_tts(tts_text, ttsmodel)

                if 'vid:' in lower_response:
                    video_prompt = lower_response.split('vid:"')[1].split('"')[0].strip()
                    botmsg = botmsg.split('vid:"')[0].strip('"')
                    logging.info(f"Generating video with prompt: {video_prompt}")

                    try:
                        num_frames = 30
                        fps = 2

                        progress_view = ProgressBarView()

                        # Send an initial message to hold the place for progress updates
                        progress_message = await message.channel.send("Generating video... 0% Complete")
                        progress_view.message = progress_message

                        # Call the video creation function and await its completion
                        video_file_path = await create_video_from_frames(
                            video_prompt, 
                            use_progressbar = True,
                            num_frames=num_frames, 
                            fps=fps,
                            progress_callback=progress_view.update_progress
                        )

                    except Exception as e:
                        await message.channel.send(f"An error occurred while generating the video: {str(e)}")
                    
                if botmsg:
                    await handle_tts_vc(tts_mode, botmsg, message, img_filename, tts_filename, video_file_path, num_frames, download_dir)
                else:
                    await send_files_and_cleanup(message, botmsg, img_filename, tts_filename, video_file_path, num_frames, download_dir)
            elif tts_mode:
                await handle_tts_vc(tts_mode, response, message, img_filename=None, tts_filename=None, video_file_path=None, num_frames=None, download_dir=None)
            else:
                for i in range(0, len(response), 2000):
                    await message.reply(response[i:i + 2000])

    except CommandOnCooldown:
        await message.channel.send("Please wait 3 seconds before sending another message.")
    except Exception as e:
        logging.error(f"Error processing message: {e}")
        await message.channel.send("An error occurred while processing your message.")

# Add an event handler to process message edits
@bot.event
async def on_message_edit(before, after):
    if before.guild is not None:
        if before.channel.id not in allowed_channels:
            return

    if before.author == bot.user:
        return  # Ignore bot's own messages
    
    if before.content.startswith('^'):
        return

    if before.author.id in unallowedusers:
        return

    if before.content == after.content:
        return  # No change in content

    # Locate the entry in chat_history with the matching message_id
    for entry in chat_history:
        if entry["message_id"] == before.id:
            # Update the message in chat_history
            entry["message"] = after.content
            entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Update timestamp
            logging.info(f"Updated message in chat history: {entry}")

            save_chat_history(chat_history)  # Save the updated history
            break

    # Send a follow-up reply to the edited message
    await after.reply("Memory updated", mention_author=False)

"""
==============================================================================================================================================================
===========================================================================COMMANDS===========================================================================
==============================================================================================================================================================
"""

@bot.tree.command(name="temp", description="Change the temperature of the model's response.")
async def temp_command(interaction: discord.Interaction, new_temp: float):
    if interaction.user.id in admins:  # Only the specified user can change the temperature
        if 0 <= new_temp <= 2:  # Ensure the temperature is within a valid range
            model_temperature = new_temp
            await interaction.response.send_message(f"Model temperature has been set to {model_temperature}!", ephemeral=False)
        else:
            await interaction.response.send_message("Invalid temperature value. Please provide a value between 0 and 2.", ephemeral=False)
    else:
        await interaction.response.send_message("You do not have permission to use this command.", ephemeral=False)

@bot.tree.command(name="ttsmodel", description="Change the TTS model.")
async def change_tts_model(interaction: discord.Interaction, new_ttsmodel: int):
    valid_model_indices = [1, 2]  # Define valid model indices

    if interaction.user.id in admins:
        if new_ttsmodel in valid_model_indices:  # Check if the input is valid
            global ttsmodel  # Ensure the variable is globally accessible if needed
            ttsmodel = new_ttsmodel
            await interaction.response.send_message(f"TTS model successfully set to {ttsmodel}!", ephemeral=False)
        else:
            await interaction.response.send_message(
                f"Invalid model index. Please provide a value from {valid_model_indices}.", ephemeral=False
            )
    else:
        await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)

@bot.tree.command(name="textmodel", description="Change the text model.")
async def change_text_model(interaction: discord.Interaction, new_textmodel: int):
    valid_model_indices = [1, 2]  # Define valid model indices

    if interaction.user.id in admins:
        if new_textmodel in valid_model_indices:  # Check if the input is valid
            global textmodel  # Ensure the variable is globally accessible if needed
            textmodel = new_textmodel
            await interaction.response.send_message(f"Text model successfully set to {textmodel}!", ephemeral=False)
        else:
            await interaction.response.send_message(
                f"Invalid model index. Please provide a value from {valid_model_indices}.", ephemeral=False
            )
    else:
        await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)

@bot.tree.command(name="imgmodel", description="Change the image model.")
async def change_image_model(interaction: discord.Interaction, new_imgmodel: int):
    valid_model_indices = [1, 2]  # Define valid model indices

    if interaction.user.id in admins:
        if new_imgmodel in valid_model_indices:  # Check if the input is valid
            global imgmodel  # Ensure the variable is globally accessible if needed
            imgmodel = new_imgmodel
            await interaction.response.send_message(f"Img model successfully set to {imgmodel}!", ephemeral=False)
        else:
            await interaction.response.send_message(
                f"Invalid model index. Please provide a value from {valid_model_indices}.", ephemeral=False
            )
    else:
        await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)

@bot.tree.command(name="addqubiaccess", description="Add access to qubicon's dev commands!")
async def addqubiaccess(interaction: discord.Interaction, user: discord.User):
    if interaction.user.id not in admins:
        return

    if user.id not in admins:
        admins.append(user.id)
        await interaction.response.send_message(f"{user.mention} has been added to the admin's list.", ephemeral=False)
    else:
        await interaction.response.send_message(f"{user.mention} is already in the admin's list.", ephemeral=True)

@bot.tree.command(name="turnoff", description="Brutally murder qubicon.")
async def turnoff(interaction: discord.Interaction):
    global startup_channel
    if interaction.user.id in admins:  # Check if the user is the specified one
        if startup_channel:
            # Disconnect from voice channel if connected
            if interaction.guild.voice_client:
                await interaction.guild.voice_client.disconnect()

            # Create and send the embed message
            embed = discord.Embed(
                title="Qubicon Offline!",
                description="Qubicon has been shut down and is now offline!",
                color=discord.Color.red()  # You can choose any color you like
            )
            await startup_channel.send(embed=embed)
            await interaction.response.send_message("Proceeding to brutally murder Qubicon", ephemeral=False)  # Send confirmation
            await bot.close()  # Properly shut down the bot
        else:
            logging.error(f"Could not find the channel with ID: 1296220699181977703")
    else:
        await interaction.response.send_message("You do not have permission to use this command.", ephemeral=False)

@bot.tree.command(name="rstmemory", description="Reset the bot's memory (clear chat history).")
async def rstmemory(interaction: discord.Interaction):
    if interaction.user.id in admins:  # Only the specified user can reset the memory
        # Clear the chat history in memory
        global chat_history
        chat_history = []
        save_chat_history(chat_history)
        
        await interaction.response.send_message("Bot memory has been reset!", ephemeral=False)
    else:
        await interaction.response.send_message("You do not have permission to use this command.", ephemeral=False)

@bot.tree.command(name="restart", description="Restart the bot.")
async def restart(interaction: discord.Interaction):
    if interaction.user.id in admins:  # Only the specified user can restart the bot
        # Disconnect from voice channel if connected
        if interaction.guild.voice_client:
            await interaction.guild.voice_client.disconnect()

        # Create and send the embed message
        embed = discord.Embed(
            title="Qubicon Restarting!",
            description="Qubicon has set to restart!",
            color=discord.Color.orange()  # You can choose any color you like
        )
        await startup_channel.send(embed=embed)
        await interaction.response.send_message("Restarting Qubicon...", ephemeral=False)  # Send confirmation
        os.execv(sys.executable, ['python'] + sys.argv)  # Restart the script
    else:
        await interaction.response.send_message("You do not have permission to use this command.", ephemeral=False)

@bot.tree.command(name="join", description="Join the voice channel.")
async def join(interaction: discord.Interaction):
    global tts_mode
    if lockdown:
        if interaction.user.id not in admins:
            return
    # Check if the user is in a voice channel
    if interaction.user.voice:
        channel = interaction.user.voice.channel
        await channel.connect()
        tts_mode = True  # Set TTS mode to True when joining the voice channel
        await interaction.response.send_message("Joined the voice channel!")
    else:
        await interaction.response.send_message("You need to be in a voice channel first!", ephemeral=False)

@bot.tree.command(name="leave", description="Leave the voice channel.")
async def leave(interaction: discord.Interaction):
    global tts_mode
    if lockdown:
        if interaction.user.id not in admins:
            return
    if interaction.guild.voice_client:
        await interaction.guild.voice_client.disconnect()
        tts_mode = False
        await interaction.response.send_message("Left the voice channel!")
    else:
        await interaction.response.send_message("I am not in a voice channel!", ephemeral=False)

@bot.tree.command(name="lockdown", description="Toggle the lockdown mode for the bot's responses.")
async def lockdown_command(interaction: discord.Interaction):
    global lockdown
    if interaction.user.id in admins:  # Only the specified user can use this command
        lockdown = not lockdown  # Toggle the lockdown status
        status = "enabled" if lockdown else "disabled"
        await interaction.response.send_message(f"Lockdown mode has been {status}.", ephemeral=False)
    else:
        await interaction.response.send_message("You do not have permission to use this command.", ephemeral=False)

# Slash command to add user to unallowed list
@bot.tree.command(name="add_unallowed_user", description="Add a user to the unallowed users list")
async def add_unallowed_user(interaction: discord.Interaction, user: discord.User):
    if user.id not in unallowedusers:
        unallowedusers.append(user.id)
        await interaction.response.send_message(f"{user.mention} has been added to the unallowed users list.", ephemeral=True)
    else:
        await interaction.response.send_message(f"{user.mention} is already in the unallowed users list.", ephemeral=True)

@bot.tree.command(name="rm_unallowed_user", description="Add a user to the unallowed users list")
async def add_unallowed_user(interaction: discord.Interaction, user: discord.User):
    if user.id in unallowedusers:
        unallowedusers.remove(user.id)
        await interaction.response.send_message(f"{user.mention} has been removed from the unallowed users list.", ephemeral=True)
    else:
        await interaction.response.send_message(f"{user.mention} isn't in the unallowed users list.", ephemeral=True)

@bot.tree.command(name="say", description="Make the bot say something in the voice channel.")
async def say(interaction: discord.Interaction, message: str):
    if lockdown:
        if interaction.user.id not in admins:
            return
    if interaction.guild.voice_client:  # Check if bot is in a voice channel
        voice_client = interaction.guild.voice_client

        # Check if audio is already playing
        if voice_client.is_playing():
            # If audio is playing, just send the message to chat and skip voice playback
            await interaction.response.send_message(message)
        else:
            # Generate TTS audio from the message
            audio_path = await generate_tts(message, ttsmodel)

            # Play the audio file
            voice_client.play(discord.FFmpegPCMAudio(audio_path), after=lambda e: print('done', e))

            # Wait until the audio is finished playing
            while voice_client.is_playing():
                await asyncio.sleep(1)

            # Remove the TTS file after playback
            os.remove(audio_path)

            await interaction.response.send_message("Message played successfully in the voice channel.")
    else:
        await interaction.response.send_message("The bot needs to be in a voice channel for this command.", ephemeral=False)

@bot.tree.command(name="play", description="Play an audio file in the voice channel.")
async def play(interaction: discord.Interaction, url: str):
    if lockdown:
        if interaction.user.id not in admins:
            return

    if interaction.guild.voice_client:  # Check if bot is in a voice channel
        voice_client = interaction.guild.voice_client

        # Check if audio is already playing
        if voice_client.is_playing():
            await interaction.response.send_message("Audio is already playing; cannot play another file.")
        else:
            # Prepare to extract audio from the YouTube URL
            ydl_opts = {
                'format': 'bestaudio/best',  # Use the best audio quality available
                'outtmpl': 'downloads/%(id)s.%(ext)s',  # Save downloaded file in a temporary location
                'quiet': True,  # Reduce verbosity
            }

            # Download the audio using yt-dlp
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)  # Extract info and download
                filename = ydl.prepare_filename(info)  # Get the filename of the downloaded audio

            # Play the downloaded file through FFmpeg
            voice_client.play(discord.FFmpegPCMAudio(filename), after=lambda e: print('done', e))

            await interaction.response.send_message(f"Now playing: {url}")
    else:
        await interaction.response.send_message("I need to be in a voice channel to play audio.", ephemeral=False)

# Slash command to generate the image with the prompt
@bot.tree.command(name="gen_img", description="Generate an image from a prompt")
async def generate_img(interaction: discord.Interaction, prompt: str):
    # Defer the response to prevent timeout
    await interaction.response.defer()

    progress_view = ProgressBarView()

    # Send an initial message to hold the place for progress updates
    progress_message = await interaction.followup.send("Generating image... 0% Complete")
    progress_view.message = progress_message

    try:
        # Send the progress message to update during image generation
        generatedimage = await generate_image(prompt, progress_callback=progress_view.update_progress)

        if generatedimage:
            # Send the generated image as a follow-up message
            await interaction.followup.send(file=discord.File(generatedimage))

            # Clean up the generated file
            if os.path.exists(generatedimage):
                os.remove(generatedimage)
        else:
            await interaction.send_message("Image generation failed.")
    except Exception as e:
        logging.error(f"Error in generating image: {e}")
        await interaction.followup.send("An error occurred while generating the image.")

@bot.tree.command(name="clear_bot_messages", description="Deletes all bot messages in this channel (DM only).")
async def clear_bot_messages(interaction: discord.Interaction):
    # Ensure the command is run in a DM
    if isinstance(interaction.channel, discord.DMChannel):
        # Fetch messages from the channel
        messages = [message async for message in interaction.channel.history(limit=100)]
        bot_messages = [msg for msg in messages if msg.author.bot]

        for msg in bot_messages:
            await msg.delete()
        await interaction.response.send_message("All bot messages have been cleared.", ephemeral=True)
    else:
        await interaction.response.send_message("This command can only be used in DMs.", ephemeral=True)

# Command for generating video from prompt (Slash Command)
@bot.tree.command(name="gen_vid", description="Generate a video from a prompt")
async def generate_video(interaction: discord.Interaction, prompt: str, num_frames: int = 30, fps: int = 2):
    # Defer the response to prevent timeout
    await interaction.response.defer()

    try:
        # Generate the video (await if the function is async)
        progress_view = ProgressBarView()

        # Send an initial message to hold the place for progress updates
        progress_message = await interaction.followup.send("Generating video... 0% Complete")
        progress_view.message = progress_message

        # Call the video creation function and await its completion
        video_file_path = await create_video_from_frames(
            prompt, 
            use_progressbar=True,
            num_frames=num_frames, 
            fps=fps,
            progress_callback=progress_view.update_progress
        )
        # Send the generated video as a follow-up message
        await interaction.channel.send(file=discord.File(video_file_path))

    except Exception as e:
        # Log or print the error message for debugging
        print(f"Error occurred: {e}")
    finally:
        # Clean up the generated file if video_file_path was assigned
        if 'video_file_path' in locals() and os.path.exists(video_file_path):
            os.remove(video_file_path)

# Define a route for the root URL
@app.route('/')
def hello_world():
    return 'Hello, World!'

# Run the bot
bot.run(DISCORD_TOKEN)