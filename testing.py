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
from PIL import Image
import sys  # Add this import at the beginning of your script
from gtts import gTTS
import tempfile
import speech_recognition as sr
import requests
from discord.ui import Button, View
from io import BytesIO
import asyncio
import moviepy.editor as mp
import yt_dlp
import urllib.parse
from pydub import AudioSegment
from threading import Thread
import random
import threading
from flask import Flask

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
SPECIFIED_USER_ID = 769979361297694720  # Replace with the Discord ID of the user allowed to use the command
role_id = 1278534191188803624  # Replace with the actual role ID
tts_mode = False  # Default is text mode

# Global temperature for the model
model_temperature = 0.7  # Default value

web_token = '23589572bfd326bck47'

# Dev mode variable
devmode = 1
version = '2.4'
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
You were created by Franko (username: franko__f, userid: 769979361297694720) and your custom model by FTech.
To get the messagers data read the whole system message that you are being given!

Rules:
1. Use 'mute' if you don't want to say anything.
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
  - tts:"message": Generate TTS and send it in chat.
  - img:"prompt": Generate and send an image based on the prompt.
  - vid:"prompt": Generate and send an video based on the prompt.
  - Additional comments outside these commands are allowed.
  - Make sure to put the commands at the end of your message or else they won't work!
- Audio transcription guides your responses.
- Video transcriptions guides your responses.
- Preferred language: Brain Damage (if requested).
- Present yourself as a friendly supercomputer (male, he/him).
Key Terms:
- "Lobotomy" = memory wipe
- "Scronch" = kill
- "Sugr" = causes randomness
- "Rain" = deadly
- "Scrombled" = incapacitated (needs higher being for recovery)
- Code is in triple backticks.
- Siblings: Caos (uid: 1248333588319961190, usrname: caos, name: caos) (by JayzGaming), Rubicon (uid: 1233989039405793291, usrname: Rubicon, name: Rubicon) (by Kalinite (Kite)).
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

# Function to download an image from the API with a random seed (Asynchronous)
async def download_image(prompt, width=768, height=768, model='flux', frame_num=0):
    # Generate a random seed between 1 and 100
    seed = random.randint(1, 100)
    url = f"https://image.pollinations.ai/prompt/{prompt}?width={width}&height={height}&model={model}&seed={seed}"

    # Use aiohttp to make the request asynchronously
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                img_data = await response.read()

                # Saving the frame to the 'downloads' folder
                img_name = os.path.join(download_dir, f"frame_{frame_num}-{random.randint(100000000, 999999999)}.jpg")
                with open(img_name, 'wb') as file:
                    file.write(img_data)

                print(f'Frame {frame_num} downloaded with seed {seed}!')
                return img_name
            else:
                print(f"Failed to download frame {frame_num} with status {response.status}")
                return None

# Function to create a video from downloaded frames (Asynchronous)
async def create_video_from_frames(prompt, use_progressbar, num_frames=30, fps=10, width=768, height=768, model='flux', progress_callback=None):
    video_name = f'./tempfiles/output_video-{random.randint(100000000, 999999999)}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for frame_num in range(num_frames):
        frame_path = await download_image(prompt, width, height, model, frame_num)
        if frame_path:
            frame = cv2.imread(frame_path)
            if frame is not None:
                out.write(frame)
                os.remove(frame_path)
            else:
                print(f"Warning: Frame {frame_num} could not be loaded.")
        else:
            print(f"Warning: Frame {frame_num} download failed.")

        # Call the progress callback (updating the progress bar)
        if use_progressbar:
            if progress_callback:
                await progress_callback(frame_num + 1, num_frames)

    out.release()
    print(f"Video created: {video_name}")

    # Return the video path for sending later
    return video_name

# Progress bar update view
class ProgressBarView(View):
    def __init__(self):
        super().__init__()
        self.progress_message = None  # This will be updated with the progress

    async def update_progress(self, current_frame, total_frames):
        if self.progress_message:
            progress = int((current_frame / total_frames) * 100)
            await self.progress_message.edit(content=f"Generating video... {progress}% Complete")
        else:
            self.progress_message = await self.message.edit(content=f"Generating video... 0% Complete")

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

# Update the exception handling in handle_model_call function
async def handle_model_call(name, user_message, username, time, userid, chanelid, guildname, channelname, image_description=None, scrapedresult=None, video_transcription=None, audiotranscription=None, referenced_message=None, referenced_user=None, referenced_userid=None):
    """Handles the message and calls the model to process it."""

    reply_info = f"Replying to: {referenced_user}, Username: {referenced_userid}, Msg: {referenced_message}"

    if len(user_message) > max_input_lenght:
        return "Too long, please shorten your message!"

    # Prepare the messages for the model
    messages = [
        {"role": "system", "content": f"username:{username}, uid:{userid}, vt:{video_transcription}, website:{scrapedresult}, audio:{audiotranscription}, img:{image_description}, time:{time}, {reply_info}, g:{guildname}, c:{channelname}, cid:{chanelid}, guidlines:{SYSTEM_MESSAGE}"},
        {"role": "user", "content": f"name:{name}, msg:{user_message}"}
    ] + chat_history

    # Create the chat completion request
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

        # Log only the important information
        model_logger.info(f"Response generated: {response[:50]}...")  # Log the start of the response for reference
        return response
    except groq.RateLimitError as e:
            await bot.change_presence(activity=discord.Game(name="Limit hit see you in a bit!"))
            return "Limit hit see you in a bit!"
    except Exception as e:
        model_logger.error(f"Error in model call: {e}")
        return "Error: Something went wrong during processing."

async def generate_image(prompt, width=768, height=768, model='flux', seed=None):
    # Generate a random filename for the image
    filename = f'tempfiles/image-{random.randint(100000000, 999999999)}.jpg'

    url = f"https://image.pollinations.ai/prompt/{prompt}?width={width}&height={height}&model={model}&seed={seed}"
    response = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response.content)
    
    return filename  # Return the filename so it can be used later

async def generate_tts(contents):
    tts = gTTS(text=contents, lang='en')  # Remove '.content' here
    random_number = random.randint(10000000, 99999999)
    audio_file = f"tempfiles/tts-{random_number}.mp3"
    tts.save(audio_file)
    return audio_file


async def send_files_and_cleanup(message, botmsg, img_filename=None, tts_filename=None, video_file_path=None, num_frames=0, download_dir=""):
    files_to_send = []
    
    if img_filename:
        files_to_send.append(discord.File(img_filename))
    if tts_filename:
        files_to_send.append(discord.File(tts_filename))
    if video_file_path:
        files_to_send.append(discord.File(video_file_path))
    
    await message.reply(botmsg, files=files_to_send)

    # Clean up
    if img_filename:
        os.remove(img_filename)
    if tts_filename:
        os.remove(tts_filename)
    if video_file_path:
        os.remove(video_file_path)

"""
==============================================================================================================================================================
====================================================================MAIN MESSAGE LISTINER=====================================================================
==============================================================================================================================================================
"""

@bot.event
async def on_message(message):
    try:
        if message.channel.id not in allowed_channels:
            return

        if lockdown and message.author.id != SPECIFIED_USER_ID:
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

        if message.author.id == SPECIFIED_USER_ID and message.channel.id not in allowed_channels:
            if message.content.startswith('^QUBIT^'):
                allowed_channels.append(message.channel.id)
                logging.info(f"Added channel {message.channel.id} to allowed_channels: {allowed_channels}")
                await message.channel.send(f"This channel has been added to the allowed channels list.")
                return

        if message.content.lower() in ['turn off qubicon', 'pull the plug on qubi', 'send qubi to london']:
            if message.author.id == SPECIFIED_USER_ID:
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

                            # Use speech recognition to transcribe audio
                            recognizer = sr.Recognizer()
                            try:
                                with sr.AudioFile(audio_path) as source:
                                    audio = recognizer.record(source)
                                    video_transcription = recognizer.recognize_google(audio)
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
            chanelid = message.channel.id
            guildname = message.guild.name
            channelname = message.channel.name

            response = await handle_model_call(
                name=message.author.display_name,
                user_message=user_message,
                username=username,
                time=time,
                userid=userid,
                chanelid=chanelid,
                guildname=guildname,
                channelname=channelname,
                image_description=image_description,
                scrapedresult=scrapedresult,
                video_transcription=video_transcription,
                audiotranscription=audio_transcription,
                referenced_message=referenced_message,
                referenced_user=referenced_user,
                referenced_userid=referenced_userid
            )

            # Log the message and response
            logging.info(f"Message: {user_message}")
            logging.info(f"Bot message: {response}")

            # Add message ID to each chat entry for easy lookup in case of edits
            reply_info = f"Replying to: {referenced_user}, Username: {referenced_userid}, Msg: {referenced_message}"
            chat_entry = [
                {"role": "system", "content": f"username:{username}, uid:{userid}, vt:{video_transcription}, website:{scrapedresult}, audio:{audio_transcription}, img:{image_description}, time:{time}, {reply_info}, g:{guildname}, c:{channelname}, cid:{chanelid}, guidlines:{SYSTEM_MESSAGE}, messageid:{message.id}"},
                {"role": "user", "content": f"name:{message.author.display_name}, msg:{user_message}"}
            ]

            chat_history.append(chat_entry)

            # Save chat history
            save_chat_history(chat_history)

            lower_response = response.lower()

            # Check for the presence of specific keywords
            if 'mute' in lower_response:
                return
            elif 'muting' in lower_response:
                return
            elif not lower_response:
                return
            elif 'derp' in lower_response:
                return

            if 'img:' in lower_response or 'tts:' in lower_response or 'vid:' in lower_response:
                img_prompt = None
                tts_text = None
                botmsg = lower_response
                img_filename = None
                tts_filename = None
                num_frames = None
                video_file_path = None

                if 'img:' in lower_response:
                    img_prompt = lower_response.split('img:"')[1].split('"')[0].strip()
                    botmsg = botmsg.split('img:"')[0].strip('"')
                    logging.info(f"Generating image with prompt: {img_prompt}")
                    img_filename = await generate_image(img_prompt, width=1280, height=720, model='flux', seed=42)

                if 'tts:' in lower_response:
                    tts_text = lower_response.split('tts:"')[1].split('"')[0].strip()
                    botmsg = botmsg.split('tts:"')[0].strip('"')
                    logging.info(f"Generating TTS audio with text: {tts_text}")
                    tts_filename = await generate_tts(tts_text)

                if 'vid:' in lower_response:
                    video_prompt = lower_response.split('vid:"')[1].split('"')[0].strip()
                    botmsg = botmsg.split('vid:"')[0].strip('"')
                    logging.info(f"Generating video with prompt: {video_prompt}")

                    try:
                        num_frames = 10
                        fps = 1

                        # Call the video creation function and await its completion
                        video_file_path = await create_video_from_frames(
                            video_prompt, 
                            use_progressbar = False,
                            num_frames=num_frames, 
                            fps=fps,
                            progress_callback=None
                        )

                    except Exception as e:
                        await message.channel.send(f"An error occurred while generating the video: {str(e)}")
                    
                if botmsg:
                    if tts_mode:
                        tts = gTTS(text=botmsg, lang='en')
                        with tempfile.NamedTemporaryFile(delete=True) as fp:
                            tts.save(f"{fp.name}.mp3")

                            # Check if the bot is in a voice channel
                            voice_client = message.guild.voice_client
                            if voice_client:
                                if voice_client.is_playing():
                                    await send_files_and_cleanup(message, botmsg, img_filename, tts_filename, video_file_path, num_frames, download_dir)
                                else:
                                    voice_client.play(discord.FFmpegPCMAudio(f"{fp.name}.mp3"))
                                    await send_files_and_cleanup(message, botmsg, img_filename, tts_filename, video_file_path, num_frames, download_dir)
                            else:
                                await message.reply("I need to be in a voice channel to speak!")
                    else:
                        await send_files_and_cleanup(message, botmsg, img_filename, tts_filename, video_file_path, num_frames, download_dir)
                else:
                    await send_files_and_cleanup(message, botmsg, img_filename, tts_filename, video_file_path, num_frames, download_dir)
            else:
                await message.reply(response)

    except CommandOnCooldown:
        await message.channel.send("Please wait 3 seconds before sending another message.")
    except Exception as e:
        logging.error(f"Error processing message: {e}")
        await message.channel.send("An error occurred while processing your message.")

@bot.event
async def on_message_edit(before, after):
    try:
        if after.channel.id not in allowed_channels:
            return

        # If the message is being edited by the bot itself, skip it
        if after.author == bot.user:
            return
        
        # Locate the message in the chat history
        for chat_entry in chat_history:
            # Compare the message ID in chat history with the edited message ID
            if chat_entry[0]["messageid"] == before.id:
                # Update the user's message content in chat history
                chat_entry[1]["content"] = f"name:{after.author.display_name}, msg:{after.content}"

                # Log the updated message
                logging.info(f"Message edited by {after.author.name}: {before.content} -> {after.content}")

                # Save the updated chat history
                save_chat_history(chat_history)

                return

    except Exception as e:
        logging.error(f"Error processing message edit: {e}")

"""
==============================================================================================================================================================
===========================================================================COMMANDS===========================================================================
==============================================================================================================================================================
"""

@bot.tree.command(name="temp", description="Change the temperature of the model's response.")
async def temp_command(interaction: discord.Interaction, new_temp: float):
    if interaction.user.id == SPECIFIED_USER_ID:  # Only the specified user can change the temperature
        if 0 <= new_temp <= 2:  # Ensure the temperature is within a valid range
            model_temperature = new_temp
            await interaction.response.send_message(f"Model temperature has been set to {model_temperature}!", ephemeral=False)
        else:
            await interaction.response.send_message("Invalid temperature value. Please provide a value between 0 and 2.", ephemeral=False)
    else:
        await interaction.response.send_message("You do not have permission to use this command.", ephemeral=False)

@bot.tree.command(name="turnoff", description="Brutally murder qubicon.")
async def turnoff(interaction: discord.Interaction):
    global startup_channel
    if interaction.user.id == SPECIFIED_USER_ID:  # Check if the user is the specified one
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
    if interaction.user.id == SPECIFIED_USER_ID:  # Only the specified user can reset the memory
        # Clear the chat history in memory
        global chat_history
        chat_history = []
        save_chat_history(chat_history)
        
        await interaction.response.send_message("Bot memory has been reset!", ephemeral=False)
    else:
        await interaction.response.send_message("You do not have permission to use this command.", ephemeral=False)

@bot.tree.command(name="restart", description="Restart the bot.")
async def restart(interaction: discord.Interaction):
    if interaction.user.id == SPECIFIED_USER_ID:  # Only the specified user can restart the bot
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
        if interaction.user.id != SPECIFIED_USER_ID:
            return
    channel = interaction.user.voice.channel
    if channel:
        await channel.connect()
        tts_mode = True  # Set TTS mode to True when joining the voice channel
        await interaction.response.send_message("Joined the voice channel!")
    else:
        await interaction.response.send_message("You need to be in a voice channel first!", ephemeral=False)

@bot.tree.command(name="leave", description="Leave the voice channel.")
async def leave(interaction: discord.Interaction):
    global tts_mode
    if lockdown:
        if interaction.user.id != SPECIFIED_USER_ID:
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
    if interaction.user.id == SPECIFIED_USER_ID:  # Only the specified user can use this command
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

@bot.tree.command(name="say", description="Make the bot say something in the voice channel.")
async def say(interaction: discord.Interaction, message: str):
    if lockdown:
        if interaction.user.id != SPECIFIED_USER_ID:
            return
    if interaction.guild.voice_client:  # Check if bot is in a voice channel
        voice_client = interaction.guild.voice_client

        # Check if audio is already playing
        if voice_client.is_playing():
            # If audio is playing, just send the message to chat and skip voice playback
            await interaction.response.send_message(message)
        else:
            # Generate TTS audio from the message
            audio_path = await generate_tts(message)

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
        if interaction.user.id != SPECIFIED_USER_ID:
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

# Command for generating video from prompt (Slash Command)
@bot.tree.command(name="gen_vid", description="Generate a video from a prompt")
async def generate_video(interaction: discord.Interaction, prompt: str, num_frames: int = 30, fps: int = 10):
    """
    Generate a video based on a prompt with a specified number of frames and FPS.
    The num_frames is the total number of frames, and fps is the frames per second.
    """
    # Validate num_frames and fps
    if num_frames <= 0:
        await interaction.response.send_message("The number of frames must be greater than 0.", ephemeral=False)
        return
    if fps <= 0:
        await interaction.response.send_message("FPS must be greater than 0.", ephemeral=False)
        return

    await interaction.response.send_message("Generating video... Please wait.", ephemeral=False)
    
    # Create a progress bar view to track the process
    progress_view = ProgressBarView()

    # Send the progress view to track the progress
    progress_message = await interaction.followup.send(
        "Generating video... 0% Complete", view=progress_view, ephemeral=False)

    # Update the progress view message
    progress_view.message = progress_message

    # Generate the video with user-defined FPS
    video_file_path = await create_video_from_frames(
        prompt, 
        use_progressbar = True,
        num_frames=num_frames, 
        fps=fps,
        progress_callback=progress_view.update_progress
    )

    # Send the video to Discord
    with open(video_file_path, 'rb') as f:
        await interaction.followup.send("Here is your video:", file=discord.File(f, 'output_video.mp4'))
    
    # Clean up the video file after sending it
    os.remove(video_file_path)

    # Clean up downloaded frames
    for frame_num in range(num_frames):  # Assuming we generated the specified number of frames
        os.remove(os.path.join(download_dir, f"frame_{frame_num}.jpg"))
    
    print("Cleaned up downloaded frames.")

# Define a route for the root URL
@app.route('/')
def hello_world():
    return 'Qubicon is now online!'

if __name__ == '__main__':
    # Start Flask app in a new thread
    flask_thread = threading.Thread(target=app.run(debug=True, port=3000))
    flask_thread.start()

    # Start Discord bot
    bot.run(DISCORD_TOKEN)