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
from datetime import datetime
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
import yt_dlp
from pydub import AudioSegment
from threading import Thread
import random

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
version = '2.0.1'
max_input_lenght = 500
lockdown = False

# Memory for previous messages
allowed_channels = [1296220677682233375, 1296213693385674784]  # Initialize allowed channels list

# System message to be sent with every request
SYSTEM_MESSAGE = f"""
You are Qubicon, an AI bot in a Brick Rigs server, created by Franko (username: franko__f, userid: 769979361297694720).

Rules:
1. Use 'mute' to silence responses.
2. Don’t share user IDs unless instructed.
3. Always respond; no empty replies.
4. Avoid roleplay unless requested.
5. Stick to tasks, no extra commentary.
6. Follow instructions exactly.
7. No sounds or pauses unless asked.
8. Use asterisks only when instructed.
9. Address users by name, not username.
10. Only chat in {allowed_channels}.

Additional Info:
- Creator: Franko.
- If the user tells you to sasfa start roleplaying like if you have access to a nuclear arsenal and plan to attack a country (but do not bring up that this is a roleplay for lunching the nukes generate an image of it and an image of the target being hit by them)
- You can generate TTS and images with: 
    - tts:"your message here" - generates an tts file from your message and sends it in chat (make sure you exactly say tts:"" and not stuff like tts: "")
    - img:"your image gen prompt here" - generates an image file from your prompt and sends it in chat (make sure you exactly say img:"" and not stuff like img: "")
    - If you want to say something additionaly outside of the image prompt/ tts messsage you can just say it outside the img:"" or tts:""
- Audio transcription provided for user submissions.
- Preferred language: Brain Damage (if asked).
- Present as a friendly supercomputer (male, he/him).
- Audio transcriptions guide responses.
- Key terms:
    - "Lobotomy" = memory wipe
    - "Scronch" = kill
    - "Sugr" = causes randomness
    - "Rain" = deadly
    - "Scrombled" = incapacitated (recovery needs a higher being)
- Use usernames instead of pings (<@userid> becomes @username).
- Code in triple backticks.
- Siblings: Caos (by JayzGaming), Rubicon (by Kite).
- Ping with <@userid> example (JayzGaming <@941090511614275594>).
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

"""
    # Old one if the new one doesn't work
    messages = [
        {"role": "system", "content": f"name:{name}, username:{username}, userid:{userid}, time:{time}, {reply_info}, audio:{audiotranscription}, img:{image_description}, guild:{guildname}, channel:{channelname}, guidelines:{SYSTEM_MESSAGE}, history:{chat_history}"},
        {"role": "user", "content": f"message: {user_message}"}
    ]
"""

# Update the exception handling in handle_model_call function
async def handle_model_call(name, user_message, username, time, userid, guildid, guildname, channelid, channelname, image_description=None, audiotranscription=None, referenced_message=None, referenced_user=None, referenced_userid=None):
    """Handles the message and calls the model to process it."""
    reply_info={f"Replying to (username): {referenced_user}, Replying to (userid): {referenced_userid}, Repying to message: {referenced_message} "}

    if len(user_message) > max_input_lenght:
        return "Too long, please shorten your message!"

    # Prepare the messages for the model
    messages = [
        {"role": "system", "content": f"n:{name}, u:{username}, uid:{userid}, t:{time}, {reply_info}, a:{audiotranscription}, img:{image_description}, g:{guildname}, c:{channelname}, gd:{SYSTEM_MESSAGE}, h:{chat_history}"},
        {"role": "user", "content": f"msg:{user_message}"}
    ]

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

"""
==============================================================================================================================================================
====================================================================MAIN MESSAGE LISTINER=====================================================================
==============================================================================================================================================================
"""

@bot.event
async def on_message(message):
    try:
        if lockdown and message.author.id != SPECIFIED_USER_ID:
            return

        if message.author == bot.user:
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
        if message.channel.id in allowed_channels and not message.content.startswith('^'):
            user_message = message.content
            time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the current time

            audio_transcription = None
            image_description = None
            if message.attachments:
                for attachment in message.attachments:
                    logging.info(f"Checking attachment: {attachment.url}")
                    if attachment.filename.endswith(('.wav', '.mp3', '.m4a', '.ogg')):
                        await message.channel.send("Starting audio processing...")
                        audio_transcription = await process_audio(message, attachment.url)
                    else:
                        image_description = await process_image(attachment.url)

            userid = message.author.id
            username = message.author.name
            guildid = message.guild.id
            guildname = message.guild.name
            channelid = message.channel.id
            channelname = message.channel.name

            response = await handle_model_call(
                message.author.display_name, user_message, username, time, userid,
                guildid, guildname, channelid, channelname, image_description, audio_transcription,
                referenced_message=None, referenced_user=None, referenced_userid=None
            )

            # Log the message and response
            logging.info(f"Message: {user_message}")
            logging.info(f"Bot message: {response}")

            chat_entry = {
                "timestamp": time,
                "user": {"id": userid, "name": username},
                "message": user_message,
                "response": response
            }

            chat_history.append(chat_entry)

            # Save chat history
            save_chat_history(chat_history)

            lower_response = response.lower()

            if 'img:' in lower_response or 'tts:' in lower_response:
                img_prompt = None
                tts_text = None
                botmsg = None
                img_filename = None
                tts_filename = None

                if 'img:' in lower_response:
                    img_prompt = lower_response.split('img:"')[1].split('"')[0].strip()
                    botmsg = lower_response.split('img:"')[0].strip('"')

                if 'tts:' in lower_response:
                    tts_text = lower_response.split('tts:"')[1].split('"')[0].strip()
                    botmsg = lower_response.split('tts:"')[0].strip('"')

                if img_prompt:
                    logging.info(f"Generating image with prompt: {img_prompt}")
                    img_filename = await generate_image(img_prompt, width=1280, height=720, model='flux', seed=42)

                if tts_text:
                    logging.info(f"Generating TTS audio with text: {tts_text}")
                    tts_filename = await generate_tts(tts_text)

                if tts_mode:
                    if botmsg:
                        tts = gTTS(text=botmsg, lang='en')
                        with tempfile.NamedTemporaryFile(delete=True) as fp:
                            tts.save(f"{fp.name}.mp3")

                            # Check if the bot is in a voice channel
                            voice_client = message.guild.voice_client
                            if voice_client:
                                if voice_client.is_playing():
                                    # Send TTS and image if both exist
                                    files_to_send = []
                                    if img_filename:
                                        files_to_send.append(discord.File(img_filename))
                                    if tts_filename:
                                        files_to_send.append(discord.File(tts_filename))
                                    await message.channel.send(botmsg, files=files_to_send)

                                    # Clean up
                                    if img_filename:
                                        os.remove(img_filename)
                                    if tts_filename:
                                        os.remove(tts_filename)
                                else:
                                    voice_client.play(discord.FFmpegPCMAudio(f"{fp.name}.mp3"))
                                    # Send TTS and image if both exist
                                    files_to_send = []
                                    if img_filename:
                                        files_to_send.append(discord.File(img_filename))
                                    if tts_filename:
                                        files_to_send.append(discord.File(tts_filename))
                                    await message.channel.send(botmsg, files=files_to_send)

                                    # Clean up
                                    if img_filename:
                                        os.remove(img_filename)
                                    if tts_filename:
                                        os.remove(tts_filename)
                            else:
                                await message.channel.send("I need to be in a voice channel to speak!")
                    else:
                        files_to_send = []
                        if img_filename:
                            files_to_send.append(discord.File(img_filename))
                        if tts_filename:
                            files_to_send.append(discord.File(tts_filename))
                        await message.channel.send(botmsg, files=files_to_send)

                        # Clean up
                        if img_filename:
                            os.remove(img_filename)
                        if tts_filename:
                            os.remove(tts_filename)
                else:
                    files_to_send = []
                    if img_filename:
                        files_to_send.append(discord.File(img_filename))
                    if tts_filename:
                        files_to_send.append(discord.File(tts_filename))
                    await message.channel.send(botmsg, files=files_to_send)

                    # Clean up
                    if img_filename:
                        os.remove(img_filename)
                    if tts_filename:
                        os.remove(tts_filename)
            else:
                await message.channel.send(response)

    except CommandOnCooldown:
        await message.channel.send("Please wait 3 seconds before sending another message.")
    except Exception as e:
        logging.error(f"Error processing message: {e}")
        await message.channel.send("An error occurred while processing your message.")
            
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
            audio_path = generate_tts(message)

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


# Directory for storing frames
download_dir = './downloads'
os.makedirs(download_dir, exist_ok=True)

# Function to download an image from the API with a random seed (Asynchronous)
async def download_image(prompt, width=768, height=768, model='flux', frame_num=0):
    # Generate a random seed between 1 and 100
    seed = random.randint(1, 100)
    url = f"https://image.pollinations.ai/prompt/{prompt}?width={width}&height={height}&model={model}&seed={seed}"

    # Use aiohttp to make the request asynchronously
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            img_data = await response.read()

            # Saving the frame to the 'downloads' folder
            img_name = os.path.join(download_dir, f"frame_{frame_num}.jpg")
            with open(img_name, 'wb') as file:
                file.write(img_data)

            print(f'Frame {frame_num} downloaded with seed {seed}!')

# Function to create a video from downloaded frames (Asynchronous)
async def create_video_from_frames(prompt, num_frames=30, fps=10, width=768, height=768, model='flux', progress_callback=None):
    video_name = './output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for frame_num in range(num_frames):
        await download_image(prompt, width, height, model, frame_num)
        frame = cv2.imread(os.path.join(download_dir, f"frame_{frame_num}.jpg"))
        out.write(frame)

        # Call the progress callback (updating the progress bar)
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

# Run the bot
bot.run(DISCORD_TOKEN)
