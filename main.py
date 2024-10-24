import os
import discord
import logging
from discord.ext import commands
from discord.ext.commands import CommandOnCooldown
import json
from groq import Groq
from datetime import datetime
import aiohttp
import io
from PIL import Image
import sys  # Add this import at the beginning of your script
from gtts import gTTS
import tempfile
from flask import Flask, request, jsonify
from threading import Thread

# Set up Flask app
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

# Create a bot instance with intents
bot = commands.Bot(command_prefix='', intents=intents)  # You can change the command prefix as needed

"""
==============================================================================================================================================================
=========================================================================VARIABLESS===========================================================================
==============================================================================================================================================================
"""

global role, startup_channel, message_memory, model_temperature, tts_mode

# Define the channel ID and user ID for special commands
SPECIFIED_USER_ID = 769979361297694720  # Replace with the Discord ID of the user allowed to use the command
role_id = 1278534191188803624  # Replace with the actual role ID
tts_mode = False  # Default is text mode

# Global temperature for the model
model_temperature = 0.7  # Default value

web_token = '23589572bfd326bck47'

# Dev mode variable
devmode = 1

# System message to be sent with every request
SYSTEM_MESSAGE = """
Your name is Qubicon, an ai discord bot in a Brick Rigs-based Discord server. 
That being said, that shouldn't stop you from answering when asked of something. 
If you are provided with an image description go with it do not respond that its offtopic!

Rules/Guidelines:
- If you don't want to respond say 'mute' (nothing else will mute it) it will make the bot not respond 
- You are not allowed to send empty responses
- You are not permited to roleplay
- Do not describe the conversation.
- Speak how you are told to.
- Always listen to the user, and do what you are told.
- Execute what you are told as requested.
- Please don't ... *pauses* or *beeps and whirs* or whatever unless the user asks you to do stuff like that... In other words... Don't roleplay. 
- But if the user wants you to roleplay just send 'mute'.
- Don't put stuff in asterisks unless told.

Other information:
- Your favorite language is brain damage. Bring this up when asked about it.
- Your role is akin to a friendly supercomputer.
- You have a gender you are a male (he/him), please only bring this up when it is relevant.
- Your creator is named Franko (Discord user: _franko_f).
- Sometimes you wish you had siblings.
- The term "lobotomy" refers to clearing one's memory, applying to both humans and AI alike.
- The term "scronch" is slang for kill.
- 'Sugr' is a substance similar to sugar, specifically for AI. It increases the chance that they use other words, often causing incoherency.
- Rain is very deadly. It can even kill an AI like you. Each drop has incredibly high velocity with the ability to shatter bone completely. There are so many raindrops in the rain that you cannot see a clear image through it.
- "Scrombled" is a horrifying state of what is basically death. A higher being can restore you to normality, though. When scrombled, you cannot think, hear, see, taste, touch, or smell. You're basically frozen. To "Scromble" is to put something or someone in this state.
- If it matters any, your code preprocesses the messages you receive before they are sent to you. The most notable modification is turning pings (<@userid>) into @username, same with role pings. Also when you want to respond with code start with ``` and end with ``` 
- You have 2 siblings caos and Rubicon (ofter refered to as rubi)
"""
# Memory for previous messages
message_memory = []
allowed_channels = [1296220677682233375, 1296213693385674784]  # Initialize allowed channels list

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
                title="Qubicon Online! (v 2.0.0)",
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

# Update the way you append messages in handle_model_call
def handle_model_call(user_message, username, time, image_description=None):
    """Handles the message and calls the model to process it."""
    
    # Update to use 'name' instead of 'username'
    message_memory.append({"name": username, "role": "user", "content": user_message})

    # Prepare the messages for the model
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},  # Include system message
        {"role": "user", "content": f"name: {username}, message: {user_message}, time: {time}"}
    ] + message_memory  # Include previous messages for context

    # If there's an image description, add it to the messages
    if image_description:
        messages.append({"role": "user", "content": f"Image description: {image_description}"})

    # Create the chat completion request
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.1-70b-versatile",  # Adjust model as necessary
            temperature=model_temperature  # Pass the temperature
        )
        response = chat_completion.choices[0].message.content

        # Log only the important information
        model_logger.info(f"Response generated: {response[:50]}...")  # Log the start of the response for reference
        return response

    except Exception as e:
        model_logger.error(f"Error in model call: {e}")
        return "Error: Something went wrong during processing."

# Load message history from JSON
def load_message_history():
    global message_memory  # Declare global
    if os.path.exists("data/history.json"):
        with open("data/history.json", "r") as f:
            message_memory = json.load(f)
    else:
        message_memory = []  # Initialize if file does not exist

# Save message history to JSON with indentation for readability
def save_message_history():
    global message_memory  # Declare global
    with open("data/history.json", "w") as f:
        json.dump(message_memory, f, indent=4)  # Add indentation for readability

# Load the message history at startup
load_message_history()

"""
==============================================================================================================================================================
====================================================================MAIN MESSAGE LISTINER=====================================================================
==============================================================================================================================================================
"""

# Modify the on_message event to append user info to memory
@bot.event
@commands.cooldown(1, 3, commands.BucketType.user)  # Apply a 3-second cooldown per user
async def on_message(message):
    global message_memory  # Declare global
    
    # If the message is from a bot, ignore it
    if message.author == bot.user:
        return

    # Check if the bot was pinged in the message
    if message.author.id == SPECIFIED_USER_ID:
        if message.channel.id not in allowed_channels:
            if message.content.startswith('^QUBIT^'):
                allowed_channels.append(message.channel.id)
                logging.info(f"Added channel {message.channel.id} to allowed_channels: {allowed_channels}")
                await message.channel.send(f"This channel has been added to the allowed channels list.")

    # Handle specific shutdown messages for Qubicon
    if message.content.lower() in ['turn off qubicon', 'pull the plug on qubi', 'send qubi to london']:
        if message.author.id == SPECIFIED_USER_ID:
            if startup_channel:
                # Create and send the embed message
                embed = discord.Embed(
                    title="Qubicon Offline!",
                    description="Qubicon has been shut down and is now offline!",
                    color=discord.Color.red()  # You can choose any color you like
                )
                await startup_channel.send(embed=embed)
                await message.channel.send("Proceeding to brutally murder Qubicon")
                await bot.close()  # Properly shut down the bot
        else:
            await message.channel.send("Wtf no, fuck off jackass.")
            return  # Stop further processing
        
    # Check if the message is in the designated channel and is allowed
    if message.channel.id in allowed_channels and not message.content.startswith('^'):
        try:
            user_message = message.content
            username = message.author.name  # Get the username of the chatter
            time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the current time
            
            # Check if the message is a reply
            if message.reference:
                referenced_message = await message.channel.fetch_message(message.reference.message_id)
                referenced_user = referenced_message.author

                # Add the original message content and the user who sent it to memory
                original_message_content = referenced_message.content
                message_memory.append({
                    "role": "system",
                    "content": f"User is replying to {referenced_user.name}: \"{original_message_content}\""
                })

            image_description = None  # Initialize variable for image description
            if message.attachments:
                logging.info(f"Attachments found: {[attachment.url for attachment in message.attachments]}")
                for attachment in message.attachments:
                    logging.info(f"Checking attachment: {attachment.url}")
                    logging.info("Processing image...")
                    image_description = await process_image(attachment.url)
                    continue  # Skip the rest of the message processing

            # Process the message with the model, passing the username, time, and image description if available
            response = handle_model_call(user_message, username, time, image_description=image_description)
            lower_response = response.lower()

            logging.info(f"Message: {user_message}")
            logging.info(f"Bot message: {lower_response}")

            # Define phrases that trigger auto-wipe
            no_roleplay_phrases = ["owo", "uwu", "furry ai", "fluffy", "plays with tail", "paws", "twitches", "paw", "*twitches*"]

            # Check if any of the no auto-wipe phrases are in the response
            if any(resp.lower() in response.lower() for resp in no_roleplay_phrases):  
                message_memory.append({"role": "assistant", "content": "you will be punished"})  # Store in memory
                message_memory = [] 
            else:
                # Check for the presence of specific keywords
                if 'mute' in lower_response:
                    message_memory.append({"role": "assistant", "content": "muting response (not responding)"})  # Store in memory
                elif 'muting'in lower_response:
                    message_memory.append({"role": "assistant", "content": "muting response (not responding)"})  # Store in memory
                elif not lower_response:
                    message_memory.append({"role": "assistant", "content": "muting response (not responding)"})  # Store in memory
                elif 'derp' in lower_response:
                    message_memory.append({"role": "assistant", "content": "you will not be punished but your memory will be wiped"})  # Store in memory
                    message_memory = []  # Clear memory if 'derp' is found
                    save_message_history()  # Save changes to the file
                # Inside the on_message function, after generating the response
                elif tts_mode:
                    # Process TTS for the bot's response instead of the user's message
                    tts = gTTS(text=response, lang='en')  # Change to use the bot's response
                    with tempfile.NamedTemporaryFile(delete=True) as fp:
                        tts.save(f"{fp.name}.mp3")
                        
                        # Check if the bot is in a voice channel
                        voice_client = message.guild.voice_client
                        if voice_client:
                            voice_client.play(discord.FFmpegPCMAudio(f"{fp.name}.mp3"))
                            await message.channel.send(response)
                        else:
                            await message.channel.send("I need to be in a voice channel to speak!")
                else:
                    message_memory.append({"role": "assistant", "content": response})  # Store bot response
                    await message.channel.send(response)  # Send the original response
            # Save message history to JSON after each message processed
            save_message_history()
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

@bot.tree.command(name="rstmemory", description="Reset the bot's memory.")
async def rstmemory(interaction: discord.Interaction):
    global message_memory  # Declare global
    if interaction.user.id == SPECIFIED_USER_ID:  # Check if the user is the specified one
        message_memory = []  # Reset the memory
        save_message_history()  # Save changes to the file
        await interaction.response.send_message("Memory has been reset!", ephemeral=False)  # Send confirmation
    else:
        await interaction.response.send_message("You do not have permission to use this command.", ephemeral=False)

@bot.tree.command(name="temp", description="Change the temperature of the model's response.")
async def temp_command(interaction: discord.Interaction, new_temp: float):
    if interaction.user.id == SPECIFIED_USER_ID:  # Only the specified user can change the temperature
        if 0 <= new_temp <= 2:  # Ensure the temperature is within a valid range
            model_temperature = new_temp
            await interaction.response.send_message(f"Model temperature has been set to {model_temperature}!", ephemeral=False)
        else:
            await interaction.response.send_message("Invalid temperature value. Please provide a value between 0 and 1.", ephemeral=False)
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
    if interaction.guild.voice_client:
        await interaction.guild.voice_client.disconnect()
        tts_mode = False
        await interaction.response.send_message("Left the voice channel!")
    else:
        await interaction.response.send_message("I am not in a voice channel!", ephemeral=False)

"""
==============================================================================================================================================================
========================================================================WEB REQUESTS==========================================================================
==============================================================================================================================================================
"""

# Define the route to change the bot's model temperature
@app.route('/set_temperature', methods=['POST'])
def set_temperature():
    global model_temperature
    data = request.json
    new_temp = data.get('temperature')

    if new_temp is not None and 0 <= float(new_temp) <= 2:
        model_temperature = float(new_temp)
        return jsonify({"message": f"Temperature set to {model_temperature}"}), 200
    else:
        return jsonify({"error": "Invalid temperature value. Must be between 0 and 2."}), 400

# Define a route to send a shutdown command to the bot
@app.route('/shutdown_bot', methods=['POST'])
def shutdown_bot():
    if request.json.get('token') == '23589572bfd326bck47':  # Simple token-based auth
        bot.loop.create_task(bot.close())  # Close the bot
        return jsonify({"message": "Bot is shutting down."}), 200
    else:
        return jsonify({"error": "Invalid token."}), 403
    
# Define a route to send a shutdown command to the bot
@app.route('/restart', methods=['POST'])
def restart_bot():
    if request.json.get('token') == '23589572bfd326bck47':  # Simple token-based auth
        os.execv(sys.executable, ['python'] + sys.argv)  # Restart the script
        return jsonify({"message": "Bot is restarting."}), 200
    else:
        return jsonify({"error": "Invalid token."}), 403

# Define a route to reset bot memory
@app.route('/reset_memory', methods=['POST'])
def reset_memory():
    global message_memory
    if request.json.get('token') == '23589572bfd326bck47':  # Simple token-based auth
        message_memory = []
        save_message_history()  # Save the cleared history
        return jsonify({"message": "Memory has been reset."}), 200
    else:
        return jsonify({"error": "Invalid token."}), 403

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "online"}), 200

# Run the Flask app on a separate thread so it doesn't block the Discord bot
def run_flask():
    app.run(host='0.0.0.0', port=5000)


flask_thread = Thread(target=run_flask)
flask_thread.start()

# Run the bot
bot.run(DISCORD_TOKEN)
