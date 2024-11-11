import discord
import aiohttp
import requests
from discord.ext import commands
from PIL import Image
import io
import cv2
import os
from pydub import AudioSegment
import speech_recognition as sr
import time
import asyncio
from tempfile import NamedTemporaryFile
from groq import Groq
import groq

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
# Initialize the Groq client
client = Groq(api_key=os.getenv("GROQ"))
bot = commands.Bot(command_prefix="!", intents=intents)

recognizer = sr.Recognizer()

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

async def process_audio(message, audio_url):
    """Processes the audio URL, converts it, and returns a transcription."""
    try:
        response = requests.get(audio_url)
        if response.status_code != 200:
            await message.channel.send("Failed to download audio.")
            return None

        audio_data = io.BytesIO(response.content)

        # Convert audio to WAV format using pydub
        audio_segment = AudioSegment.from_file(audio_data)
        if audio_segment is None:
            await message.channel.send("Failed to convert audio to WAV.")
            return None

        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)

        with sr.AudioFile(wav_io) as source:
            audio = recognizer.record(source)
        audiotranscription = recognizer.recognize_google(audio)

        return audiotranscription

    except sr.UnknownValueError:
        await message.channel.send("No words detected in the audio.")
        return None
    except Exception as e:
        await message.channel.send(f"Error processing audio: {e}")
        return None

async def handle_model_call(video_descritpion=None, audiotranscription=None):
    """Handles the message and calls the model to process it."""

    # Prepare the messages for the model
    messages = [
        {"role": "system", "content": f"You will be provided with an audio transcription and all frames description of a video we want you to tell us about the video that was givven to you by text form"},
        {"role": "user", "content": f"audio:{audiotranscription}, video desc: {video_descritpion}"}
    ]

    # Create the chat completion request
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.1-70b-versatile",  # Adjust model as necessary
            temperature=0.7  # Pass the temperature
        )
        response = chat_completion.choices[0].message.content

        return response
    except groq.RateLimitError as e:
            await bot.change_presence(activity=discord.Game(name="Limit hit see you in a bit!"))
            return "Limit hit see you in a bit!"
    except Exception as e:
        return "Error: Something went wrong during processing."


async def split_video_into_frames(video_url):
    """Downloads a video from a URL and splits it into frames."""
    response = requests.get(video_url)
    if response.status_code != 200:
        return "Error: Couldn't download the video."

    # Ensure the ./tempfiles directory exists
    temp_dir = './tempfiles'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Save the video temporarily in the ./tempfiles directory
    temp_video_path = os.path.join(temp_dir, 'temp_video.mp4')
    with open(temp_video_path, 'wb') as f:
        f.write(response.content)

    try:
        # Open video file using OpenCV
        video_capture = cv2.VideoCapture(temp_video_path)
        frame_urls = []
        frame_count = 0

        # Check if video was successfully opened
        if not video_capture.isOpened():
            print(f"Error: Failed to open video file {temp_video_path}")
            return "Error: Couldn't open the video file."

        while True:
            ret, frame = video_capture.read()
            print(f"Frame {frame_count}: ret={ret}, frame={frame is not None}")  # Debugging output

            if not ret or frame is None:
                print(f"Exiting: No more frames or invalid frame at {frame_count}")
                break  # Exit the loop if frame is not valid

            frame_count += 1
            frame_filename = f'frame_{frame_count}.jpg'
            frame_path = os.path.join(temp_dir, frame_filename)

            # Save the frame to a temporary file
            if frame is not None:
                cv2.imwrite(frame_path, frame)
                frame_urls.append(frame_path)
                print(f"Saved frame {frame_count} as {frame_filename}")  # Debugging output
            else:
                print(f"Skipping invalid frame {frame_count}")

        # Close the video capture to release the file handle
        video_capture.release()
        print(f"Processed {frame_count} frames.")

    except Exception as e:
        print(f"Error processing video: {e}")
        return "Error: Couldn't process the video frames."

    finally:
        # Cleanup the temporary video file
        try:
            os.remove(temp_video_path)
        except PermissionError:
            print(f"Error deleting temp file: {temp_video_path}")

    return frame_urls

@bot.event
async def on_message(message):
    """Handles messages containing video and/or audio attachments."""
    if message.author == bot.user:
        return

    if message.attachments:
        video_attachment = None
        audio_attachment = None

        # Check for video and audio attachments
        for attachment in message.attachments:
            if attachment.filename.lower().endswith(('.mp4', '.avi', '.mov')):
                video_attachment = attachment
            elif attachment.filename.lower().endswith(('.mp3', '.wav', '.ogg')):
                audio_attachment = attachment

        tasks = []

        if video_attachment:
            video_url = video_attachment.url
            tasks.append(process_video_and_audio(message, video_url, audio_attachment))

        elif audio_attachment:
            audio_url = audio_attachment.url
            tasks.append(process_audio(message, audio_url))

        # Process all tasks concurrently
        await asyncio.gather(*tasks)

async def send_large_message(channel, content, file=None):
    """Splits a large message into multiple parts if it's too long (2000 char limit)."""
    # Split the content into chunks of 2000 characters
    for i in range(0, len(content), 2000):
        await channel.send(content[i:i+2000], file=file)

async def process_video_and_audio(message, video_url, audio_attachment):
    """Process both video and audio concurrently and send the results to the model."""

    # First, process the video frames
    frame_urls = await split_video_into_frames(video_url)

    # Process frames and collect descriptions
    frame_descriptions = []
    for frame_path in frame_urls:
        # Upload the frame image to Discord before deleting
        file = discord.File(frame_path)
        upload = await message.channel.send(file=file)
        frame_url = upload.attachments[0].url

        # Process the image from Discord URL
        description = await process_image(frame_url)
        frame_descriptions.append(description)

        # After processing the image, delete the uploaded file from the server
        await upload.delete()

        # Now delete the temporary frame file locally
        try:
            os.remove(frame_path)
            print(f"Deleted {frame_path}")
        except PermissionError:
            print(f"Error deleting temp file: {frame_path}")

    # Combine all descriptions into one message
    combined_description = "\n".join(frame_descriptions)

    # Process audio if attached
    audio_transcription = None
    if audio_attachment:
        audio_url = audio_attachment.url
        # Process the audio and get transcription
        audio_transcription = await process_audio(message, audio_url)

    # Call the model with the video description and audio transcription
    model_response = await handle_model_call(video_descritpion=combined_description, audiotranscription=audio_transcription)

    # Send the model's response back to the same Discord channel
    await send_large_message(message.channel, model_response)

    # Return or log the model response
    print(f"Model response: {model_response}")
    return model_response

# Start the bot
bot.run(os.getenv("QB_TOKEN"))
