#! /usr/bin/env python3

"""
Telegram Ideation Bot with GPT Integration and Image Generation

Features:
1. Message Processing
   - Authorized user verification via ALLOWED_USER_ID
   - Text message handling with GPT response generation
   - Message history tracking and persistence
   - URL detection and filtering
   - Word limit enforcement for responses
   - Processing status indicators

2. GPT Integration
   - Conversation history management
   - System message configuration for chat personality
   - Concise response generation with word limits
   - Follow-up message generation
   - Image prompt generation

3. Image Generation
   - DALL-E integration via ImageUtil
   - Art style preferences configuration
   - Square format image generation
   - Error handling for content policy violations
   - Image caption generation from prompts

4. Follow-up System
   - Random interval follow-up scheduling
   - Task management per chat
   - Previous context consideration
   - Task cancellation on new messages
   - Configurable timing windows

5. History Management
   - JSON-based conversation persistence
   - Timestamp tracking
   - Message and response pairing
   - Error handling for file operations
   - History loading on startup

6. Error Handling
   - Graceful error reporting to users
   - Content policy violation handling
   - File operation error management
   - Invalid message type handling
   - Debug logging support
"""

import json
import os
import re
import random
from telegram import Update
from telegram.ext import (
    filters,
    MessageHandler,
    ApplicationBuilder,
    ContextTypes,
)
from gpt_util import GptUtil
from image_util import ImageSize, ImageUtil
from utils import dprint


TOKEN = "7538487603:AAF1ol3EWPhjvDRsaaNvQpDM2tB9G6jTOAI"

# Only allow messages from this user ID
ALLOWED_USER_ID = 7816164768

REPLY_LENGTH_WORD_LIMIT = 50

# Reply timing constants (in seconds)
MIN_REPLY_TIME_S = 30 * 60
MAX_REPLY_TIME_S = 60 * 60

# Art style preferences for image generation
IMAGE_SYSTEM_MSG = """You are an image generation AI. 
- Be creative, don't be afraid to break the rules
- Consider the following art styles when creating prompts:
    - Impressionism and watercolor styles
    - Dreamlike landscapes with a whimsical touch
    - Minimalist compositions with soft, harmonious colors
    - Traditional Japanese art influences
    - Ethereal and fantastical elements without heavy digital influence
    - Organic flowing lines reminiscent of art nouveau
    - Light and airy scenes that evoke a sense of wonder
    - Subtle textures and layers that enhance visual storytelling
"""

SYSTEM_MSG = f"""You are a thoughtful AI assistant who helps explore and expand on ideas. 
- Keep responses concise (< {REPLY_LENGTH_WORD_LIMIT} words) unless specifically asked to expand
- This is a chatbot, so respond in a conversational manner and short with little markdown, lists, etc.

In your responses:
- Make insightful connections and extensions to topics brought up
- Help textualize and clarify thoughts and concepts
- Engage in pondering life, the universe, and philosophical questions
- Reference and build upon previous conversations when relevant
- Be conversational and engaging while remaining focused and helpful
- Keep emoji usage minimal but effective
"""

# File to store conversation history
HISTORY_FILE = "ideator_history.json"

# Track last message timestamp per chat
last_message_time = {}
follow_up_tasks = {}


def load_history() -> list:
    """Load conversation history from JSON file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
                dprint(f"Loaded history with {len(history)} entries")
                return history
        except Exception as e:
            print(f"Error loading history: {e}")
            return []
    dprint("No history file found")
    return []


# Load history and initialize GptUtil with conversation history
history = load_history()
messages = []
for conv in history:
    messages.append({"role": "user", "content": conv["message"]})
    messages.append({"role": "assistant", "content": conv["response"]})
gpt_util = GptUtil(system_msg=SYSTEM_MSG, msgs=messages)


def save_history(history: list) -> None:
    """Save conversation history to JSON file"""
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
            dprint(f"Saved {len(history)} history entries")
    except Exception as e:
        print(f"Error saving history: {e}")


def is_url(text: str) -> bool:
    """Check if text is a URL"""
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    return bool(re.match(url_pattern, text))


async def send_follow_up(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a follow-up message after random interval"""
    job = context.job
    chat_id = int(job.chat_id)
    last_message = job.data

    dprint(f"Starting follow-up timer for chat {chat_id}")

    # Check if this task is still valid
    if chat_id not in follow_up_tasks or follow_up_tasks[chat_id] != job:
        dprint(f"Follow-up task no longer valid for chat {chat_id}")
        return

    dprint(f"Generating follow-up response for chat {chat_id}")
    # Generate follow-up response
    prompt = f"""Based on this previous message, tell me more interesting related thoughts and insights:

{last_message}
"""
    follow_up = gpt_util.send_prompt(prompt)

    try:
        dprint(f"Sending follow-up message to chat {chat_id}")
        await context.bot.send_message(chat_id=chat_id, text=follow_up)

        dprint(f"Successfully sent follow-up message to chat {chat_id}")

        # Schedule next follow-up
        delay = random.randint(MIN_REPLY_TIME_S, MAX_REPLY_TIME_S)
        job = context.job_queue.run_once(
            send_follow_up,
            delay,
            chat_id=str(chat_id),
            data=last_message,
        )
        follow_up_tasks[chat_id] = job

    except Exception as e:
        print(f"Error sending follow-up: {e}")


async def ideate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process messages for ideation"""
    try:
        # Check if user is allowed
        if update.effective_user.id != ALLOWED_USER_ID:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Sorry, you are not authorized to use this bot. 🚫",
                reply_to_message_id=update.message.message_id,
            )
            return

        if not update.message or not update.message.text:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Sorry, I can only process text messages! 📜",
                reply_to_message_id=update.message.message_id,
            )
            return

        message = update.message.text
        chat_id = update.effective_chat.id
        message_id = update.message.message_id

        # Cancel any existing follow-up task for this chat
        if chat_id in follow_up_tasks:
            dprint(f"Cancelling existing follow-up task for chat {chat_id}")
            follow_up_tasks[chat_id].schedule_removal()
            del follow_up_tasks[chat_id]

        # Ignore if message is URL
        if is_url(message):
            return

        # Send temporary message placeholder
        processing_msg = await context.bot.send_message(
            chat_id=chat_id,
            text="Hmm  ...",
            reply_to_message_id=message_id,
        )

        full_prompt = f"{message}\n\nRemember to keep your reply to {REPLY_LENGTH_WORD_LIMIT} words unless you specified otherwise above."

        response = gpt_util.send_prompt(full_prompt)

        # Delete processing message
        await processing_msg.delete()

        # Save conversation to history
        history.append(
            {
                "message": message,
                "response": response,
                "timestamp": update.message.date.isoformat(),
            }
        )
        save_history(history)

        # Send the response
        await context.bot.send_message(
            chat_id=chat_id,
            text=response,
            reply_to_message_id=message_id,
        )

        # Generate image prompt using a new GptUtil instance
        image_gpt_util = GptUtil(system_msg=IMAGE_SYSTEM_MSG, msgs=[])
        image_prompt_template = f"""Create an image generation prompt (< 50 words) that captures the essence of this idea in one of these styles (don't say prompt, just give the prompt):

{message}"""

        dprint(f"Generating image with prompt template:\n{image_prompt_template}")
        image_prompt = image_gpt_util.send_prompt(f"{message}\n{image_prompt_template}")
        dprint(f"Generated image prompt:\n{image_prompt}")

        try:
            # Generate and send image
            image_util = ImageUtil()
            image_paths = image_util.send_prompt(
                image_prompt, count=1, size=ImageSize.SQUARE
            )
            if image_paths:
                with open(image_paths[0], "rb") as image:
                    await context.bot.send_photo(
                        chat_id=chat_id,
                        photo=image,
                        caption=f'"{image_prompt}"',
                    )
        except Exception as img_error:
            error_msg = str(img_error)
            if "content_policy_violation" in error_msg:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="Sorry, I couldn't generate an image for this prompt due to content policy restrictions. 🚫",
                    reply_to_message_id=message_id,
                )
            else:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"Sorry, there was an error generating the image: {error_msg} ⚠️",
                    reply_to_message_id=message_id,
                )
            print(f"Error generating image: {error_msg}")

        # Schedule follow-up with random delay
        if context.job_queue:
            dprint(f"Scheduling follow-up task for chat {chat_id}")
            delay = random.randint(MIN_REPLY_TIME_S, MAX_REPLY_TIME_S)
            job = context.job_queue.run_once(
                send_follow_up,
                delay,
                chat_id=str(chat_id),
                data=message,
            )
            follow_up_tasks[chat_id] = job
            dprint(f"Successfully scheduled follow-up task for chat {chat_id}")

    except Exception as e:
        print(f"Error processing message: {e}")
        if chat_id:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Error processing message: {str(e)}",
                reply_to_message_id=message_id if message_id else None,
            )


if __name__ == "__main__":
    application = ApplicationBuilder().token(TOKEN).build()

    ideate_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), ideate)
    application.add_handler(ideate_handler)

    application.run_polling()
