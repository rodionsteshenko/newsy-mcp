"""
GPT Utility for managing conversations with OpenAI's GPT models.

Features:
1. Asynchronous GPT API interaction with configurable models
2. Message history management with system message validation
3. Tool integration support for extended functionality
4. Debug logging with dprint() for development
5. Message serialization and deserialization
6. Error handling and validation
7. Tool call processing with result management
"""

import os
from tools import call_tool_by_func, Tool
from openai import AsyncOpenAI
import json
from utils import dprint

# Debug flag
DEBUG = True

MODEL = "gpt-4o-mini"
MAX_TOKENS = 8192
SYSTEM_MSG = ""


# Create async client instance
client = AsyncOpenAI(api_key=os.getenv("RODION_OPENAI_API_KEY"))


class GptUtil:
    _instance_counter = 0

    def __init__(
        self,
        system_msg: str = SYSTEM_MSG,
        tools: list[Tool] = None,
        msgs: list[dict] = None,
    ):
        GptUtil._instance_counter += 1
        self.instance_id = GptUtil._instance_counter
        dprint(f"Creating GptUtil instance {self.instance_id}")

        self.system_msg = system_msg
        self.tools = tools if tools else []

        if msgs:
            # Add debug logging for incoming messages
            dprint(f"Initializing with {len(msgs)} messages")
            for i, msg in enumerate(msgs):
                dprint(
                    f"Message {i}: role={msg.get('role')}, content preview={msg.get('content', '')[:50]}..."
                )

            # Validate that msgs starts with system message
            if not msgs or msgs[0]["role"] != "system":
                dprint("Messages array missing system message, adding it")
                msgs.insert(
                    0,
                    {
                        "role": "system",
                        "content": system_msg,
                    },
                )
            self.msgs = msgs.copy()
        else:
            dprint("No messages provided, starting with system message only")
            self.msgs = [
                {
                    "role": "system",
                    "content": self.system_msg,
                }
            ]

        # Validate system message is present
        assert self.msgs[0]["role"] == "system", "System message must be first message"
        dprint(f"Initialized with {len(self.msgs)} total messages")

    def is_modified(self) -> bool:
        return len(self.msgs) > 1

    def update_system_message(self, new_system_msg: str) -> None:
        """Update the system message while preserving conversation history."""
        dprint(f"Updating system message (instance {self.instance_id})")
        dprint(f"Previous message count: {len(self.msgs)}")

        self.system_msg = new_system_msg

        if self.msgs:
            # Keep existing messages but update system message
            self.msgs[0] = {
                "role": "system",
                "content": self.system_msg,
            }
        else:
            # Initialize with just system message if no messages exist
            self.msgs = [
                {
                    "role": "system",
                    "content": self.system_msg,
                },
            ]

        dprint(f"Updated message count: {len(self.msgs)}")

    def update_tools(self, new_tools: list[Tool]) -> None:
        """Update the tools list while preserving conversation history."""
        dprint(f"Updating tools (instance {self.instance_id})")
        dprint(f"Previous tools count: {len(self.tools)}")

        self.tools = new_tools

        dprint(f"Updated tools count: {len(self.tools)}")
        for i, tool in enumerate(self.tools):
            dprint(f"Tool {i}: {tool.name}")

    def to_dict(self):
        return {
            "system_msg": self.system_msg,
            "tools": [tool.to_dict() for tool in self.tools],
            "msgs": self.msgs,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["system_msg"],
            [Tool.from_dict(tool) for tool in data["tools"]],
            data["msgs"],
        )

    async def send_prompt(self, prompt: str) -> str:
        # Validate system message is still present
        if not self.msgs or self.msgs[0]["role"] != "system":
            dprint("Error: System message missing before prompt, restoring it")
            self.msgs.insert(
                0,
                {
                    "role": "system",
                    "content": self.system_msg,
                },
            )

        if prompt:
            self.msgs.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

        # Add debug logging for message state
        dprint("Messages before API call:")
        for i, msg in enumerate(self.msgs):
            dprint(
                f"  Message {i}: role={msg.get('role')}, content preview={msg.get('content', '')[:50]}..."
            )

        dprint(f"Sending messages (count: {len(self.msgs)})")

        async def _send_prompt_helper() -> str:
            while True:  # Keep going until we get content
                try:
                    if self.tools:
                        response = await client.chat.completions.create(
                            model=MODEL,
                            messages=self.msgs,
                            tools=[t.schema for t in self.tools],
                            tool_choice="auto",
                        )
                    else:
                        response = await client.chat.completions.create(
                            model=MODEL,
                            messages=self.msgs,
                        )

                    message = response.choices[0].message

                    # Handle content if present
                    if content := message.content:
                        self.msgs.append(
                            {
                                "role": "assistant",
                                "content": content,
                            }
                        )
                        # Add debug logging for final message state
                        dprint(f"Final message count after response: {len(self.msgs)}")
                        dprint(f"Final message: {content}")
                        return content

                    # Handle tool calls
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        dprint("Tool calls detected")
                        for tool_call in message.tool_calls:
                            tool_name = tool_call.function.name
                            tool_args = json.loads(tool_call.function.arguments)
                            dprint(f"Processing tool: {tool_name}")
                            dprint(f"Tool arguments: {tool_args}")

                            result = None
                            for tool in self.tools:
                                if tool.name == tool_name:
                                    try:
                                        result = await call_tool_by_func(
                                            tool.func, tool_args
                                        )
                                        dprint(f"Tool result: {result}")
                                        if result is None:
                                            result = f"Error: {tool_name} returned None"
                                    except Exception as e:
                                        result = (
                                            f"Error executing {tool_name}: {str(e)}"
                                        )
                                        dprint(f"Tool error: {result}", error=True)

                            self.msgs.append(
                                {
                                    "role": "assistant",
                                    "tool_calls": [
                                        {
                                            "id": tool_call.id,
                                            "function": {
                                                "name": tool_call.function.name,
                                                "arguments": tool_call.function.arguments,
                                            },
                                            "type": "function",
                                        }
                                    ],
                                }
                            )

                            # Convert result to string if it's a dictionary
                            result_content = (
                                json.dumps(result)
                                if isinstance(result, dict)
                                else str(result)
                            )

                            self.msgs.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "name": tool_call.function.name,
                                    "content": result_content,
                                }
                            )

                        # Continue the loop to get content after tool calls

                except Exception as e:
                    print(f"Error in GPT request: {str(e)}")
                    raise e

        return await _send_prompt_helper()
