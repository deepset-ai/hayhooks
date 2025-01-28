from typing import List, Union, Dict
from hayhooks.server.routers.openai import Message


def is_user_message(msg: Union[Message, Dict]) -> bool:
    if isinstance(msg, Message):
        return msg.role == "user"
    return msg.get("role") == "user"


def get_content(msg: Union[Message, Dict]) -> str:
    if isinstance(msg, Message):
        return msg.content
    return msg.get("content")


def get_last_user_message(messages: List[Union[Message, Dict]]) -> Union[str, None]:
    user_messages = (msg for msg in reversed(messages) if is_user_message(msg))

    for message in user_messages:
        return get_content(message)

    return None
