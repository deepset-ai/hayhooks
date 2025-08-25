import pytest
from pydantic import ValidationError

from hayhooks.open_webui import (
    MessageEventData,
    NotificationEventData,
    OpenWebUIEvent,
    StatusEventData,
    create_chat_completion_event,
    create_details_tag,
    create_message_event,
    create_notification_event,
    create_replace_event,
    create_source_event,
    create_status_event,
)


class TestStatusEventData:
    def test_status_event_data_creation(self):
        data = StatusEventData(description="Processing request")
        assert data.description == "Processing request"
        assert data.done is False
        assert data.hidden is False

    def test_status_event_data_with_optional_params(self):
        data = StatusEventData(description="Task completed", done=True, hidden=True)
        assert data.description == "Task completed"
        assert data.done is True
        assert data.hidden is True

    def test_status_event_data_validation_error(self):
        with pytest.raises(ValidationError):
            StatusEventData()  # missing required description


class TestMessageEventData:
    def test_message_event_data_creation(self):
        data = MessageEventData(content="Hello, world!")
        assert data.content == "Hello, world!"

    def test_message_event_data_empty_content(self):
        data = MessageEventData(content="")
        assert data.content == ""

    def test_message_event_data_validation_error(self):
        with pytest.raises(ValidationError):
            MessageEventData()  # missing required content


class TestNotificationEventData:
    def test_notification_event_data_default_type(self):
        data = NotificationEventData(content="System notification")
        assert data.content == "System notification"
        assert data.type == "info"

    def test_notification_event_data_all_types(self):
        for notification_type in ["info", "success", "warning", "error"]:
            data = NotificationEventData(content="Test message", type=notification_type)
            assert data.type == notification_type

    def test_notification_event_data_validation_error_invalid_type(self):
        with pytest.raises(ValidationError):
            NotificationEventData(content="Test", type="invalid")

    def test_notification_event_data_validation_error_missing_content(self):
        with pytest.raises(ValidationError):
            NotificationEventData()  # missing required content


class TestOpenWebUIEvent:
    def test_open_webui_event_with_status_data(self):
        status_data = StatusEventData(description="Loading")
        event = OpenWebUIEvent(type="status", data=status_data)
        assert event.type == "status"
        assert isinstance(event.data, StatusEventData)
        assert event.data.description == "Loading"

    def test_open_webui_event_with_message_data(self):
        message_data = MessageEventData(content="Hello")
        event = OpenWebUIEvent(type="message", data=message_data)
        assert event.type == "message"
        assert isinstance(event.data, MessageEventData)
        assert event.data.content == "Hello"

    def test_open_webui_event_with_dict_data(self):
        custom_data = {"custom_field": "custom_value", "number": 42}
        event = OpenWebUIEvent(type="custom", data=custom_data)
        assert event.type == "custom"
        assert event.data == custom_data

    def test_open_webui_event_to_dict(self):
        status_data = StatusEventData(description="Processing", done=True)
        event = OpenWebUIEvent(type="status", data=status_data)

        result = event.to_dict()
        expected = {"type": "status", "data": {"description": "Processing", "done": True, "hidden": False}}
        assert result == expected

    def test_open_webui_event_to_dict_with_custom_data(self):
        custom_data = {"key": "value", "items": [1, 2, 3]}
        event = OpenWebUIEvent(type="custom", data=custom_data)

        result = event.to_dict()
        expected = {"type": "custom", "data": custom_data}
        assert result == expected


class TestFactoryFunctions:
    def test_create_status_event_default_params(self):
        event = create_status_event("Processing request")
        assert event.type == "status"
        assert isinstance(event.data, StatusEventData)
        assert event.data.description == "Processing request"
        assert event.data.done is False
        assert event.data.hidden is False

    def test_create_status_event_with_all_params(self):
        event = create_status_event("Task completed", done=True, hidden=True)
        assert event.type == "status"
        assert event.data.description == "Task completed"
        assert event.data.done is True
        assert event.data.hidden is True

    def test_create_chat_completion_event(self):
        completion_data = {"id": "123", "model": "gpt-4", "usage": {"tokens": 100}}
        event = create_chat_completion_event(completion_data)
        assert event.type == "chat:completion"
        assert event.data == completion_data

    def test_create_message_event(self):
        event = create_message_event("Message content")
        assert event.type == "message"
        assert isinstance(event.data, MessageEventData)
        assert event.data.content == "Message content"

    def test_create_replace_event(self):
        event = create_replace_event("Replace with this")
        assert event.type == "replace"
        assert isinstance(event.data, MessageEventData)
        assert event.data.content == "Replace with this"

    def test_create_source_event(self):
        source_data = {"url": "https://example.com", "metadata": "source metadata"}
        event = create_source_event(source_data)
        assert event.type == "source"
        assert event.data == source_data

    def test_create_notification_event_default_type(self):
        event = create_notification_event("Default notification")
        assert event.type == "notification"
        assert isinstance(event.data, NotificationEventData)
        assert event.data.content == "Default notification"
        assert event.data.type == "info"

    def test_create_notification_event_all_types(self):
        test_cases = [
            ("info", "Info message"),
            ("success", "Success message"),
            ("warning", "Warning message"),
            ("error", "Error message"),
        ]

        for notification_type, content in test_cases:
            event = create_notification_event(content, notification_type)
            assert event.type == "notification"
            assert isinstance(event.data, NotificationEventData)
            assert event.data.content == content
            assert event.data.type == notification_type

    def test_create_details_tag(self):
        assert create_details_tag(tool_name="test_tool", summary="Test Summary", content="Test Content") == (
            '<details type="test_tool" done="true">\n<summary>Test Summary</summary>\n\nTest Content\n</details>\n\n'
        )


class TestEdgeCases:
    def test_empty_strings(self):
        message_event = create_message_event("")
        assert message_event.data.content == ""

    def test_unicode_content(self):
        unicode_content = "Hello 🌍! Special chars: åæø, 中文, العربية"
        message_event = create_message_event(unicode_content)
        assert message_event.data.content == unicode_content

    def test_nested_data_structures(self):
        complex_data = {
            "metadata": {
                "timestamp": "2024-01-01T00:00:00Z",
                "user": {"id": 123, "name": "Test User"},
                "nested": {"deep": {"value": [1, 2, {"key": "value"}]}},
            }
        }
        event = create_source_event(complex_data)
        assert event.data == complex_data

        serialized = event.to_dict()
        assert serialized["data"] == complex_data

    def test_serialization_consistency(self):
        status_event = create_status_event("Test", done=True, hidden=False)

        dict1 = status_event.to_dict()
        dict2 = status_event.to_dict()

        assert dict1 == dict2
        assert dict1 == {"type": "status", "data": {"description": "Test", "done": True, "hidden": False}}
