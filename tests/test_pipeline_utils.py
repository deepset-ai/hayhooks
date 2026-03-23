import pytest

from hayhooks.server.pipelines.utils import get_input_files, get_last_user_input_text


class TestGetLastUserInputText:
    def test_standard_input_text(self):
        items = [
            {
                "role": "user",
                "type": "message",
                "content": [{"type": "input_text", "text": "Hello world"}],
            }
        ]
        assert get_last_user_input_text(items) == "Hello world"

    def test_returns_last_user_message(self):
        items = [
            {
                "role": "user",
                "type": "message",
                "content": [{"type": "input_text", "text": "First"}],
            },
            {
                "role": "assistant",
                "type": "message",
                "content": [{"type": "output_text", "text": "Response"}],
            },
            {
                "role": "user",
                "type": "message",
                "content": [{"type": "input_text", "text": "Second"}],
            },
        ]
        assert get_last_user_input_text(items) == "Second"

    def test_string_content_shorthand(self):
        items = [{"role": "user", "type": "message", "content": "Hello shorthand"}]
        assert get_last_user_input_text(items) == "Hello shorthand"

    def test_type_none_is_accepted(self):
        items = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "No type key"}],
            }
        ]
        assert get_last_user_input_text(items) == "No type key"

    def test_empty_list(self):
        assert get_last_user_input_text([]) is None

    def test_no_user_messages(self):
        items = [
            {
                "role": "assistant",
                "type": "message",
                "content": [{"type": "output_text", "text": "Hi"}],
            }
        ]
        assert get_last_user_input_text(items) is None

    def test_non_dict_items_skipped(self):
        items = ["not a dict", 42, None, {"role": "user", "content": "Valid"}]
        assert get_last_user_input_text(items) == "Valid"

    def test_non_dict_content_parts_skipped(self):
        items = [
            {
                "role": "user",
                "type": "message",
                "content": ["string part", 123, {"type": "input_text", "text": "Real text"}],
            }
        ]
        assert get_last_user_input_text(items) == "Real text"

    def test_function_call_output_only(self):
        items = [
            {
                "role": "user",
                "type": "function_call_output",
                "content": [{"type": "input_text", "text": "Should skip"}],
            }
        ]
        assert get_last_user_input_text(items) is None

    def test_empty_content_list(self):
        items = [{"role": "user", "type": "message", "content": []}]
        assert get_last_user_input_text(items) is None

    def test_multiple_content_parts_returns_last_input_text(self):
        items = [
            {
                "role": "user",
                "type": "message",
                "content": [
                    {"type": "input_text", "text": "First part"},
                    {"type": "input_image", "url": "http://example.com/img.png"},
                    {"type": "input_text", "text": "Last part"},
                ],
            }
        ]
        assert get_last_user_input_text(items) == "Last part"


class TestGetInputFiles:
    def test_extracts_single_file(self):
        items = [
            {
                "role": "user",
                "type": "message",
                "content": [
                    {"type": "input_text", "text": "Summarize"},
                    {"type": "input_file", "file_id": "file-abc123"},
                ],
            }
        ]
        result = get_input_files(items)
        assert len(result) == 1
        assert result[0]["file_id"] == "file-abc123"

    def test_extracts_multiple_files(self):
        items = [
            {
                "role": "user",
                "type": "message",
                "content": [
                    {"type": "input_file", "file_id": "file-aaa"},
                    {"type": "input_file", "file_id": "file-bbb"},
                ],
            }
        ]
        result = get_input_files(items)
        assert [f["file_id"] for f in result] == ["file-aaa", "file-bbb"]

    def test_extracts_from_multiple_user_messages(self):
        items = [
            {
                "role": "user",
                "type": "message",
                "content": [{"type": "input_file", "file_id": "file-1"}],
            },
            {
                "role": "assistant",
                "type": "message",
                "content": [{"type": "output_text", "text": "OK"}],
            },
            {
                "role": "user",
                "type": "message",
                "content": [{"type": "input_file", "file_id": "file-2"}],
            },
        ]
        result = get_input_files(items)
        assert [f["file_id"] for f in result] == ["file-1", "file-2"]

    def test_skips_non_user_roles(self):
        items = [
            {
                "role": "assistant",
                "type": "message",
                "content": [{"type": "input_file", "file_id": "file-nope"}],
            }
        ]
        assert get_input_files(items) == []

    def test_skips_string_content(self):
        items = [{"role": "user", "type": "message", "content": "Just text, no files"}]
        assert get_input_files(items) == []

    def test_empty_list(self):
        assert get_input_files([]) == []

    def test_skips_non_dict_items(self):
        items = ["garbage", 42, {"role": "user", "content": [{"type": "input_file", "file_id": "file-ok"}]}]
        result = get_input_files(items)
        assert len(result) == 1
        assert result[0]["file_id"] == "file-ok"

    def test_skips_non_dict_content_parts(self):
        items = [
            {
                "role": "user",
                "type": "message",
                "content": ["not a dict", {"type": "input_file", "file_id": "file-ok"}],
            }
        ]
        result = get_input_files(items)
        assert len(result) == 1

    def test_skips_input_file_without_file_id(self):
        items = [
            {
                "role": "user",
                "type": "message",
                "content": [{"type": "input_file"}],
            }
        ]
        assert get_input_files(items) == []

    def test_preserves_extra_keys(self):
        items = [
            {
                "role": "user",
                "type": "message",
                "content": [{"type": "input_file", "file_id": "file-x", "filename": "doc.pdf"}],
            }
        ]
        result = get_input_files(items)
        assert result[0]["filename"] == "doc.pdf"
