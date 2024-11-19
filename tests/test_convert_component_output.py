from hayhooks.server.pipelines.models import convert_component_output
from openai.types.completion_usage import CompletionTokensDetails, PromptTokensDetails


def test_convert_component_output_with_nested_models():
    sample_response = [
        {
            'model': 'gpt-4o-mini-2024-07-18',
            'index': 0,
            'finish_reason': 'stop',
            'usage': {
                'completion_tokens': 52,
                'prompt_tokens': 29,
                'total_tokens': 81,
                'completion_tokens_details': CompletionTokensDetails(
                    accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0
                ),
                'prompt_tokens_details': PromptTokensDetails(audio_tokens=0, cached_tokens=0),
            },
        }
    ]

    converted_output = convert_component_output(sample_response)

    assert converted_output == [
        {
            'model': 'gpt-4o-mini-2024-07-18',
            'index': 0,
            'finish_reason': 'stop',
            'usage': {
                'completion_tokens': 52,
                'prompt_tokens': 29,
                'total_tokens': 81,
                'completion_tokens_details': {
                    'accepted_prediction_tokens': 0,
                    'audio_tokens': 0,
                    'reasoning_tokens': 0,
                    'rejected_prediction_tokens': 0,
                },
                'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0},
            },
        }
    ]
