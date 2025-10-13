"""
Pipeline wrapper with UTF-8 characters for testing.
测试 UTF-8 字符支持 🌍 こんにちは мир
"""

from haystack import Pipeline

from hayhooks import BasePipelineWrapper


class PipelineWrapper(BasePipelineWrapper):
    """
    Wrapper class with UTF-8 characters in docstring.
    Unicode test: 你好世界 🚀 привет Δοκιμή
    """

    def setup(self):
        """Initialize the pipeline with UTF-8 characters: 测试 🎉"""
        self.pipeline = Pipeline()
        # Comment with UTF-8: こんにちは мир 你好
        self.greeting = "Hello 世界 🌍"

    def run_api(self, test_param: str) -> dict:
        """
        Run the API with UTF-8 support.

        Args:
            test_param: Test parameter (测试参数 🔧)

        Returns:
            Result with UTF-8 characters (返回 UTF-8 字符 ✨)
        """
        # Return message with UTF-8 characters
        return {
            "result": f"Response: {test_param}",
            "greeting": self.greeting,
            "message": "UTF-8 test: 你好世界 🌍 こんにちは мир Δοκιμή",
        }
