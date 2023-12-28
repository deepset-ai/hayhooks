from haystack.core.component import component


@component
class Hello:
    @component.output_types(output=str)
    def run(self, word: str = "world"):
        """
        Takes a string in input and returns "Hello, <string>!"
        in output.
        """
        return {"output": f"Hello, {word}!"}
