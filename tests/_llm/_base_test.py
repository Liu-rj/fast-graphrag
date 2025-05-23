# type: ignore
import unittest
from unittest.mock import AsyncMock, patch

from fast_graphrag._llm._base import BaseLLMService, format_and_send_prompt

# Assuming these are defined somewhere in your codebase
PROMPTS = {
    "example_prompt": "Hello, {name}!"
}

class TestFormatAndSendPrompt(unittest.IsolatedAsyncioTestCase):

    @patch("fast_graphrag._llm._base.PROMPTS", PROMPTS)
    async def test_format_and_send_prompt(self):
        mock_llm = AsyncMock(spec=BaseLLMService())
        mock_response = (str(), [{"key": "value"}])
        mock_llm.send_message = AsyncMock(return_value=mock_response)

        result = await format_and_send_prompt(
            prompt_key="example_prompt",
            llm=mock_llm,
            format_kwargs={"name": "World"},
            response_model=str
        )

        mock_llm.send_message.assert_called_once_with(
            prompt="Hello, World!",
            response_model=str
        )
        self.assertEqual(result, mock_response)

    @patch("fast_graphrag._llm._base.PROMPTS", PROMPTS)
    async def test_format_and_send_prompt_with_additional_args(self):
        mock_llm = AsyncMock(spec=BaseLLMService())
        mock_response = (str(), [{"key": "value"}])
        mock_llm.send_message = AsyncMock(return_value=mock_response)

        result = await format_and_send_prompt(
            prompt_key="example_prompt",
            llm=mock_llm,
            format_kwargs={"name": "World"},
            response_model=str,
            model="test_model",
            max_tokens=100
        )

        mock_llm.send_message.assert_called_once_with(
            prompt="Hello, World!",
            response_model=str,
            model="test_model",
            max_tokens=100
        )
        self.assertEqual(result, mock_response)

if __name__ == "__main__":
    unittest.main()
