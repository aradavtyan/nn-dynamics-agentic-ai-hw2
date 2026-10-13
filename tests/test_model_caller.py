import os
import unittest
from unittest.mock import Mock, patch

from task6.agentic_system import ModelCaller


class TestModelCaller(unittest.TestCase):
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-only-key"}, clear=True)
    def test_openrouter_endpoint_and_messages(self):
        with patch("task6.agentic_system.OpenAI") as factory:
            factory.return_value.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content="  result  "))]
            )
            caller = ModelCaller("provider/model")
            self.assertEqual(caller.call("instructions", "request"), "result")
            factory.assert_called_once_with(api_key="test-only-key", base_url="https://openrouter.ai/api/v1")
            factory.return_value.chat.completions.create.assert_called_once_with(
                model="provider/model",
                messages=[{"role": "system", "content": "instructions"}, {"role": "user", "content": "request"}],
            )
            factory.return_value.responses.create.assert_not_called()

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-only-key"}, clear=True)
    def test_openrouter_empty_response_and_invalid_model(self):
        with patch("task6.agentic_system.OpenAI") as factory:
            with self.assertRaises(ValueError):
                ModelCaller("missing-provider")
            factory.assert_not_called()
            factory.return_value.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content=None))]
            )
            with self.assertRaisesRegex(ValueError, "no text"):
                ModelCaller("provider/model").call("instructions", "request")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-key"}, clear=True)
    def test_openai_still_uses_responses(self):
        with patch("task6.agentic_system.OpenAI") as factory:
            factory.return_value.responses.create.return_value = Mock(output_text=" result ")
            self.assertEqual(ModelCaller("test-model").call("instructions", "request"), "result")
            factory.return_value.responses.create.assert_called_once_with(
                model="test-model", instructions="instructions", input="request"
            )
            factory.return_value.chat.completions.create.assert_not_called()
