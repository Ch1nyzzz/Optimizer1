"""mini-SWE-agent model adapter for Qwen3.5 reasoning responses."""

from __future__ import annotations

from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel


class Qwen35ToolModel(LitellmModel):
    """Drop vLLM/OpenAI `reasoning` before replaying assistant history."""

    def query(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        message = super().query(messages, **kwargs)
        message.pop("reasoning", None)
        message.pop("reasoning_content", None)
        return message


class Qwen35TextModel(LitellmTextbasedModel):
    """Text-action variant that also drops vLLM/OpenAI `reasoning` history."""

    def query(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        message = super().query(messages, **kwargs)
        message.pop("reasoning", None)
        message.pop("reasoning_content", None)
        return message
