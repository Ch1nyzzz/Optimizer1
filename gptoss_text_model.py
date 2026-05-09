"""mini-SWE-agent model adapter for gpt-oss responses with harmony tags."""

from __future__ import annotations

import re

from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel


class GptOssTextModel(LitellmTextbasedModel):
    """Strip gpt-oss harmony control tokens before replaying chat history."""

    def query(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        message = super().query(messages, **kwargs)
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = re.sub(r"<\|[^|]+?\|>", "", content)
        return message
