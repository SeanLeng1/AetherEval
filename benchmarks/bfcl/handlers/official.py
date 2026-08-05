"""Transport adapter for BFCL V3's official prompt-mode handlers."""

from .common import query_rendered_prompt


class OfficialPromptHandlerAdapter:
    """Preserve an upstream BFCL prompt/decoder and replace its transport.

    The registry uses this as the first base of a runtime subclass whose second
    base is the model-specific official handler.
    """

    def _query_prompting(self, inference_data: dict):
        formatted_prompt = self._format_prompt(
            inference_data["message"],
            inference_data["function"],
        )
        inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}
        return query_rendered_prompt(
            self,
            formatted_prompt,
            skip_special_tokens=getattr(self, "skip_special_tokens", None),
        )
