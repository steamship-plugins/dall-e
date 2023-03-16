"""Generator plugin for DALL-E."""
import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Type

import openai
from pydantic import Field
from steamship import Block, MimeTypes, Steamship
from steamship.data.block import BlockUploadType
from steamship.invocable import Config, InvocableResponse, InvocationContext
from steamship.plugin.generator import Generator
from steamship.plugin.inputs.raw_block_and_tag_plugin_input import RawBlockAndTagPluginInput
from steamship.plugin.outputs.raw_block_and_tag_plugin_output import RawBlockAndTagPluginOutput
from steamship.plugin.request import PluginRequest
from tenacity import after_log, before_sleep_log, retry, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_exponential_jitter


class ImageSizeEnum(str, Enum):
    """Supported Image Sizes for generation."""

    large = "1024x1024"
    medium = "512x512"
    small = "256x256"

    @classmethod
    def list(cls):
        """List all supported image sizes."""
        return list(map(lambda c: c.value, cls))


class DallEPlugin(Generator):
    """Plugin for generating images from text prompts from DALL-E."""

    class DallEPluginConfig(Config):
        """Configuration for the DALL-E Plugin."""

        openai_api_key: str = Field(
            "",
            description="An openAI API key to use. If left default, will use Steamship's API "
            "key.",
        )
        n: int = Field(
            1, gt=0, lt=11, description="Default number of images to generate for each prompt."
        )
        size: ImageSizeEnum = Field(
            "1024x1024",
            description="Default size of the output images. Must be one of:"
            f"{ImageSizeEnum.list()}.",
        )
        max_retries: int = Field(
            8, gte=0, lt=16, description="Maximum number of retries to make when generating."
        )
        request_timeout: float = Field(
            600,
            description="Timeout for requests to OpenAI completion API. Default is 600 " "seconds.",
        )

    @classmethod
    def config_cls(cls) -> Type[Config]:
        """Return configuration template for the generator."""
        return cls.DallEPluginConfig

    config: DallEPluginConfig

    def __init__(
        self,
        client: Steamship = None,
        config: Dict[str, Any] = None,
        context: InvocationContext = None,
    ):
        super().__init__(client, config, context)
        openai.api_key = self.config.openai_api_key

    def generate_with_retry(
        self, user: str, prompt: str, options: Optional[dict] = None
    ) -> List[Block]:
        """Use tenacity to retry the completion call."""
        logging.info(f"Making OpenAI dall-e call on behalf of user with id: {user}")

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential_jitter(jitter=5),
            before_sleep=before_sleep_log(logging.root, logging.INFO),
            retry=(
                retry_if_exception_type(openai.error.Timeout)
                | retry_if_exception_type(openai.error.APIError)
                | retry_if_exception_type(openai.error.APIConnectionError)
                | retry_if_exception_type(openai.error.RateLimitError)
            ),
            after=after_log(logging.root, logging.INFO),
        )
        def _generate_with_retry(prompt: str, n: int, size: str) -> Any:
            return openai.Image.create(prompt=prompt, n=n, size=size)

        n = options.get("n", self.config.n) if options else self.config.n
        size = options.get("size", self.config.size) if options else self.config.size

        if n > 10 or n < 1:
            raise ValueError(f"'n' must be a number between 1 and 10 (received: {n})")

        if size not in ImageSizeEnum.list():
            raise ValueError(f"'size' must be one of {ImageSizeEnum.list()} (received: {size})")

        openai_result = _generate_with_retry(prompt=prompt, n=n, size=size)
        logging.info("Retry statistics: " + json.dumps(_generate_with_retry.retry.statistics))

        # Fetch data for images
        urls = [obj["url"] for obj in openai_result["data"]]

        return [
            Block(url=url, mime_type=MimeTypes.PNG, upload_type=BlockUploadType.URL) for url in urls
        ]

    def run(
        self, request: PluginRequest[RawBlockAndTagPluginInput]
    ) -> InvocableResponse[RawBlockAndTagPluginOutput]:
        """Run the image generator against all the text, combined."""
        prompt = " ".join([block.text for block in request.data.blocks if block.text is not None])
        user_id = self.context.user_id if self.context is not None else "testing"
        generated_blocks = self.generate_with_retry(
            prompt=prompt, user=user_id, options=request.data.options
        )

        return InvocableResponse(data=RawBlockAndTagPluginOutput(blocks=generated_blocks))
