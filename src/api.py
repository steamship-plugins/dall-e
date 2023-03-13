import logging
from typing import Any, Dict, List, Optional, Type
import json
import openai
from pydantic import Field
from steamship import Steamship, Block, MimeTypes
from steamship.data.block import BlockUploadType
from steamship.invocable import Config, InvocableResponse, InvocationContext
from steamship.plugin.generator import Generator
from steamship.plugin.inputs.raw_block_and_tag_plugin_input import RawBlockAndTagPluginInput
from steamship.plugin.outputs.raw_block_and_tag_plugin_output import RawBlockAndTagPluginOutput
from steamship.plugin.request import PluginRequest
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    before_sleep_log, wait_exponential_jitter,
)



class DallEPlugin(Generator):
    """
    Plugin for generating images from text prompts from DALL-E.
    """

    class DallEPluginConfig(Config):
        openai_api_key: str = Field("",
                                    description="An openAI API key to use. If left default, will use Steamship's API key.")
        n: int = Field(1, description="How many images to generate for each prompt.")
        size: str = Field("1024x1024", description="The size of the ouptut images.  May be \"1024x1024\", \"512x512\", or \"256x256\".")
        max_retries: int = Field(8, description="Maximum number of retries to make when generating.")
        request_timeout: float = Field(600,
                                                 description="Timeout for requests to OpenAI completion API. Default is 600 seconds.")

    @classmethod
    def config_cls(cls) -> Type[Config]:
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



    def generate_with_retry(self, user: str, prompt: str) -> List[Block]:
        """Use tenacity to retry the completion call."""

        logging.info(f"Making OpenAI dall-e call on behalf of user with id: {user}")
        """Call the API to generate the next section of text."""

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
        def _generate_with_retry(prompt: str) -> Any:
            return openai.Image.create(prompt=prompt, n=self.config.n, size=self.config.size)

        openai_result =  _generate_with_retry(prompt=prompt)
        logging.info("Retry statistics: " + json.dumps(_generate_with_retry.retry.statistics))

        # Fetch data for images
        urls = [obj['url'] for obj in openai_result['data']]

        return [Block(url=url, mime_type=MimeTypes.PNG, upload_type=BlockUploadType.URL) for url in urls]



    def run(
            self, request: PluginRequest[RawBlockAndTagPluginInput]
    ) -> InvocableResponse[RawBlockAndTagPluginOutput]:
        """Run the image generator against all the text, combined """

        prompt = " ".join([block.text for block in request.data.blocks if block.text is not None])
        user_id = self.context.user_id if self.context is not None else "testing"
        generated_blocks = self.generate_with_retry(prompt=prompt, user=user_id)

        return InvocableResponse(data=RawBlockAndTagPluginOutput(blocks= generated_blocks))
