"""Streaming Generator plugin for DALL-E."""
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Type

import openai
import requests
from openai import OpenAI
from pydantic import Field, ValidationError, validator
from steamship import MimeTypes, Steamship, SteamshipError
from steamship.invocable import Config, InvocableResponse, InvocationContext
from steamship.plugin.inputs.raw_block_and_tag_plugin_input import RawBlockAndTagPluginInput
from steamship.plugin.inputs.raw_block_and_tag_plugin_input_with_preallocated_blocks import (
    RawBlockAndTagPluginInputWithPreallocatedBlocks,
)
from steamship.plugin.outputs.block_type_plugin_output import BlockTypePluginOutput
from steamship.plugin.outputs.plugin_output import OperationType, OperationUnit, UsageReport
from steamship.plugin.outputs.raw_block_and_tag_plugin_output import RawBlockAndTagPluginOutput
from steamship.plugin.outputs.stream_complete_plugin_output import StreamCompletePluginOutput
from steamship.plugin.request import PluginRequest
from steamship.plugin.streaming_generator import StreamingGenerator


class ModelEnum(str, Enum):
    """Supported Models."""

    DALLE2 = "dall-e-2"
    DALLE3 = "dall-e-3"

    @classmethod
    def list(cls):
        """List all supported image sizes."""
        return list(map(lambda c: c.value, cls))


class ImageSizeEnum(str, Enum):
    """Supported Image Sizes for generation."""

    large = "1024x1024"
    medium = "512x512"  # dall-e-2 only
    small = "256x256"  # dall-e-2 only
    large_landscape = "1792x1024"  # dall-e-3 only
    large_portrait = "1024x1792"  # dall-e-3 only

    @classmethod
    def list(cls):
        """List all supported image sizes."""
        return list(map(lambda c: c.value, cls))


class QualityEnum(str, Enum):
    """Supported Quality for Generation (dall-e-3 only)."""

    standard = "standard"
    hd = "hd"

    @classmethod
    def list(cls):
        """List all supported image sizes."""
        return list(map(lambda c: c.value, cls))


class StyleEnum(str, Enum):
    """Supported style for Generation (dall-e-3 only)."""

    vivid = "vivid"
    natural = "natural"

    @classmethod
    def list(cls):
        """List all supported image sizes."""
        return list(map(lambda c: c.value, cls))


class DallEPlugin(StreamingGenerator):
    """Plugin for generating images from text prompts from DALL-E."""

    class DallEPluginConfig(Config):
        """Configuration for the DALL-E Plugin."""

        openai_api_key: str = Field(
            "",
            description="An openAI API key to use. If left default, will use Steamship's API "
            "key.",
        )
        model: str = Field(
            default="dall-e-2",
            description=f"Model to use for image generation. Must be one of: {ModelEnum.list()}. Not available for runtime override.",
        )
        n: int = Field(
            1, gt=0, lt=11, description="Default number of images to generate for each prompt."
        )
        size: ImageSizeEnum = Field(
            "1024x1024",
            description="Size of the output images. Must be one of:"
            f"{ImageSizeEnum.list()}. Not available for runtime override.",
        )
        max_retries: int = Field(
            8, gte=0, lt=16, description="Maximum number of retries to make when generating."
        )
        request_timeout: float = Field(
            600,
            description="Timeout for requests to OpenAI completion API. Default is 600 " "seconds.",
        )
        quality: str = Field(
            default=QualityEnum.standard.value,
            description=f"The quality of the image that will be generated. Must be one of: {QualityEnum.list()}."
            "'hd' creates images with finer details and greater consistency across the image. "
            "This param is only supported for `dall-e-3`.",
        )
        style: str = Field(
            default=StyleEnum.vivid.value,
            description=f"The style of the generated images. Must be one of: {StyleEnum.list()}. "
            "Vivid causes the model to lean towards generating hyper-real and dramatic images. "
            "Natural causes the model to produce more natural, less hyper-real looking images. "
            "This param is only supported for `dall-e-3`.",
        )

        @validator("n")
        def images_requested_matches_model(cls, v, values, **kwargs):
            """Validate n matches model."""
            if model := values.get("model"):
                if model == ModelEnum.DALLE3.value:
                    if v > 1:
                        raise ValueError(
                            "dall-e-3 only supports a single image generation at a time."
                        )
            return v

        @validator("size")
        def size_matches_model(cls, v, values, **kwargs):
            """Validate size matches model."""
            if model := values.get("model"):
                if model == ModelEnum.DALLE3.value:
                    if v not in [
                        ImageSizeEnum.large.value,
                        ImageSizeEnum.large_landscape.value,
                        ImageSizeEnum.large_portrait.value,
                    ]:
                        raise ValueError(f"dall-e-3 does not support size: {v}.")
                if model == ModelEnum.DALLE2.value:
                    if v not in [
                        ImageSizeEnum.large.value,
                        ImageSizeEnum.medium.value,
                        ImageSizeEnum.small.value,
                    ]:
                        raise ValueError(f"dall-e-2 does not support size: {v}.")
            return v

        @validator("quality")
        def valid_quality(cls, v, **kwargs):
            """Validate quality value."""
            if v not in QualityEnum.list():
                raise ValueError(f"Quality must be one of: {QualityEnum.list()}")
            return v

        @validator("style")
        def valid_style(cls, v, **kwargs):
            """Validate style value."""
            if v not in StyleEnum.list():
                raise ValueError(f"Style must be one of: {StyleEnum.list()}")
            return v

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
    ) -> List[str]:
        """Generate image(s) with the options provided. Returns URLs of generated images."""
        logging.debug(f"Making OpenAI dall-e call on behalf of user with id: {user}")

        def _generate_with_retry(image_prompt: str, api_inputs: Dict) -> Any:
            client = OpenAI(
                # default is 2
                max_retries=0,
                api_key=self.config.openai_api_key,
            )
            logging.warning("dall-e inputs", extra={"inputs": api_inputs})
            return client.with_options(max_retries=self.config.max_retries).images.generate(
                prompt=image_prompt, **api_inputs
            )

        # use runtime overrides
        try:
            inputs = self._inputs_from_config_and_runtime_params(options)
        except ValidationError as ve:
            raise SteamshipError(f"Invalid runtime parameterization of plugin: {ve}")

        openai_result = _generate_with_retry(image_prompt=prompt, api_inputs=inputs)
        return [obj.url for obj in openai_result.data]

    def run(
        self, request: PluginRequest[RawBlockAndTagPluginInputWithPreallocatedBlocks]
    ) -> InvocableResponse[RawBlockAndTagPluginOutput]:
        """Run the image generator against all the text, combined."""
        prompt = " ".join([block.text for block in request.data.blocks if block.text is not None])
        user_id = self.context.user_id if self.context is not None else "testing"
        urls = self.generate_with_retry(prompt=prompt, user=user_id, options=request.data.options)

        usage = []
        output_blocks = request.data.output_blocks
        for idx, url in enumerate(urls):
            resp = requests.get(url)
            if resp.ok:
                logging.debug(f"[{idx}] appending to block: {output_blocks[idx].id}")
                output_blocks[idx].append_stream(resp.content)
                output_blocks[idx].finish_stream()
                usage.append(
                    UsageReport(
                        operation_type=OperationType.RUN,
                        operation_unit=OperationUnit.UNITS,
                        operation_amount=1,  # single image generated
                        audit_id=f"{user_id}-{hash(prompt)}",  # do we want a better audit id ?
                    )
                )
            else:
                logging.error(
                    f"generation failed for block during content retrieval: ({resp.status_code}, {resp.content})"
                )
                output_blocks[idx].abort_stream()

        logging.debug("returning finished state for stream.")
        return InvocableResponse(data=StreamCompletePluginOutput(usage=usage))

    def determine_output_block_types(
        self, request: PluginRequest[RawBlockAndTagPluginInput]
    ) -> InvocableResponse[BlockTypePluginOutput]:
        """Provide hints to engine on number of blocks and their mime-types."""
        inputs = self._inputs_from_config_and_runtime_params(request.data.options)
        num_images = inputs.get("n", 1)
        block_types_to_create = [MimeTypes.PNG] * num_images
        return InvocableResponse(
            data=BlockTypePluginOutput(block_types_to_create=block_types_to_create)
        )

    def _inputs_from_config_and_runtime_params(self, options: Optional[dict]) -> dict:
        if options is not None and "model" in options:
            raise SteamshipError(
                "Model may not be overridden in runtime options. "
                "Please configure 'model' when creating a plugin instance."
            )

        if options is not None and "size" in options:
            raise SteamshipError(
                "Size may not be overridden in runtime options. "
                "Please configure 'size' when creating a plugin instance."
            )

        if options is not None and "quality" in options:
            raise SteamshipError(
                "Quality may not be overridden in runtime options. "
                "Please configure 'quality' when creating a plugin instance."
            )

        temp_config = DallEPlugin.DallEPluginConfig(**self.config.dict())
        temp_config.extend_with_dict(options, overwrite=True)
        validated_config = DallEPlugin.DallEPluginConfig(**temp_config.dict())
        return validated_config.dict(exclude={"openai_api_key", "request_timeout", "max_retries"})
