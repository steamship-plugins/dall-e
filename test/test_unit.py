import pytest
from pydantic import ValidationError
from steamship import Block, File, MimeTypes, Steamship, SteamshipError
from steamship.plugin.inputs.raw_block_and_tag_plugin_input import RawBlockAndTagPluginInput
from steamship.plugin.inputs.raw_block_and_tag_plugin_input_with_preallocated_blocks import (
    RawBlockAndTagPluginInputWithPreallocatedBlocks,
)
from steamship.plugin.outputs.plugin_output import UsageReport
from steamship.plugin.request import PluginRequest

from api import DallEPlugin, ImageSizeEnum


def run_test_streaming(
    client: Steamship, plugin: DallEPlugin, blocks: [Block], options: dict
) -> ([UsageReport], [Block]):
    blocks_to_allocate = plugin.determine_output_block_types(
        PluginRequest(data=RawBlockAndTagPluginInput(blocks=blocks, options=options))
    )
    file = File.create(client, blocks=[])
    output_blocks = []
    for block_type_to_allocate in blocks_to_allocate.data.block_types_to_create:
        assert block_type_to_allocate == MimeTypes.PNG.value
        output_blocks.append(
            Block.create(
                client,
                file_id=file.id,
                mime_type=MimeTypes.PNG.value,
                streaming=True,
            )
        )

    response = plugin.run(
        PluginRequest(
            data=RawBlockAndTagPluginInputWithPreallocatedBlocks(
                blocks=blocks, options=options, output_blocks=output_blocks
            )
        )
    )
    result_blocks = [Block.get(client, _id=block.id) for block in output_blocks]
    return response.data.usage, result_blocks


def test_generator():
    with Steamship.temporary_workspace() as client:
        dalle = DallEPlugin()
        usage, new_blocks = run_test_streaming(
            client, dalle, [Block(text="a cat riding a bicycle")], options={}
        )

        assert new_blocks is not None
        assert len(new_blocks) == 1

        assert usage is not None
        assert len(usage) == 1


def test_generator_overrides():
    with Steamship.temporary_workspace() as client:
        dalle = DallEPlugin(config={"size": "512x512"})
        usage, new_blocks = run_test_streaming(
            client, dalle, [Block(text="a cat riding a bicycle")], options={"n": 2}
        )

        assert new_blocks is not None
        assert len(new_blocks) == 2

        assert usage is not None
        assert len(usage) == 2


def test_generator_validation():
    with pytest.raises(ValidationError):
        DallEPlugin(config={"n": 133})

    with pytest.raises(ValidationError):
        DallEPlugin(config={"size": "blahadlsadf"})

    with pytest.raises(SteamshipError) as e:
        dalle = DallEPlugin()
        dalle.generate_with_retry(user="foo", prompt="bar", options={"n": 12})

    assert "Invalid runtime parameterization of plugin" in str(e)

    with pytest.raises(ValidationError):
        DallEPlugin(config={"n": 4, "model": "dall-e-3"})

    with pytest.raises(ValidationError):
        DallEPlugin(config={"model": "dall-e-3", "style": "crazytown"})

    with pytest.raises(ValidationError):
        DallEPlugin(config={"model": "dall-e-3", "quality": "just ok"})

    with pytest.raises(ValidationError):
        DallEPlugin(config={"model": "dall-e-2", "size": ImageSizeEnum.large_portrait})

    with pytest.raises(ValidationError):
        DallEPlugin(config={"model": "dall-e-3", "size": ImageSizeEnum.medium})


def test_runtime_config_validation_returns_values():
    plugin = DallEPlugin(config={"model": "dall-e-3"})
    config_dict = plugin._inputs_from_config_and_runtime_params(options={"n": 1})
    assert config_dict.get("quality")
    assert config_dict.get("size") == ImageSizeEnum.large
    assert config_dict.get("n") == 1
    assert config_dict.get("style")


def test_no_override_of_model_at_runtime():
    plugin = DallEPlugin(config={"model": "dall-e-2"})

    with pytest.raises(SteamshipError):
        plugin._inputs_from_config_and_runtime_params(options={"model": "dall-e-3"})

    with pytest.raises(SteamshipError):
        plugin._inputs_from_config_and_runtime_params(options={"size": "should fail"})

    with pytest.raises(SteamshipError):
        plugin._inputs_from_config_and_runtime_params(options={"quality": "should fail"})
