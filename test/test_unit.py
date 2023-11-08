import pytest
from pydantic import ValidationError
from steamship import Block, SteamshipError, TaskState
from steamship.plugin.inputs.raw_block_and_tag_plugin_input import RawBlockAndTagPluginInput
from steamship.plugin.request import PluginRequest

from api import DallEPlugin, ImageSizeEnum


def test_generator():
    dalle = DallEPlugin()
    request = PluginRequest(
        data=RawBlockAndTagPluginInput(blocks=[Block(text="a cat riding a bicycle")])
    )
    response = dalle.run(request)

    assert response.status.state is TaskState.succeeded
    assert response.data is not None

    assert len(response.data.blocks) == 1
    assert response.data.blocks[0].url is not None


def test_generator_overrides():
    dalle = DallEPlugin()
    request = PluginRequest(
        data=RawBlockAndTagPluginInput(
            blocks=[Block(text="a cat riding a bicycle")], options={"n": 2, "size": "512x512"}
        )
    )
    response = dalle.run(request)

    assert response.status.state is TaskState.succeeded
    assert response.data is not None

    assert len(response.data.blocks) == 2
    for block in response.data.blocks:
        assert block.url is not None


def test_generator_validation():
    with pytest.raises(ValidationError):
        DallEPlugin(config={"n": 133})

    with pytest.raises(ValidationError):
        DallEPlugin(config={"size": "blahadlsadf"})

    with pytest.raises(SteamshipError) as e:
        dalle = DallEPlugin()
        dalle.generate_with_retry(user="foo", prompt="bar", options={"n": 12})

    assert "Invalid runtime parameterization of plugin" in str(e)

    with pytest.raises(SteamshipError) as e:
        dalle = DallEPlugin()
        dalle.generate_with_retry(user="foo", prompt="bar", options={"size": "fadsfadsf"})

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
