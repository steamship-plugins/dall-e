import pytest
from pydantic import ValidationError
from steamship import Block, TaskState
from steamship.plugin.inputs.raw_block_and_tag_plugin_input import RawBlockAndTagPluginInput
from steamship.plugin.request import PluginRequest

from src.api import DallEPlugin


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
        DallEPlugin(config={"n": 123})

    with pytest.raises(ValidationError):
        DallEPlugin(config={"size": "blahadlsadf"})

    with pytest.raises(ValueError) as e:
        dalle = DallEPlugin()
        dalle.generate_with_retry(user="foo", prompt="bar", options={"n": 12})

    assert "received: 12" in str(e)

    with pytest.raises(ValueError) as e:
        dalle = DallEPlugin()
        dalle.generate_with_retry(user="foo", prompt="bar", options={"size": "fadsfadsf"})

    assert "fadsfadsf" in str(e)
