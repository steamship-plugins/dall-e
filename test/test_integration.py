"""Test dall-e generator plugin via integration tests."""

from filetype import filetype
from steamship import MimeTypes, Steamship

GENERATOR_HANDLE = "dall-e"


def test_generator():
    with Steamship.temporary_workspace() as steamship:
        dalle = steamship.use_plugin(GENERATOR_HANDLE)

        task = dalle.generate(
            text="A cat on a bicycle",
            append_output_to_file=True,
            options={"n": 2, "size": "256x256"},
        )
        task.wait()
        blocks = task.output.blocks

        assert blocks is not None
        assert len(blocks) == 2
        for block in blocks:
            # check that Steamship thinks it is a PNG and that the bytes seem like a PNG
            assert block.mime_type == MimeTypes.PNG
            assert filetype.guess_mime(block.raw()) == MimeTypes.PNG


def test_generator_streaming():
    with Steamship.temporary_workspace() as steamship:
        dalle = steamship.use_plugin(GENERATOR_HANDLE)

        task = dalle.generate(
            text="A cat on a bicycle",
            append_output_to_file=True,
            options={"n": 2, "size": "256x256"},
            streaming=True,
        )
        task.wait()

        blocks = task.output.blocks
        assert blocks is not None
        assert len(blocks) == 2
        for block in blocks:
            # check that Steamship thinks it is a PNG and that the bytes seem like a PNG
            assert block.mime_type == MimeTypes.PNG
            assert filetype.guess_mime(block.raw()) == MimeTypes.PNG
