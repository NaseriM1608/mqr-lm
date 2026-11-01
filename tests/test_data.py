import hashlib

import numpy as np
import pytest
import torch

from mqr.data import ByteTokenizer, TokenStream, load_tokenizer


def test_byte_tokenizer_round_trip_multilingual_and_special_ids(tmp_path):
    tokenizer = ByteTokenizer()
    text = "English. Grüße aus Köln. العربية جميلة. فارسی را می‌خوانیم. 🌍"
    ids = tokenizer.encode(text)
    assert ids[-1] == 257
    assert ids[:-1] == list(text.encode("utf-8"))
    assert tokenizer.decode([258, *ids, 256]) == text
    assert tokenizer.decode([0xF0]) == "\ufffd"
    with pytest.raises(ValueError, match="Invalid byte token"):
        tokenizer.decode([259])
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    assert load_tokenizer(path).encode(text) == ids


def test_next_token_shifts_and_generator_state_reproducibility():
    stream = TokenStream(torch.arange(40), context_length=7)
    first_generator = torch.Generator().manual_seed(54)
    second_generator = torch.Generator().manual_seed(54)
    first = stream.sample(5, first_generator)
    repeated = stream.sample(5, second_generator)
    assert all(torch.equal(left, right) for left, right in zip(first, repeated))
    x, targets = first
    assert x.shape == targets.shape == (5, 7)
    assert x.dtype == targets.dtype == torch.long
    assert torch.equal(x + 1, targets)
    state = first_generator.get_state()
    expected = stream.sample(4, first_generator)
    second_generator.set_state(state)
    actual = stream.sample(4, second_generator)
    assert all(torch.equal(left, right) for left, right in zip(expected, actual))


def test_shortest_valid_stream_includes_final_target():
    stream = TokenStream(torch.tensor([3, 8, 2, 7]), 3)
    x, y = stream.sample(2, torch.Generator().manual_seed(9))
    assert x.tolist() == [[3, 8, 2], [3, 8, 2]]
    assert y.tolist() == [[8, 2, 7], [8, 2, 7]]


def test_memmap_stays_mapped_and_fingerprint_is_dtype_independent(tmp_path):
    tokens = np.arange(100, dtype="<u4")
    path = tmp_path / "tokens.npy"
    np.save(path, tokens)
    stream = TokenStream.from_file(path, context_length=10, vocab_size=100)
    assert isinstance(stream.tokens, np.memmap)
    assert stream.tokens.mode == "r"
    assert stream.max_token_id == 99
    assert stream.fingerprint == hashlib.sha256(tokens.tobytes()).hexdigest()
    assert TokenStream(torch.arange(100), 10).fingerprint == stream.fingerprint
    x, y = stream.sample(3, torch.Generator().manual_seed(11))
    assert torch.equal(x + 1, y)
    # A caller may mutate its batch without writing through to the corpus file.
    x.fill_(-1)
    assert stream.tokens.min() == 0


@pytest.mark.parametrize(
    ("tokens", "context"),
    [
        (torch.arange(4), 4),
        (torch.arange(4), 0),
        (torch.zeros(3, 4, dtype=torch.long), 2),
        (torch.tensor([1.0, 2.0, 3.0]), 2),
        (torch.tensor([-1, 2, 3]), 2),
    ],
)
def test_reject_invalid_streams(tokens, context):
    with pytest.raises(ValueError):
        TokenStream(tokens, context)


def test_vocabulary_bounds_checked_before_training(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.array([1, 4, 9], dtype=np.uint32))
    with pytest.raises(ValueError, match="outside"):
        TokenStream.from_file(path, context_length=2, vocab_size=9)


def test_text_reading_preserves_exact_utf8_bytes(tmp_path):
    path = tmp_path / "text.txt"
    text = "Hello\r\nGrüße\r\nسلام\r\n"
    path.write_bytes(text.encode("utf-8"))
    stream = TokenStream.from_text(path, context_length=2)
    assert stream.tokens.tolist() == ByteTokenizer().encode(text)
