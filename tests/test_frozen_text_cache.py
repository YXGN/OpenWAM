"""Text-cache contract: identical values, no gradients/RNG changes, no aliasing."""

import pytest
import torch

from openwam.model.video_backbone.wan.encode import FrozenTextCache, encode_text


class Tokenizer:
    def __call__(self, prompts, **kwargs):
        ids = torch.zeros(len(prompts), 512, dtype=torch.long)
        for i, prompt in enumerate(prompts):
            ids[i, :len(prompt)] = torch.tensor([ord(c) % 32 + 1 for c in prompt])
        return ids, ids.ne(0).long()


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(34, 8)
        self.dropout = torch.nn.Dropout(0.1)
        self.calls = 0

    def forward(self, ids, mask):
        self.calls += 1
        return self.dropout(self.embed(ids))


def test_cache_preserves_order_padding_rng_and_does_not_alias():
    encoder = Encoder().eval().requires_grad_(False)
    tokenizer = Tokenizer()
    cache = FrozenTextCache(tokenizer=tokenizer, text_encoder=encoder)
    prompts = ["hammer", "charger", "hammer", "cake"]
    expected, lengths = encode_text(prompts, tokenizer=tokenizer, text_encoder=encoder, device="cpu")
    rng = torch.get_rng_state().clone()
    cold, cold_lengths = cache.encode(prompts, device="cpu")
    assert torch.equal(cold, expected) and torch.equal(cold_lengths, lengths)
    cold.zero_()  # In-place downstream work must not corrupt the cache.
    calls = encoder.calls
    actual, actual_lengths = cache.encode(prompts[::-1], device="cpu")
    assert encoder.calls == calls
    assert torch.equal(actual, expected.flip(0))
    assert torch.equal(actual_lengths, lengths.flip(0))
    assert torch.equal(rng, torch.get_rng_state())
    assert not actual.requires_grad
    actual.zero_()
    assert torch.equal(cache.encode(prompts, device="cpu")[0], expected)
    cache.encode(["new", "hammer"], device="cpu")
    assert encoder.calls == calls + 1


@pytest.mark.parametrize("bad_state", ["training", "trainable", "child_training"])
def test_cache_rejects_stochastic_or_trainable_encoder(bad_state):
    encoder = Encoder().eval().requires_grad_(False)
    cache = FrozenTextCache(tokenizer=Tokenizer(), text_encoder=encoder)
    cache.encode(["hammer"], device="cpu")
    if bad_state == "training":
        encoder.train()
    elif bad_state == "trainable":
        encoder.embed.weight.requires_grad_(True)
    else:
        encoder.dropout.train()
    with pytest.raises(ValueError, match="frozen encoder"):
        cache.encode(["hammer"], device="cpu")


def test_cache_refreshes_after_dtype_change():
    encoder = Encoder().eval().requires_grad_(False)
    cache = FrozenTextCache(tokenizer=Tokenizer(), text_encoder=encoder)
    cache.encode(["cake"], device="cpu")
    encoder.double()
    actual, _ = cache.encode(["cake"], device="cpu")
    assert actual.dtype == torch.float64
    assert encoder.calls == 2
