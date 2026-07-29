"""Stub tokenizer/config mimicking the transformers API for local smoke tests."""

import torch


class _Encoding:
    pass


class StubProcessor:
    def __init__(self, **kwargs):
        self.pad_token = None
        self.eos_token = "</s>"
        self.bos_token = "<s>"
        self.padding_side = "right"
        self.model_max_length = 10**9

    def apply_chat_template(
        self, messages, tokenize=False, bos_token=None, add_generation_prompt=True
    ):
        return " ".join(m["content"] for m in messages)

    def __call__(
        self,
        text,
        truncation=True,
        padding=True,
        return_tensors="pt",
        padding_side="left",
    ):
        batch = [t.split() for t in text]
        max_len = max(len(t) for t in batch)
        input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
        mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, toks in enumerate(batch):
            n = len(toks)
            input_ids[i, max_len - n :] = torch.arange(1, n + 1)
            mask[i, max_len - n :] = 1
        enc = _Encoding()
        enc.input_ids = input_ids
        enc.attention_mask = mask
        return enc


class StubConfig:
    max_position_embeddings = 512


class StubPretrainedModel:
    """Stands in for AutoModel.from_pretrained, keeping the kwargs it was given."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class StubEmbedModel:
    """Stands in for the language model itself: two hidden layers of width 4.

    Enough for `etp embed` to run end to end on CPU, with no checkpoint to read
    and no network.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, input_ids, attention_mask, output_hidden_states):
        batch, length = input_ids.shape

        class Output:
            hidden_states = (
                torch.ones(batch, length, 4),
                torch.ones(batch, length, 4) * 2,
            )

        return Output()
