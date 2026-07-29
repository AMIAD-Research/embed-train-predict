import hydra
from omegaconf import OmegaConf

_instantiate_cache = {}


def _cached_instantiate(cfg):
    """Instantiate a Hydra config once per process."""
    if not OmegaConf.is_config(cfg) and not isinstance(cfg, dict):
        return cfg
    key = OmegaConf.to_yaml(cfg) if OmegaConf.is_config(cfg) else repr(cfg)
    if key not in _instantiate_cache:
        _instantiate_cache[key] = hydra.utils.instantiate(cfg)
    return _instantiate_cache[key]


def text_collate(data, processor, config, prompt=None, text_key="text"):
    batch = {key: [item[key] for item in data] for key in data[0]}

    output = {key: values for key, values in batch.items() if key != text_key}
    output["source_text"] = batch[text_key]

    processor = _cached_instantiate(processor)
    config = _cached_instantiate(config)

    # Cap the truncation length with max_position_embeddings
    limit = getattr(config, "max_position_embeddings", None)
    declared = getattr(processor, "model_max_length", None)
    if limit is not None and (declared is None or declared > limit):
        processor.model_max_length = limit

    processor.padding_side = "left"
    if processor.pad_token is None:
        processor.pad_token = processor.eos_token

    texts = batch[text_key]

    if prompt is not None:
        texts = [
            processor.apply_chat_template(
                [
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": f'"{text}" {prompt["user"]}'},
                ],
                tokenize=False,
                bos_token=processor.bos_token,
                add_generation_prompt=True,
            )
            for text in texts
        ]
        output["prompt_text"] = texts

    encoded = processor(
        text=texts,
        truncation=True,
        padding=True,
        return_tensors="pt",
        padding_side="left",
    )

    output["ids"] = encoded.input_ids
    output["mask"] = encoded.attention_mask

    return output
