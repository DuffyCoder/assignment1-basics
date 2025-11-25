from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .softmax import Softmax
from .tokenizer import Tokenizer
from .transformer_lm import TransformerLM


class Decoding(nn.Module):
    """
    Utility for autoregressively decoding from ``TransformerLM``.

    The class keeps track of the model hyper-parameters and exposes a ``generate``
    method that:
      * tokenizes an input prompt (or accepts token ids directly),
      * repeatedly runs the language model to obtain the next-token distribution,
      * samples according to the configured temperature / top-p settings,
      * stops when ``<|endoftext|>`` is produced or when the user-provided maximum
        number of new tokens has been generated.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        max_length: int,
        special_tokens: list[str] | None = None,
        top_p: float = 0.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.max_length = max_length
        self.special_tokens = special_tokens or []
        self.temperature = temperature
        self.top_p = top_p
        self._sampler = Softmax(temperature=temperature, top_p=top_p)
        self._stop_token = "<|endoftext|>" if "<|endoftext|>" in self.special_tokens else None

    def _prepare_prompt(
        self,
        prompt: str | Sequence[int],
        tokenizer: Tokenizer,
    ) -> list[int]:
        if isinstance(prompt, str):
            prompt_tokens = tokenizer.encode(prompt)
        else:
            prompt_tokens = list(prompt)

        if not prompt_tokens:
            raise ValueError("Prompt must contain at least one token.")
        return prompt_tokens

    def _maybe_get_stop_token_id(self, tokenizer: Tokenizer) -> int | None:
        if self._stop_token is None:
            return None
        encoded = tokenizer.encode(self._stop_token)
        return encoded[0] if encoded else None

    @torch.no_grad()
    def generate(
        self,
        prompt: str | Sequence[int],
        tokenizer: Tokenizer,
        weights: dict[str, Tensor],
        max_new_tokens: int | None = None,
        device: torch.device | str | None = None,
        return_tokens: bool = False,
    ):
        """
        Decode from the language model using autoregressive sampling.

        Args:
            prompt: Either a string or a list of token ids to condition on.
            tokenizer: Tokenizer capable of converting between text and ids.
            weights: State dict for ``TransformerLM`` (same format as training).
            max_new_tokens: Optional cap on the number of new tokens to generate.
                If omitted, ``self.max_length`` is used.
            device: Optional torch device. If ``None`` we infer it from the weights.
            return_tokens: Whether to return token ids alongside the decoded text.

        Returns:
            str | dict: By default returns the decoded completion text. If
            ``return_tokens`` is ``True`` a dictionary containing the completion text
            and the generated token ids is returned.
        """

        if max_new_tokens is None:
            max_new_tokens = self.max_length
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer.")

        prompt_tokens = self._prepare_prompt(prompt, tokenizer)
        generated_tokens = list(prompt_tokens)
        stop_token_id = self._maybe_get_stop_token_id(tokenizer)

        # Ensure the LM inputs and weights live on the same device.
        weight_device = next(iter(weights.values())).device
        target_device = torch.device(device) if device is not None else weight_device
        if weight_device != target_device:
            weights = {k: v.to(target_device) for k, v in weights.items()}

        # Instantiate a TransformerLM once and reuse it by updating the indices.
        dummy_in_indices = torch.tensor(
            [generated_tokens[-self.context_length:]],
            dtype=torch.long,
            device=target_device,
        )
        lm = TransformerLM(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            rope_theta=self.rope_theta,
            weights=weights,
            in_indices=dummy_in_indices,
        )

        new_tokens = 0
        while new_tokens < max_new_tokens and len(generated_tokens) < self.max_length:
            context = generated_tokens[-self.context_length :]
            input_ids = torch.tensor([context], dtype=torch.long, device=target_device)
            lm.in_indices = input_ids
            logits = lm()[0, -1, :]
            probs = self._sampler(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            generated_tokens.append(next_token)
            new_tokens += 1
            if stop_token_id is not None and next_token == stop_token_id:
                break

        completion_ids = generated_tokens[len(prompt_tokens) :]
        completion_text = tokenizer.decode(completion_ids)

        if return_tokens:
            return {
                "completion": completion_text,
                "new_token_ids": completion_ids,
                "all_token_ids": generated_tokens,
            }
        return completion_text

    forward = generate
