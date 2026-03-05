"""
modulus.tests.test_presets

Tests for model-specific optimizer presets.
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp

from modulus.optim.presets import (
    build_llama_grouped_hyperball_tx,
    make_llama_like_labels,
    make_llama_like_mask,
)


def _toy_llama_like_params(width: int = 16, vocab: int = 64, rank: int = 4):
    key = jax.random.PRNGKey(0)
    k = jax.random.split(key, 8)
    s = 1.0 / jnp.sqrt(float(width))
    return {
        "model": {
            "embed_tokens": {"embedding": jax.random.normal(k[0], (vocab, width)) * 0.01},
            "layers": {
                "0": {
                    "self_attn": {
                        "q_proj": {"kernel": jax.random.normal(k[1], (width, width)) * s},
                        "o_proj": {"kernel": jax.random.normal(k[2], (width, width)) * s},
                    },
                    "mlp": {
                        "gate_proj": {"kernel": jax.random.normal(k[3], (width, width * 4)) * s},
                        "down_proj": {"kernel": jax.random.normal(k[4], (width * 4, width)) * s},
                    },
                    "input_layernorm": {"scale": jnp.ones((width,))},
                    "post_attention_layernorm": {"scale": jnp.ones((width,))},
                    "adapter": {
                        "lora_A": jax.random.normal(k[5], (width, rank)) * 0.01,
                        "lora_B": jnp.zeros((rank, width)),
                    },
                }
            },
        },
        "lm_head": {"kernel": jax.random.normal(k[6], (width, vocab)) * s},
        "misc": {"proj": jax.random.normal(k[7], (width, width)) * s},
    }


class PresetTests(unittest.TestCase):
    def test_llama_like_labels(self):
        params = _toy_llama_like_params()
        labels = make_llama_like_labels()(params)

        self.assertEqual(labels["model"]["layers"]["0"]["self_attn"]["q_proj"]["kernel"], "attn")
        self.assertEqual(labels["model"]["layers"]["0"]["mlp"]["gate_proj"]["kernel"], "mlp")
        self.assertEqual(labels["model"]["embed_tokens"]["embedding"], "embed")
        self.assertEqual(labels["model"]["layers"]["0"]["input_layernorm"]["scale"], "norm")
        self.assertEqual(labels["lm_head"]["kernel"], "other")

    def test_llama_like_mask_defaults(self):
        params = _toy_llama_like_params()
        mask = make_llama_like_mask()(params)

        self.assertTrue(mask["model"]["layers"]["0"]["self_attn"]["q_proj"]["kernel"])
        self.assertTrue(mask["model"]["layers"]["0"]["mlp"]["gate_proj"]["kernel"])
        self.assertFalse(mask["model"]["embed_tokens"]["embedding"])
        self.assertFalse(mask["model"]["layers"]["0"]["adapter"]["lora_A"])
        self.assertFalse(mask["lm_head"]["kernel"])
        self.assertFalse(mask["model"]["layers"]["0"]["input_layernorm"]["scale"])

    def test_llama_grouped_tx_build_and_update(self):
        params = _toy_llama_like_params()
        tx = build_llama_grouped_hyperball_tx()(params)
        st = tx.init(params)
        grads = jax.tree.map(jnp.ones_like, params)
        updates, _ = tx.update(grads, st, params)
        self.assertIn("model", updates)
        self.assertIn("lm_head", updates)


if __name__ == "__main__":
    unittest.main()
