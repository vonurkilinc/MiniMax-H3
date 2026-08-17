import unittest

from src.generate import lora_target_modules


class LoraTargetMappingTests(unittest.TestCase):
    def test_main_block_fused_qkv_expands_in_qkv_order(self):
        self.assertEqual(
            lora_target_modules("diffusion_model.blocks.7.attn.qkv_proj"),
            (
                "transformer_blocks.7.attn.to_q",
                "transformer_blocks.7.attn.to_k",
                "transformer_blocks.7.attn.to_v",
            ),
        )

    def test_main_block_output_and_mlp_names(self):
        self.assertEqual(
            lora_target_modules("diffusion_model.blocks.3.attn.out_proj"),
            ("transformer_blocks.3.attn.to_out.0",),
        )
        self.assertEqual(
            lora_target_modules("diffusion_model.blocks.3.mlp.fc1"),
            ("transformer_blocks.3.ff.net.0.proj",),
        )
        self.assertEqual(
            lora_target_modules("diffusion_model.blocks.3.mlp.fc2"),
            ("transformer_blocks.3.ff.net.2",),
        )

    def test_token_refiner_names(self):
        self.assertEqual(
            lora_target_modules("diffusion_model.token_refiner.blocks.1.attn.out_proj"),
            ("token_refiner.refiner_blocks.1.attn.to_out.0",),
        )

    def test_existing_diffusers_name_passes_through(self):
        name = "transformer_blocks.0.attn.to_q"
        self.assertEqual(lora_target_modules(name), (name,))


if __name__ == "__main__":
    unittest.main()
