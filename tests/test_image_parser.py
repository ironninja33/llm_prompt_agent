"""Unit tests for the image metadata parser using real example data.

Training directories: only .txt files are read (images are ignored).
Output directories: .png and .jpg files are parsed for embedded prompts.
"""

import json
import os
import unittest

from src.services.image_parser import (
    parse_file,
    ParsedImageData,
    _extract_from_comfyui_workflow,
    _extract_from_filename,
    _split_positive_negative,
    _parse_a1111_metadata,
    _decode_user_comment,
)

# Paths to example data
EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "examples"
)
TRAIN_DIR = os.path.join(EXAMPLES_DIR, "train_lora", "salma")
OUTPUT_DIR = os.path.join(EXAMPLES_DIR, "output")
ASA_DIR = os.path.join(OUTPUT_DIR, "asa_akira")
JJ_DIR = os.path.join(OUTPUT_DIR, "jeneveve_jolie")


# ── Training text files ─────────────────────────────────────────────────

class TestTrainingTextFiles(unittest.TestCase):
    """Test parsing of real training data .txt files."""

    def test_dusk_till_dawn_caption(self):
        path = os.path.join(TRAIN_DIR,
            "-From-Dusk-Till-Dawn-Screencaps-salma-hayek-23080164-1280-800.txt")
        result = parse_file(path)
        self.assertIsNotNone(result)
        self.assertIn("salma hayek", result.prompt.lower())
        self.assertIn("dusk till dawn", result.prompt.lower())
        self.assertIsNone(result.base_model)
        self.assertEqual(result.loras, [])

    def test_bikini_caption(self):
        result = parse_file(os.path.join(TRAIN_DIR, "1__1_.txt"))
        self.assertIsNotNone(result)
        self.assertIn("salma hayek", result.prompt.lower())
        self.assertIn("bikini", result.prompt.lower())

    def test_all_training_txt_files_parseable(self):
        """Every .txt file in the training directory should parse."""
        txt_files = [
            os.path.join(TRAIN_DIR, f)
            for f in os.listdir(TRAIN_DIR) if f.endswith(".txt")
        ]
        self.assertGreater(len(txt_files), 0)
        for path in txt_files:
            result = parse_file(path)
            self.assertIsNotNone(result, f"Failed: {os.path.basename(path)}")
            self.assertGreater(len(result.prompt), 10, f"Too short: {os.path.basename(path)}")


# ── Output PNG files (ComfyUI workflow) ──────────────────────────────────

class TestOutputPNGFiles(unittest.TestCase):
    """Test parsing of real output PNG files containing ComfyUI workflows."""

    def _png_files(self):
        if not os.path.isdir(ASA_DIR):
            self.skipTest("asa_akira output directory not found")
        return [
            os.path.join(ASA_DIR, f) for f in os.listdir(ASA_DIR)
            if f.lower().endswith(".png")
        ]

    def test_png_extracts_prompt_from_workflow(self):
        for path in self._png_files():
            result = parse_file(path)
            self.assertIsNotNone(result)
            self.assertIn("photorealistic", result.prompt.lower())

    def test_png_extracts_base_model(self):
        for path in self._png_files():
            result = parse_file(path)
            self.assertIsNotNone(result.base_model)
            self.assertIn("chroma", result.base_model.lower())

    def test_png_extracts_loras(self):
        for path in self._png_files():
            result = parse_file(path)
            self.assertGreater(len(result.loras), 0)
            self.assertTrue(any("asa" in l.lower() for l in result.loras))

    def test_png_extracts_negative_prompt(self):
        for path in self._png_files():
            result = parse_file(path)
            self.assertIsNotNone(result.negative_prompt)
            self.assertIn("ugly", result.negative_prompt.lower())


# ── Output JPG files (EXIF UserComment, A1111 format) ────────────────────

class TestOutputJPGFiles(unittest.TestCase):
    """Test parsing of real output JPG files with EXIF-embedded prompts."""

    def _jpg_files(self):
        if not os.path.isdir(JJ_DIR):
            self.skipTest("jeneveve_jolie output directory not found")
        return [
            os.path.join(JJ_DIR, f) for f in os.listdir(JJ_DIR)
            if f.lower().endswith((".jpg", ".jpeg"))
        ]

    def test_jpg_extracts_prompt_from_exif(self):
        for path in self._jpg_files():
            result = parse_file(path)
            self.assertIsNotNone(result)
            self.assertIn("jolie", result.prompt.lower())

    def test_jpg_extracts_negative_prompt(self):
        for path in self._jpg_files():
            result = parse_file(path)
            self.assertIsNotNone(result.negative_prompt)
            self.assertIn("ugly", result.negative_prompt.lower())

    def test_jpg_extracts_model(self):
        for path in self._jpg_files():
            result = parse_file(path)
            self.assertIsNotNone(result.base_model)

    def test_jpg_extracts_loras(self):
        for path in self._jpg_files():
            result = parse_file(path)
            self.assertGreater(len(result.loras), 0)
            self.assertTrue(any("jenaveve" in l.lower() for l in result.loras))


# ── ComfyUI workflow parsing ─────────────────────────────────────────────

class TestComfyUIWorkflowParsing(unittest.TestCase):

    def test_primitive_string_positive_node(self):
        workflow = {
            "2": {
                "class_type": "PrimitiveStringMultiline",
                "inputs": {"value": "photorealistic, woman in a garden"},
                "_meta": {"title": "String (Positive)"}
            },
            "3": {
                "class_type": "PrimitiveStringMultiline",
                "inputs": {"value": "ugly, bad quality"},
                "_meta": {"title": "String (Negative)"}
            },
            "4": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": "chroma-v45.safetensors"},
            },
            "20": {
                "class_type": "Lora Loader Stack (rgthree)",
                "inputs": {
                    "lora_01": "concept_a.safetensors",
                    "strength_01": 1.0,
                    "lora_02": "detail_v2.safetensors",
                    "strength_02": 0.5,
                    "lora_03": "None",
                    "strength_03": 1.0,
                },
            },
        }

        result = _extract_from_comfyui_workflow(workflow, "/test/image.png")
        self.assertEqual(result.prompt, "photorealistic, woman in a garden")
        self.assertEqual(result.negative_prompt, "ugly, bad quality")
        self.assertEqual(result.base_model, "chroma-v45.safetensors")
        self.assertIn("concept_a.safetensors", result.loras)
        self.assertIn("detail_v2.safetensors", result.loras)
        self.assertNotIn("None", result.loras)
        self.assertEqual(len(result.loras), 2)

    def test_clip_text_encode_positive(self):
        workflow = {
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "a beautiful landscape, oil painting"},
                "_meta": {"title": "CLIP Text Encode (Positive)"}
            },
        }
        result = _extract_from_comfyui_workflow(workflow, "/test.png")
        self.assertIn("landscape", result.prompt)


# ── A1111 metadata parsing ───────────────────────────────────────────────

class TestA1111MetadataParsing(unittest.TestCase):

    def test_full_a1111_format(self):
        text = (
            "jenaveve jolie, a photograph of a woman\n"
            "Negative prompt: ugly, blurry\n"
            "Steps: 20, Sampler: euler, Model: SUPIR-v0F, "
            "Extra info: my_lora.safetensors, None, None"
        )
        pos, neg, model, loras = _parse_a1111_metadata(text)
        self.assertEqual(pos, "jenaveve jolie, a photograph of a woman")
        self.assertEqual(neg, "ugly, blurry")
        self.assertEqual(model, "SUPIR-v0F")
        self.assertEqual(loras, ["my_lora.safetensors"])

    def test_no_negative(self):
        text = "just a prompt\nSteps: 20, Sampler: euler, Model: TestModel"
        pos, neg, model, loras = _parse_a1111_metadata(text)
        self.assertEqual(pos, "just a prompt")
        self.assertIsNone(neg)
        self.assertEqual(model, "TestModel")

    def test_plain_text_only(self):
        text = "some prompt text with no metadata"
        pos, neg, model, loras = _parse_a1111_metadata(text)
        self.assertEqual(pos, text)
        self.assertIsNone(neg)
        self.assertIsNone(model)
        self.assertEqual(loras, [])


# ── Filename extraction ──────────────────────────────────────────────────

class TestFilenameExtraction(unittest.TestCase):

    def test_chroma_png_filename(self):
        result = _extract_from_filename(
            "chroma-photorealistic, asa akira, A realistic photograph_00001_.png"
        )
        self.assertIsNotNone(result)
        self.assertIn("photorealistic", result.prompt)

    def test_finetune_jpg_filename(self):
        result = _extract_from_filename(
            "finetune_v9_2-6-008408-jenaveve jolie, reverse congress, A photograph-sf.jpg"
        )
        self.assertIsNotNone(result)
        self.assertIn("jenaveve jolie", result.prompt)

    def test_short_filename_returns_none(self):
        self.assertIsNone(_extract_from_filename("/path/to/img.png"))


# ── Helpers ──────────────────────────────────────────────────────────────

class TestHelpers(unittest.TestCase):

    def test_decode_unicode_be(self):
        payload = "hello world".encode("utf-16-be")
        raw = b"UNICODE\x00" + payload
        self.assertEqual(_decode_user_comment(raw), "hello world")

    def test_decode_ascii(self):
        raw = b"ASCII\x00\x00\x00hello world"
        self.assertEqual(_decode_user_comment(raw), "hello world")

    def test_decode_string(self):
        self.assertEqual(_decode_user_comment("already a string"), "already a string")


if __name__ == "__main__":
    unittest.main(verbosity=2)
