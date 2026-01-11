import unittest
import torch
import shutil
import tempfile
import os
from transformers import AutoTokenizer, AutoConfig
from encoder_trainer.config import ModelConfig, DataConfig
from encoder_trainer.data.collators import DynamicLabelCollator
from encoder_trainer.modeling.encoder import UnifiedEncoderClassifier

class TestEncoderTrainer(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.model_name = "prajjwal1/bert-tiny" # Small model for tests
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_collator_fixed_labels(self):
        collator = DynamicLabelCollator(self.tokenizer)
        features = [
            {"text": "hello world", "labels": 0},
            {"text": "foo bar", "labels": 1}
        ]
        batch = collator(features)
        self.assertIn("input_ids", batch)
        self.assertIn("labels", batch)
        self.assertEqual(batch["labels"].shape[0], 2)

    def test_collator_sce_candidates(self):
        collator = DynamicLabelCollator(self.tokenizer, add_candidates=True)
        features = [
            {"text": "Find movie", "candidates": ["Search", "Play", "Stop"], "target_index": 0},
            {"text": "Stop music", "candidates": ["Play", "Stop"], "target_index": 1}
        ]
        batch = collator(features)
        
        self.assertIn("candidate_input_ids", batch)
        # Check shapes: Batch=2, MaxK=3
        self.assertEqual(batch["candidate_input_ids"].shape[0], 2)
        self.assertEqual(batch["candidate_input_ids"].shape[1], 3)
        self.assertIn("candidate_set_mask", batch)
        
        # Check mask for 2nd sample (only 2 candidates)
        self.assertEqual(batch["candidate_set_mask"][1].tolist(), [1, 1, 0])

    def test_model_frozen_head_forward(self):
        config = ModelConfig(model_name_or_path=self.model_name, mode="frozen_head")
        model = UnifiedEncoderClassifier(config, num_labels=2)
        
        input_ids = torch.randint(0, 100, (2, 10))
        mask = torch.ones((2, 10))
        
        output = model(input_ids, mask)
        self.assertIn("logits", output)
        self.assertEqual(output["logits"].shape, (2, 2))

    def test_model_sce_forward(self):
        config = ModelConfig(model_name_or_path=self.model_name, mode="sce", sce_hidden_dim=32)
        model = UnifiedEncoderClassifier(config)
        
        # Batch=2, Seq=10
        input_ids = torch.randint(0, 100, (2, 10))
        mask = torch.ones((2, 10))
        
        # Candidates: Batch=2, K=3, Seq=5
        cand_ids = torch.randint(0, 100, (2, 3, 5))
        cand_mask = torch.ones((2, 3, 5))
        cand_set_mask = torch.ones((2, 3))
        
        output = model(
            input_ids=input_ids,
            attention_mask=mask,
            candidate_input_ids=cand_ids,
            candidate_attention_mask=cand_mask,
            candidate_set_mask=cand_set_mask
        )
        self.assertIn("logits", output)
        # Output should be [B, K]
        self.assertEqual(output["logits"].shape, (2, 3))

if __name__ == "__main__":
    unittest.main()
