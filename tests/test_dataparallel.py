"""
tests/test_dataparallel.py
--------------------------
Unit tests verifying DataParallel wrapping, device assignment, checkpoint save/load,
and forward pass with dictionary batches.
"""

import os
import tempfile
import unittest
import torch
import torch.nn as nn
from models.core import BiSSANet
from main import (
    parse_args,
    setup_device_and_model,
    save_model_checkpoint,
    load_model_checkpoint,
)


def make_dummy_batch(batch_size=2, n_c=10, l_c=15, n_p=20, m_p=25):
    """Generate synthetic batch dict matching CPIDataset.collate_fn."""
    return {
        'COMPOUND_NODE_FEAT': torch.randn(batch_size, n_c, 78),
        'COMPOUND_ADJ': torch.eye(n_c).unsqueeze(0).repeat(batch_size, 1, 1),
        'COMPOUND_EMBEDDING': torch.randn(batch_size, l_c, 384),
        'COMPOUND_NODE_NUM': torch.full((batch_size,), n_c, dtype=torch.long),
        'COMPOUND_SMILES_LENGTH': torch.full((batch_size,), l_c, dtype=torch.long),
        'PROTEIN_EMBEDDING': torch.randn(batch_size, n_p, 480),
        'PROTEIN_NODE_NUM': torch.full((batch_size,), n_p, dtype=torch.long),
        'PROTEIN_NODE_FEAT': torch.randn(batch_size, m_p, 22),
        'PROTEIN_ADJ': torch.eye(m_p).unsqueeze(0).repeat(batch_size, 1, 1),
        'PROTEIN_STRUCT_NODE_NUM': torch.full((batch_size,), m_p, dtype=torch.long),
        'LABEL': torch.randn(batch_size, 1),
    }


class TestDataParallel(unittest.TestCase):

    def setUp(self):
        self.args = parse_args(['--dataset', 'davis', '--batch_size', '2'])

    def test_cpu_device_setup_and_forward(self):
        model = BiSSANet(self.args)
        model, device = setup_device_and_model(model, gpus_arg='cpu')
        self.assertEqual(device.type, 'cpu')
        self.assertFalse(isinstance(model, nn.DataParallel))

        batch = make_dummy_batch(batch_size=2)
        out = model(batch)
        self.assertEqual(out.shape[0], 2)

    def test_checkpoint_save_and_load_unwrapped(self):
        model = BiSSANet(self.args)
        model, device = setup_device_and_model(model, gpus_arg='cpu')

        # Wrap in mock DataParallel (even if on CPU, to test save/load logic)
        class MockDP(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.module = m

        mock_dp = MockDP(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, 'test_ckpt.pth')
            # Save from wrapped model
            save_model_checkpoint(mock_dp, ckpt_path)

            # Load into unwrapped model
            new_model = BiSSANet(self.args)
            load_model_checkpoint(new_model, ckpt_path, device)

            # Verify parameters match
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))


if __name__ == '__main__':
    unittest.main()
