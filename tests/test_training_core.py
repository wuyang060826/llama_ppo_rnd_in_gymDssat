"""CPU regressions for numerical edge cases in the main PPO implementation."""
import unittest
import numpy as np
import torch
from main.llama_ppo_rnd.training_core import PPOBuffer, CosineAnnealingWarmup


class TrainingCoreTests(unittest.TestCase):
    def test_single_transition_is_finite(self):
        buffer = PPOBuffer(.99, .95, torch.device("cpu"))
        state = np.zeros(25)
        buffer.add(state, 0, 1, state, True, 0, .2, "state")
        advantages, returns = buffer.compute_gae_and_normalize(0)
        self.assertTrue(torch.isfinite(advantages).all())
        self.assertTrue(torch.isfinite(returns).all())
        self.assertEqual(advantages.item(), 0)

    def test_terminal_blocks_next_episode_value(self):
        buffer = PPOBuffer(1, 1, torch.device("cpu"))
        state = np.zeros(25)
        buffer.add(state, 0, 0, state, True, 0, 5, "s")
        buffer.add(state, 0, 2, state, True, 0, 100, "s")
        _, returns = buffer.compute_gae_and_normalize(999, use_intrinsic=False)
        np.testing.assert_allclose(returns.numpy(), [-1, 1], atol=1e-5)

    def test_empty_rollout_has_explicit_error(self):
        with self.assertRaises(ValueError):
            PPOBuffer(.99, .95, torch.device("cpu")).compute_gae_and_normalize(0)

    def test_schedule_stays_at_floor_past_horizon(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([parameter], lr=1)
        schedule = CosineAnnealingWarmup(optimizer, 0, 2, .1)
        for _ in range(8):
            schedule.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], .1)


if __name__ == "__main__":
    unittest.main()
