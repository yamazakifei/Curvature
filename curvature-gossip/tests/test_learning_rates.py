"""Verify independent Actor/Critic learning rates and legacy YAML fallback."""

from curvature_gossip.learning.ctde_ppo import CTDEPPO, resolve_learning_rates


def test_learning_rate_resolution_keeps_legacy_shared_setting():
    """A legacy training.learning_rate must configure both optimizers equally."""
    assert resolve_learning_rates({"learning_rate": 1e-3}) == (1e-3, 1e-3)
    assert resolve_learning_rates({"learning_rate": 1e-3, "actor_learning_rate": 2e-4}) == (2e-4, 1e-3)


def test_learning_rate_resolution_accepts_independent_settings():
    """Explicit Actor and Critic values must be returned independently."""
    assert resolve_learning_rates({
        "actor_learning_rate": 2e-4,
        "critic_learning_rate": 1e-3,
    }) == (2e-4, 1e-3)


def test_ctdeppo_constructs_with_independent_learning_rates():
    """The PPO graph records both rates while retaining the shared constructor API."""
    model = CTDEPPO(
        actor_learning_rate=2e-4,
        critic_learning_rate=1e-3,
        seed=101,
    )
    try:
        assert model.actor_learning_rate == 2e-4
        assert model.critic_learning_rate == 1e-3
    finally:
        model.close()

