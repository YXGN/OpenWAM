import copy

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from openwam.train.utils.deepspeed_grad_counter import GraphParameterCounter, install_zero2_grad_counter


@pytest.mark.parametrize("reentrant", [None, False, True])
def test_counts_gradients_updates_and_accumulation(reentrant):
    torch.manual_seed(12)
    initial = torch.nn.ModuleList([torch.nn.Linear(4, 4, dtype=torch.float64) for _ in range(3)])

    def run(cached):
        model = copy.deepcopy(initial)
        params = list(model.parameters())
        # Include a duplicate and an unused module, as the original deduplicates nodes.
        counter = GraphParameterCounter(params + [params[0]], audit=True)

        def count_hook(p):
            counter()

        hooks = [p.register_post_accumulate_grad_hook(count_hook) for p in params] if cached else []
        optimizer = torch.optim.AdamW(params, lr=1e-3)
        records = []
        for i in range(6):
            x = torch.arange(8, dtype=torch.float64).reshape(2, 4).requires_grad_()

            def fn(x):
                y = model[0](x).tanh()
                # Reuse a module in one graph; alternate which other module participates.
                return model[0](y) + model[1 + i % 2](x)

            y = fn(x) if reentrant is None else checkpoint(fn, x, use_reentrant=reentrant)
            loss = y.square().mean()
            loss.backward()
            records.append((loss.detach().clone(), [None if p.grad is None else p.grad.clone() for p in params]))
            if i % 2:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        for h in hooks:
            h.remove()
        return model.state_dict(), records, counter

    expected, old_records, _ = run(False)
    actual, records, counter = run(True)
    assert counter.audit_checks == counter.calls > 0
    assert counter.rebuilds >= 6
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-12, atol=1e-12)
    for (a_loss, a_grads), (b_loss, b_grads) in zip(records, old_records):
        torch.testing.assert_close(a_loss, b_loss, rtol=1e-12, atol=1e-12)
        for a, b in zip(a_grads, b_grads):
            if a is None or b is None:
                assert a is None and b is None
            else:
                torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_counter_rejects_outside_backward():
    counter = GraphParameterCounter([torch.nn.Parameter(torch.ones(2))])
    with pytest.raises(RuntimeError, match="inside backward"):
        counter()


def test_installer_guard_and_idempotence(monkeypatch):
    from deepspeed.runtime.zero import stage_1_and_2 as zero

    original = zero.DeepSpeedZeroOptimizer.create_gradient_handling_hooks
    monkeypatch.setattr(zero.DeepSpeedZeroOptimizer, "create_gradient_handling_hooks", original)
    assert install_zero2_grad_counter()
    assert not install_zero2_grad_counter()
    with pytest.raises(RuntimeError, match="different audit mode"):
        install_zero2_grad_counter(audit=True)
    monkeypatch.setattr(zero.DeepSpeedZeroOptimizer, "create_gradient_handling_hooks", lambda self: None)
    with pytest.raises(RuntimeError, match="Unexpected DeepSpeed"):
        install_zero2_grad_counter()
