"""Opt-in node lookup cache for the pinned DeepSpeed 0.18.5 ZeRO-2 hooks.

Only gradient-accumulator *nodes* are cached, never participation counts.
Every hook still asks the autograd engine which nodes participate and executes
DeepSpeed's original gradient processing and epilogue state transitions.
No installed dependency files or optimizer/checkpoint schemas are modified.
"""

import hashlib
import inspect
import logging

import torch
from torch.autograd.graph import _get_grad_fn_or_grad_acc

logger = logging.getLogger(__name__)
_HOOK_SHA = "62d6a858357daa8760e43588139517c3000041a976b7a07fd464d3769c0c97be"
_COUNT_SHA = "26fe0b690a86c06c980c2d18f56c12a7848b5702389a233abeac40604dc5551a"


class GraphParameterCounter:
    """Cache leaf nodes only within the current backward graph task.

    Parameters must remain the same leaf objects during backward, as in the
    pinned optimizer's captured parameter list. Reentrant graph-task switches
    and changes to requires_grad invalidate the cache. ``audit`` compares every
    result against the unmodified DeepSpeed function and is intentionally slow.
    """

    def __init__(self, parameters, *, audit=False):
        from deepspeed.runtime.utils import count_used_parameters_in_backward

        self.parameters = tuple(parameters)
        if any(not p.is_leaf for p in self.parameters):
            raise ValueError("Gradient node cache requires leaf optimizer parameters")
        self.audit = audit
        self.original = count_used_parameters_in_backward
        self.graph_id = None
        self.flags = None
        self.nodes = ()
        self.calls = self.rebuilds = self.audit_checks = 0

    def __call__(self):
        graph_id = torch._C._current_graph_task_id()
        if graph_id < 0:
            raise RuntimeError("Gradient counter must run inside backward")
        flags = tuple(p.requires_grad for p in self.parameters)
        if graph_id != self.graph_id or flags != self.flags:
            self.nodes = tuple({_get_grad_fn_or_grad_acc(p) for p in self.parameters if p.requires_grad})
            self.graph_id, self.flags = graph_id, flags
            self.rebuilds += 1
        count = sum(map(torch._C._will_engine_execute_node, self.nodes))
        self.calls += 1
        if self.audit:
            expected = self.original(self.parameters)
            if count != expected:
                raise RuntimeError(f"Cached gradient-node count mismatch: {count} != {expected}")
            self.audit_checks += 1
        return count


def install_zero2_grad_counter(*, audit=False):
    """Install a guarded, process-local hook factory before accelerator.prepare.

    Caller must restrict this to ZeRO-2. Return False if the identical mode was
    already installed. Version/source mismatch is an explicit error, not a
    silent patch of an unknown optimizer implementation.
    """
    import deepspeed
    from deepspeed.runtime.zero import stage_1_and_2 as zero

    cls = zero.DeepSpeedZeroOptimizer
    original = cls.create_gradient_handling_hooks
    installed = getattr(original, "_openwam_node_cache_audit", None)
    if installed is not None:
        if installed != audit:
            raise RuntimeError("DeepSpeed gradient counter already installed with a different audit mode")
        return False
    if deepspeed.__version__ != "0.18.5":
        raise RuntimeError("Gradient node cache was validated only with DeepSpeed 0.18.5")
    for fn, expected in ((original, _HOOK_SHA), (zero.count_used_parameters_in_backward, _COUNT_SHA)):
        if hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() != expected:
            raise RuntimeError(f"Unexpected DeepSpeed implementation: {fn.__name__}")
    if not hasattr(torch._C, "_current_graph_task_id") or not hasattr(torch._C, "_will_engine_execute_node"):
        raise RuntimeError("Required autograd graph inspection APIs are unavailable")

    def create_gradient_handling_hooks(self):
        if not self.partition_gradients:
            raise RuntimeError("OpenWAM gradient node cache is restricted to ZeRO-2")
        parameters = [p for group in self.bit16_groups for p in group if p.requires_grad]
        counter = GraphParameterCounter(parameters, audit=audit)
        self._openwam_grad_counter = counter
        for i, group in enumerate(self.bit16_groups):
            for param in group:
                if not param.requires_grad:
                    continue

                def wrapper(param, i):
                    def grad_handling_hook(*unused):
                        self.reenter_backward_if_needed()
                        self.process_gradients(param, i)
                        self.update_hook_state_and_maybe_run_epilogue(counter())

                    self._grad_acc_hooks.append(zero.register_grad_hook(param, grad_handling_hook))

                wrapper(param, i)
        self._remaining_grad_acc_hooks = 0

    create_gradient_handling_hooks._openwam_node_cache_audit = audit
    cls.create_gradient_handling_hooks = create_gradient_handling_hooks
    logger.info("Enabled DeepSpeed 0.18.5 ZeRO-2 gradient node cache (audit=%s)", audit)
    return True
