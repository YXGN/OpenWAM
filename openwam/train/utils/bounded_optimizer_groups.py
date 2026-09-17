"""Bound optimizer groups without changing parameter order or hyperparameters."""


def bound_optimizer_groups(groups, max_numel=128_000_000):
    groups = list(groups)
    if groups and not isinstance(groups[0], dict):
        groups = [{"params": groups}]
    result = []
    original = []
    for group in groups:
        chunk = []
        size = 0
        for parameter in group["params"]:
            original.append(id(parameter))
            if parameter.numel() > max_numel:
                raise ValueError("one parameter exceeds the optimizer group bound")
            if chunk and size + parameter.numel() > max_numel:
                result.append({**group, "params": chunk})
                chunk = []
                size = 0
            chunk.append(parameter)
            size += parameter.numel()
        if chunk:
            result.append({**group, "params": chunk})
    actual = [id(parameter) for group in result for parameter in group["params"]]
    if actual != original or len(actual) != len(set(actual)):
        raise RuntimeError("optimizer parameter order or uniqueness changed while bounding groups")
    return result
