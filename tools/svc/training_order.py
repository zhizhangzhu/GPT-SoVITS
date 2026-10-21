"""Stable per-epoch sample ordering, including interrupted epoch recovery."""
import random


def epoch_order(size, seed, epoch):
    order = list(range(size))
    random.Random(f"svc-v2:{seed}:{epoch}").shuffle(order)
    return order


def resumed_order(size, seed, epoch, state=None):
    """Old checkpoints finish their current sequential epoch before shuffling."""
    if state is not None and int(state["epoch"]) == epoch:
        saved = state.get("sample_order")
        if saved is None:
            return list(range(size))
        if not isinstance(saved, list) or len(saved) != size or set(saved) != set(range(size)):
            raise ValueError("训练检查点的样本顺序损坏，已停止续训")
        return saved
    return epoch_order(size, seed, epoch)
