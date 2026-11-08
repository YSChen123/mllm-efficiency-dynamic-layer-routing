import math
import torch


def epoch_batches(epoch, batch_size, count=5000):
    if type(batch_size) is not int or batch_size <= 0 or count <= 0:
        raise ValueError('Positive batch size/count required')
    order = torch.randperm(count, generator=torch.Generator().manual_seed(epoch)).tolist()
    return [order[i:i + batch_size] for i in range(0, count, batch_size)]


def accumulated_update(network, optimizer, ids, loss_for_ids, micro_batch_size=4):
    if not ids or micro_batch_size <= 0:
        raise ValueError('Nonempty update required')
    optimizer.zero_grad(set_to_none=True)
    value = 0.
    for start in range(0, len(ids), micro_batch_size):
        part = ids[start:start + micro_batch_size]
        loss = loss_for_ids(part)
        if loss.ndim or not torch.isfinite(loss):
            raise ValueError('Expected finite scalar loss')
        weight = len(part) / len(ids)
        (loss * weight).backward()
        value += float(loss.detach()) * weight
    if not math.isfinite(value) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in network.parameters()):
        raise ValueError('Nonfinite accumulated update')
    optimizer.step()
    return value
