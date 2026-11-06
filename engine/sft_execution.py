"""Regroup one materialized SFT optimizer step without changing its samples.

The sampler and stochastic window selection remain the caller's responsibility.
Only CPU tensors from an already materialized effective batch are accepted here;
this module never moves data to a device, pins memory, or commits a data cursor.
"""
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch


_INPUT_NAMES = ("input_ids", "labels", "attention_mask")
_PADDING_MULTIPLES = (16, 32, 64, 128)


def _validate_options(execution_batch_size, padding_multiple, sort_by_length):
    if (not isinstance(execution_batch_size, int) or isinstance(execution_batch_size, bool)
            or execution_batch_size < 1):
        raise ValueError("execution_batch_size must be a positive integer")
    if (not isinstance(padding_multiple, int) or isinstance(padding_multiple, bool)
            or padding_multiple not in _PADDING_MULTIPLES):
        raise ValueError("padding_multiple must be one of 16, 32, 64, 128")
    if not isinstance(sort_by_length, bool):
        raise ValueError("sort_by_length must be a boolean")


@dataclass(frozen=True)
class SftExecutionPolicy:
    execution_batch_size: int = 1
    padding_multiple: int = 128
    sort_by_length: bool = False
    max_tokens_per_group: int = 1536

    def __post_init__(self):
        _validate_options(self.execution_batch_size, self.padding_multiple, self.sort_by_length)
        if (not isinstance(self.max_tokens_per_group, int) or isinstance(self.max_tokens_per_group, bool)
                or self.max_tokens_per_group < 2):
            raise ValueError("max_tokens_per_group must be an integer >= 2")


def validate_sft_execution(
    execution_batch_size=None,
    padding_multiple=128,
    sort_by_length=False,
    *,
    stage,
    dropout,
    torch_compile,
    micro_batch_size,
    trim_sft_padding,
    max_tokens_per_group=1536,
) -> Optional[SftExecutionPolicy]:
    """Return an explicit frozen-SFT execution policy, or None for legacy mode.

    Omitting execution_batch_size leaves every existing training stage unchanged.
    Execution regrouping is deliberately separate from semantic micro-batching:
    callers retain B1 sampling and per-example loss weighting while executing
    several independent examples together.
    """
    if execution_batch_size is None:
        SftExecutionPolicy(1, padding_multiple, sort_by_length, max_tokens_per_group)
        if padding_multiple != 128 or sort_by_length or max_tokens_per_group != 1536:
            raise ValueError("Nondefault padding_multiple/sort_by_length requires execution_batch_size")
        return None
    policy = SftExecutionPolicy(execution_batch_size, padding_multiple, sort_by_length, max_tokens_per_group)
    if (stage != "sft" or dropout != 0.0 or torch_compile is not False
            or not isinstance(micro_batch_size, int) or isinstance(micro_batch_size, bool)
            or micro_batch_size != 1 or trim_sft_padding is not True):
        raise ValueError(
            "execution_batch_size requires stage=sft, dropout=0, torch_compile=false, "
            "semantic micro_batch_size=1 and trim_sft_padding=true"
        )
    return policy


@dataclass(frozen=True)
class SftExecutionGroup:
    batch: Dict[str, torch.Tensor]
    sample_count: int
    source_indices: Tuple[int, ...]
    loss_weight: float


@dataclass(frozen=True)
class SftExecutionPlan:
    groups: Tuple[SftExecutionGroup, ...]
    total_samples: int
    actual_tokens: int
    physical_tokens: int


def _validate_source_batch(batch: Mapping[str, torch.Tensor], index: int):
    prefix = f"SFT execution sample {index}"
    if not isinstance(batch, Mapping) or set(batch) != set(_INPUT_NAMES):
        raise ValueError(prefix + " requires exactly input_ids, labels and attention_mask")
    if any(not isinstance(batch[name], torch.Tensor) for name in _INPUT_NAMES):
        raise ValueError(prefix + " requires tensors")
    ids, labels, mask = (batch[name] for name in _INPUT_NAMES)
    if (ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 2
            or labels.shape != ids.shape or mask.shape != ids.shape):
        raise ValueError(prefix + " requires matching semantic B1 tensors with T >= 2")
    if any(batch[name].device.type != "cpu" for name in _INPUT_NAMES):
        raise ValueError(prefix + " must be materialized on CPU before execution regrouping")
    if ids.dtype != torch.long or labels.dtype != torch.long:
        raise ValueError(prefix + " requires torch.long input_ids and labels")
    if mask.is_complex() or not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError(prefix + " requires a binary attention_mask")
    valid = mask.bool()
    if bool((~valid[:, :-1] & valid[:, 1:]).any()):
        raise ValueError(prefix + " requires contiguous right padding")
    length = int(valid.sum())
    if length < 2:
        raise ValueError(prefix + " requires at least two valid tokens")
    if bool((labels[~valid] != -100).any()):
        raise ValueError(prefix + " has supervised padding targets")
    if not bool((labels[:, 1:length] != -100).any()):
        raise ValueError(prefix + " has no supervised next-token targets")
    return length


def build_sft_execution_step(
    cpu_batches: Sequence[Mapping[str, torch.Tensor]],
    policy: SftExecutionPolicy,
    *,
    pad_token_id: int,
) -> SftExecutionPlan:
    """Group exactly the supplied step's B1 examples into independent rows.

    Every valid prefix and every label is copied intact; only right PAD is
    recreated. Length sorting is stable and bounded to this call. Each group's
    loss_weight is sample_count / total_samples: it must multiply a *mean of
    per-example CE*, not a CE averaged over all target tokens in the group.
    """
    if not isinstance(policy, SftExecutionPolicy):
        raise ValueError("SFT execution requires a validated SftExecutionPolicy")
    if not isinstance(pad_token_id, int) or isinstance(pad_token_id, bool) or pad_token_id < 0:
        raise ValueError("pad_token_id must be a nonnegative integer")
    if not isinstance(cpu_batches, Sequence) or not cpu_batches:
        raise ValueError("SFT execution requires a nonempty materialized optimizer step")
    lengths = [_validate_source_batch(batch, index) for index, batch in enumerate(cpu_batches)]
    mask_dtype = cpu_batches[0]["attention_mask"].dtype
    if any(batch["attention_mask"].dtype != mask_dtype for batch in cpu_batches):
        raise ValueError("SFT execution attention_mask dtypes must agree across the step")
    order = list(range(len(cpu_batches)))
    if policy.sort_by_length:
        order.sort(key=lengths.__getitem__)
    groups = []
    physical_tokens = 0
    total_samples = len(cpu_batches)
    start = 0
    while start < total_samples:
        count = min(policy.execution_batch_size, total_samples - start)
        while True:
            indices = tuple(order[start:start + count])
            longest = max(lengths[index] for index in indices)
            width = ((longest + policy.padding_multiple - 1) // policy.padding_multiple) * policy.padding_multiple
            # Retain a non-multiple source cap rather than adding another bucket.
            width = min(width, max(cpu_batches[index]["input_ids"].shape[1] for index in indices))
            if count == 1 or count * width <= policy.max_tokens_per_group:
                break
            # Reduce execution size, never truncate a real token or drop a row.
            count = max(1, count // 2)
        count = len(indices)
        ids = torch.full((count, width), pad_token_id, dtype=torch.long)
        labels = torch.full((count, width), -100, dtype=torch.long)
        mask = torch.zeros((count, width), dtype=mask_dtype)
        for row, index in enumerate(indices):
            length = lengths[index]
            source = cpu_batches[index]
            ids[row, :length].copy_(source["input_ids"][0, :length])
            labels[row, :length].copy_(source["labels"][0, :length])
            mask[row, :length].copy_(source["attention_mask"][0, :length])
        groups.append(SftExecutionGroup(
            batch={"input_ids": ids, "labels": labels, "attention_mask": mask},
            sample_count=count,
            source_indices=indices,
            loss_weight=count / total_samples,
        ))
        physical_tokens += count * width
        start += count
    return SftExecutionPlan(tuple(groups), total_samples, sum(lengths), physical_tokens)


def materialize_sft_execution_step(
    loader,
    data_iter,
    batch_sampler,
    grad_accum_steps,
    policy: SftExecutionPolicy,
    *,
    pad_token_id: int,
):
    """Read one original B1 effective batch before regrouping its execution.

    Fetch and commit each semantic sample in the same order as the legacy
    accumulation loop, including samples that cross an epoch boundary. No model
    work occurs here. The caller must treat materialization and every execution
    group as one optimizer-step transaction: on failure restore its starting
    cursor and RNG, rebuild the loader iterator, and replay the whole step.
    """
    if (not isinstance(grad_accum_steps, int) or isinstance(grad_accum_steps, bool)
            or grad_accum_steps < 1):
        raise ValueError("SFT execution grad_accum_steps must be a positive integer")
    if not isinstance(policy, SftExecutionPolicy):
        raise ValueError("SFT execution requires a validated SftExecutionPolicy")
    if getattr(batch_sampler, "batch_size", None) != 1:
        raise ValueError("SFT execution materialization requires the original B1 sampler")
    if getattr(loader, "num_workers", 0) != 0:
        raise ValueError("SFT execution materialization requires main-process data loading")
    batches = []
    for index in range(grad_accum_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        _validate_source_batch(batch, index)
        batches.append(batch)
        batch_sampler.advance(1)
    plan = build_sft_execution_step(batches, policy, pad_token_id=pad_token_id)
    return plan, data_iter
