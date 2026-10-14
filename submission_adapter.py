"""Stable grading interface for the SFT and DPO implementation.

Keep the function signatures below unchanged. Each function should be a thin
wrapper around your own implementation; you do not need to place the training
code itself in this file.
"""

from __future__ import annotations
from src import core

def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""
    return core.build_sft_batch(
        tokenizer=tokenizer,
        prompts=prompts,
        responses=responses,
        max_length=max_length,
    )


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""
    return core.compute_response_logprobs(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        responses=responses,
        max_length=max_length,
    )


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    return core.compute_sft_loss(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        responses=responses,
        max_length=max_length,
    )


def build_dpo_batch(
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    max_length,
):
    return core.build_dpo_batch(
        tokenizer=tokenizer,
        prompts=prompts,
        chosen_responses=chosen_responses,
        rejected_responses=rejected_responses,
        max_length=max_length,
    )


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    return core.compute_dpo_loss_from_logps(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        reference_chosen_logps=reference_chosen_logps,
        reference_rejected_logps=reference_rejected_logps,
        beta=beta,
    )


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    return core.compute_dpo_loss(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        prompts=prompts,
        chosen_responses=chosen_responses,
        rejected_responses=rejected_responses,
        beta=beta,
        max_length=max_length,
    )
