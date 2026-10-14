from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def _tokenize_str(tokenizer, text: str) -> list[int]:
    """Tokenizes a string into token IDs without special tokens."""
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """
    Constructs an SFT batch using 1D tensor cloning and torch.nn.utils.rnn.pad_sequence.
    """
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0

    if len(prompts) == 0:
        empty = torch.empty((0, 0), dtype=torch.long)
        return {"input_ids": empty, "attention_mask": empty, "labels": empty}

    input_tensors = []
    label_tensors = []
    mask_tensors = []

    for prompt, response in zip(prompts, responses):
        p_ids = _tokenize_str(tokenizer, prompt)
        r_ids = _tokenize_str(tokenizer, response)

        # Append EOS token before applying max_length truncation
        if eos_id is not None:
            r_ids = r_ids + [eos_id]

        full_seq = p_ids + r_ids
        if max_length is not None:
            full_seq = full_seq[:max_length]

        seq_tensor = torch.tensor(full_seq, dtype=torch.long)

        # Clone input_ids for labels and mask out the prompt prefix with -100
        lbl_tensor = seq_tensor.clone()
        prompt_cutoff = min(len(p_ids), len(full_seq))
        lbl_tensor[:prompt_cutoff] = -100

        att_tensor = torch.ones_like(seq_tensor, dtype=torch.long)

        input_tensors.append(seq_tensor)
        label_tensors.append(lbl_tensor)
        mask_tensors.append(att_tensor)

    # Batch-pad all 1D tensors to the longest sequence in the batch
    input_ids = pad_sequence(input_tensors, batch_first=True, padding_value=pad_id)
    labels = pad_sequence(label_tensors, batch_first=True, padding_value=-100)
    attention_mask = pad_sequence(mask_tensors, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _score_response_tokens(model, batch):
    """
    Computes per-token log-probabilities for unmasked response tokens using log_softmax + gather.
    Returns:
        token_logps: (B, L-1) tensor of log-probabilities (0.0 at masked positions)
        valid_mask:  (B, L-1) boolean tensor indicating active response tokens
    """
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # Shift so token t predicts token t+1
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]

    valid_mask = shift_labels.ne(-100)
    # Replace -100 with 0 temporarily so torch.gather has valid indices
    gather_indices = shift_labels.masked_fill(~valid_mask, 0).unsqueeze(-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered_logps = torch.gather(log_probs, dim=-1, index=gather_indices).squeeze(-1)

    # Zero out masked prompt and padding positions
    token_logps = gathered_logps * valid_mask.to(gathered_logps.dtype)
    return token_logps, valid_mask


def compute_response_logprobs_from_batch(model, batch):
    """Sums response-only token log-probabilities per example from a prebuilt batch."""
    token_logps, _ = _score_response_tokens(model, batch)
    return token_logps.sum(dim=-1)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Builds the batch and computes per-example summed response log-probabilities."""
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    return compute_response_logprobs_from_batch(model, batch)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Computes token-level mean negative log-likelihood over response tokens."""
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    token_logps, valid_mask = _score_response_tokens(model, batch)
    num_response_tokens = valid_mask.sum().clamp(min=1)
    return -token_logps.sum() / num_response_tokens


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Constructs paired chosen and rejected batches for DPO."""
    return {
        "chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length),
        "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length),
    }


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Computes batch-mean DPO loss from policy and reference sequence log-probs."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    margins = beta * (pi_logratios - ref_logratios)
    return -F.logsigmoid(margins).mean()


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
    """Computes end-to-end DPO loss with frozen reference model evaluation."""
    batches = build_dpo_batch(
        tokenizer, prompts, chosen_responses, rejected_responses, max_length
    )

    pi_chosen = compute_response_logprobs_from_batch(policy_model, batches["chosen"])
    pi_rejected = compute_response_logprobs_from_batch(policy_model, batches["rejected"])

    with torch.no_grad():
        ref_chosen = compute_response_logprobs_from_batch(
            reference_model, batches["chosen"]
        ).to(pi_chosen.device)
        ref_rejected = compute_response_logprobs_from_batch(
            reference_model, batches["rejected"]
        ).to(pi_rejected.device)

    return compute_dpo_loss_from_logps(
        pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta
    )