# SFT + DPO on Small Language Models (GenAI PA2)

Supervised fine-tuning of Qwen/Qwen2.5-0.5B on UltraChat, two from-scratch DPO runs
(HH-RLHF and UltraFeedback) starting from the SFT checkpoint, and LoRA persona DPO on
HuggingFaceTB/SmolLM2-360M-Instruct with TRL. The full write-up is in the submitted report.

## Code
- `src/core.py`: SFT batching with response-only labels, response log-probabilities,
  SFT loss, DPO loss. Runs on CPU.
- `submission_adapter.py`: required root-level wrapper that calls into `src/core.py`.
- SFT and both DPO training loops are plain PyTorch inside the notebooks
  (no Hugging Face Trainer or TRL DPOTrainer). TRL + PEFT are used only for the LoRA task.

## Tests
pip install pytest
PYTHONPATH=. python -m pytest -q tests/ 
## Notebooks (Kaggle, GPU T4 x2, run in this order)
| Notebook | Task | Outputs |
|---|---|---|
| 01_setup_baseline_sft | 1-3: setup, base-model generations, SFT | results/sft |
| 02_dpo_hh | 4: DPO on HH-RLHF | results/dpo_hh |
| 03_dpo_uf | 4: DPO on UltraFeedback | results/dpo_uf |
| 04_lora_dpo_run1_lr5e-5 | 5: LoRA DPO, submitted run | results/lora |
| 05_lora_dpo_run2_lr1e-5 | 5: LoRA DPO, gentler pilot | results/lora_lr1e-5 |
| 06_generate_eval_outputs | 6: SFT and DPO generations, A/B pair files | results/gen |
| 07_judge_dpo_vs_sft | 7: Gemini judge, DPO vs SFT | results/judge |
| 08_style_judge_lora | 7: style/quality judge for the LoRA runs | printed in notebook; numbers in report |

Each later notebook uses the previous notebook's saved Kaggle output as an input dataset.
Seed 42 throughout. Judge: google/gemini-3.6-flash via kaggle_benchmarks.

## Results
- `results/<run>/`: metrics, run summaries, data statistics, training curves
- `results/gen/`: fixed evaluation sets, generations (including base-model AlpacaEval outputs), A/B pair files
- `results/judge/`: judge configuration, `judge_summary.json` (all DPO-vs-SFT numbers), example pairs

Model checkpoints, per-pair judged files and LoRA style-judge files are not in the repo;
they were Kaggle session outputs. The LoRA judge results are reported in the report.
