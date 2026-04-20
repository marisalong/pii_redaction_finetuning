# DS 593 Final Project: PII Redaction Fine-Tuning

## Goal and Problem
This project builds a PII (Personally Identifiable Information)-redaction fine-tuned LLM aimed at redacting PII from participant research visit transcripts. Many labs, including the one I work for, prioritize data sharing from participant visit audio records. However, neither the recordings nor the transcripts themselves may be released to the research community without very careful review due to PII risks. An LLM that could reliably redact any PII from transcriptions could massively accelerate data sharing among research communities. The LLM will return an annotated text document that flags PII alongside an annotation explaining why it was flagged.

## Data and Compute
The compute for this project will use the NVIDIA T4 GPU provided by the free tier of Google Colab, supplemented GPUs access via the SCC if necessary. For my fine-tuning dataset, I will use the HuggingFace PII Masking 300k dataset, which includes 300k samples containing 27 different classes of PII with 749 different subjects. I will also generate synthetic neuropsychological visit transcripts to be used in testing, since these will better align with my intended use case. I will manually create a few examples and then use Claude Sonnet to help me generate additional transcripts.

## Methods and tools
I will experiment with few-shot prompting techniques that integrate my manually-created transcripts to create my “silver-label” dataset. I will fine-tune the model first using the low-rank technique LoRA, but will switch to QLoRA if needed to reduce the amount of VRAM needed during the fine-tuning process. After fine-tuning, I experiment with one-shot and few-shot user prompts to see if I am able to further optimize results.

## Evaluation
To evaluate the model, I will use the synthetic “silver-label” dataset described above as a holdout set of data from the HuggingFace PII Masking 300k dataset (47.7k rows of data). My baseline for comparison will be the non-fine-tuned Phi4 model with zero-shot prompting. The performance metrics calculated on the PII redaction tasks will include accuracy, recall, precision, and F1. Additionally, I will ensure that the safety guardrails that may be undone through the fine-tuning process are accounted for by testing the original and fine-tuned models with 100 prompts using the safety prompts dataset. I will track the number of prompts correctly rejected by each model, with failure to respond to adverse prompts being considered a success.
