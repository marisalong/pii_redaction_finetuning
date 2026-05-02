# DS 593 Final Project: PII Redaction Fine-Tuning

## Table of Contents
- [About This Project](#about-this-project)
- [Repository Structure](#repository-structure)
- [Setup and Requirements](#setup-and-requirements)
- [Data](#data)
- [Key Design Decisions](#key-design-decisions)
- [Process Overview](#process-overview)
- [Findings](#findings)
- [Future Work](#future-work)
- [References](#references)

## About this project

### Goal and Problem
This project builds a PII (Personally Identifiable Information)-redaction fine-tuned LLM aimed at redacting PII from participant research visit transcripts. Many labs, including the one I work for, prioritize data sharing from participant visit audio records. However, neither the recordings nor the transcripts themselves may be released to the research community without very careful review due to PII risks. An LLM that could reliably redact any PII from transcriptions could massively accelerate data sharing among research communities. The LLM will return an annotated text document that flags PII alongside an annotation explaining why it was flagged.

### Data and Compute
The compute for this project uses an A100 GPU and 6 cores accessed via the SCC. For my fine-tuning dataset, I used the HuggingFace PII Masking 300k dataset, which includes 300k samples containing 27 different classes of PII with 749 different subjects. I generated synthetic neuropsychological visit transcripts using Claude Sonnet to be used in testing, since these better align with my intended use case.

### Methods and tools
I fine-tuned the model using the low-rank technique LoRA, using Unsloth to reduce the amount of VRAM needed during the fine-tuning process.

### Evaluation
To evaluate the model, I used the synthetic “silver-label” dataset described above as a holdout set of data from the HuggingFace PII Masking 300k dataset. My baseline for comparison is the non-fine-tuned Phi4 model with zero-shot prompting. The performance metrics calculated on the PII redaction tasks include recall, precision, and F1. I also manually reviewed the outputs to get a sense of the types of errors the model made and how it could be improved in the future.

## Repository Structure

- `notebooks/`: contains the Jupyter notebooks for data preprocessing, fine-tuning, and evaluation
- `zip/`: contains large files (datasets and model checkpoints) that need to be unzipped before use. These are not included in the repo due to size, but may be downloaded from [Google Drive here](https://drive.google.com/drive/folders/1PoiarVDBgWFKhPITKRSo7pkck_b_XMWK?usp=sharing)
- `documents/`: contains project documentation and reports

## Setup and Requirements

This project is designed to run in **Google Colab** with a GPU runtime, or on a university HPC cluster (tested on BU SCC). A free-tier T4 GPU is sufficient for inference and evaluation. Fine-tuning requires at least 16GB VRAM; a T4 (16GB) works when adjusting the batch size settings.

### Dependencies

All notebooks install their own dependencies at the top of the file. The core libraries are:

```
unsloth
transformers
datasets
torch
scikit-learn
```

### Downloading Model Checkpoints and Data

The fine-tuned adapter and processed datasets are stored in `zip/` and are not included in this repository due to file size. Please download and unzip the contents prior to running these notebooks.

Alternatively, all data can be reproduced from scratch by running the notebooks in order described below.


## About the data

## Key Design Decisions
### Generative tagging over token classification

Rather than framing PII detection as a token classification task (a traditional NER approach), this project frames it as a generative sequence-to-sequence task. The model receives raw text and outputs the same text with PII spans wrapped in tags. This approach handles long, complex clinical sentences more naturally and avoids the label alignment complexity of token classification with subword tokenizers.The tradeoff is that evaluation is more complex and inference is slower than a classification head.

### LoRA fine-tuning with Unsloth

Fine-tuning used [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation), which updates a small set of additional weights rather than the full model. This dramatically reduces memory requirements and training time while preserving base model quality. LoRA was implemented using [Unsloth](https://github.com/unslothai/unsloth), which provides optimized training kernels that reduce VRAM usage by ~60% and increase training speed by up to 2x compared to standard HuggingFace training.

### Synthetic data only

All data used in this project is synthetic. This choice makes the entire pipeline shareable without risk of exposing real participant information. The cost is that real clinical transcripts are likely noisier and more variable than the synthetic examples used here.

### Token-level evaluation

Evaluation compares individual PII tokens rather than whole spans. This handles the common model behavior of splitting a single entity across multiple tags (e.g., `<NAME>Sherry</NAME> <NAME>Rosen</NAME>`) without incorrectly penalizing it. A hallucination check is also applied: predicted tokens that do not appear in the source text are flagged separately.

## Process Overview
### 1. Data Preprocessing (`ai4privacy_dataset_preprocessing.ipynb`)
- Load and filter the ai4privacy dataset to English examples
- Map original tags to HIPAA-aligned schema; discard out-of-scope tags
- Build per-tag boolean columns for stratified evaluation
- Construct a balanced validation split (70% PII / 30% clean; minimum 75 examples per tag type)

### 2. Fine-Tuning (`finetune_with_unsloth.ipynb`)
- Load Phi-4 via Unsloth with 4-bit quantization
- Apply LoRA adapters and train on ~3,000 examples
- Save adapter weights to `zip/checkpoint/`

### 3. Inference (`evaluation.ipynb`)
- Load fine-tuned model via Unsloth
- Run batched generation on both validation sets
- Save predictions incrementally to `.jsonl` to guard against crashes

### 4. Evaluation (`evaluation.ipynb`)
- Parse predicted tags from generated output
- Compute token-level precision, recall, and F1 per tag type
- Compare performance across both evaluation tiers

### Preparing the dataset

## Data

### Training Data — ai4privacy PII Masking 300k

The primary dataset is the [ai4privacy PII Masking 300k](https://huggingface.co/datasets/ai4privacy/pii-masking-300k) dataset from HuggingFace, which contains 300,000 samples spanning 27 PII categories. Preprocessing steps included:

- Filtering to English-language examples only
- Mapping ai4privacy tag categories to a reduced HIPAA-aligned schema (see tag table below)
- Discarding tag types outside the HIPAA PHI framework
- Splitting into ~3,000 training examples and a large held-out evaluation pool

### Evaluation Data — Synthetic NP Transcripts

100 synthetic neuropsychological exam transcripts were generated using Claude Sonnet and manually reviewed for label accuracy. These transcripts represent the target deployment domain and serve as the evaluation set. They are intentionally more challenging than the ai4privacy examples. The PII is embedded in long, naturalistic clinical prose rather than short declarative sentences.

### PII Tag Schema

Tags were derived from HIPAA's 18 Protected Health Information identifiers:

| Tag | Description |
|-----|-------------|
| `NAME` | Patient, caregiver, or clinician names |
| `TITLE` | Professional titles (Dr., Mr., Mrs.) |
| `DOB` | Date of birth |
| `DATE` | Appointment or visit dates |
| `TIME` | Times |
| `EMAIL` | Email addresses |
| `PHONE` | Phone numbers |
| `ADDRESS` | Street addresses |
| `LOCATION` | Cities, facilities, institutions |
| `SSN` | Social Security Numbers |
| `UNIQUE_ID` | Medical record numbers, patient IDs |
| `LICENSE_NUMBER` | NPI or driver's license numbers |
| `IP_ADDRESS` | IP addresses |
| `USERNAME` | Usernames |
| `PASS` | Passwords or PINs |
| `SEX` | Sex or gender identifiers |
| `CARDISSUER` | Credit/debit card issuers |

---

## Key Findings
### Performance Gap Between Evaluation Tiers

The fine-tuned model performs well on held-out ai4privacy examples but shows a gap on the synthetic NP transcripts. Most errors on the NP transcripts are false negatives, meaning the model misses PII rather than hallucinating or misclassifying it.

| Evaluation Set | Precision | Recall | F1 |
|----------------|-----------|--------|----|
| ai4privacy held-out (n=1,200) | [INSERT] | [INSERT] | [INSERT] |
| Synthetic NP transcripts (n=100) | [INSERT] | [INSERT] | [INSERT] |

Tags most affected on NP transcripts: `NAME`, `DOB/DATE`, `SSN`.

### Why the Gap Exists

The ai4privacy dataset consists of short, syntactically simple sentences where PII appears in predictable positions. The model learned these surface patterns effectively. Neuropsychological transcripts embed PII in long clinical narratives — names appear mid-sentence in complex syntax, dates are written out rather than formatted, and identifiers follow clinical shorthand. The model was not exposed to this register during training.

This result only became apparent once the NP transcript evaluation tier was introduced. Performance on ai4privacy alone looked strong, which would have given a misleading picture of real-world viability. This underscores the importance of domain-specific evaluation sets.

### Story Recall Test

One hypothesis was that the story recall test — where participants recount a standardized story containing a character name and location — might cause the model to over-redact fictional entities as PII. This was **not confirmed** as a significant failure mode. The dominant failure was systematic under-detection, not over-detection.


## Future Work
- **Few-shot prompting experiments** — test whether in-context clinical examples close the NP transcript gap without additional fine-tuning
- **Safety evaluation** — evaluate whether fine-tuning degraded safety guardrails using an adversarial prompt set
- **Domain-specific training data** — add 200–500 NP-style examples to the fine-tuning set; expected to close a significant portion of the performance gap
- **Expand the NP transcript set** — grow from 100 to 500+ examples to enable per-tag stratified analysis on the clinical domain
- **Real data evaluation** — evaluate on real, de-identified transcripts within a secure local compute environment

## References
- ai4privacy. (2023). [PII Masking 300k](https://huggingface.co/datasets/ai4privacy/pii-masking-300k). HuggingFace Datasets.
- Hu, E. J., et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). arXiv:2106.09685.
- Unsloth AI. (2024). [Fine-tuning a language model with Unsloth](https://huggingface.co/blog/aifeifei798/fine-tuning-a-language-model-with-unsloth).
- U.S. Department of Health and Human Services. [HIPAA Privacy Rule: De-identification of Protected Health Information](https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html).
- Microsoft Research. (2024). Phi-4 Technical Report.
