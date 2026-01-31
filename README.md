# Brain-Score Meets Representational Similarity Analysis

Code for ICASSP 2026 paper:

**"Brain-Score Meets Representational Similarity Analysis: A Methodological Convergence in Model-Brain Alignment"**

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

- **DERCo (EEG)**: https://osf.io/rkqbu/
- **Podcast (ECoG)**: https://openneuro.org/datasets/ds005574

## Model Download

Models are available on HuggingFace. For network issues, ModelScope is an alternative.

## Quick Start

```python
from src.brain_score import BrainScoreCalculator
from src.rsa import RSACalculator
from src.utils import load_eeg_data, load_llm_representations

# Load data
eeg_data = load_eeg_data("data/eeg/article_0_average.fif")
llm_features = load_llm_representations("data/representations/Qwen2.5-0.5B/article_0/layer_12_original.joblib")

# Compute Brain-Score
bs_calc = BrainScoreCalculator(pca_components=500, cv_folds=10)
bs_results = bs_calc.compute_all_electrodes(llm_features, eeg_data)
print(f"Brain-Score: {bs_results['mean_brain_score']:.4f}")

# Compute RSA
rsa_calc = RSACalculator()
rsa_results = rsa_calc.compute(llm_features, eeg_data)
print(f"RSA: {rsa_results['rsa_score']:.4f}")
```

## Command Line

```bash
python scripts/run_analysis.py --model Qwen2.5-0.5B --article 0
```

## Methodology

### Brain-Score (Formulas 1-3)

- PCA: k = min(500, n, d)
- Ridge regression: α ∈ [10^-3, 10^3], nested CV
- 10-fold CV, Pearson correlation, averaged across 32 channels

### RSA (Formulas 4-6)

- LLM RSM: Cosine similarity
- EEG RSM: Pearson correlation (channel-concatenated)
- Second-order: Spearman rank correlation

## Models

47 LLMs evaluated (0.5B-14B parameters):
- Qwen2.5 (10), Qwen3 (10), DeepSeek-R1 (5)
- Llama (10), Mistral (2), Others (10)

## Project Structure

```
├── src/
│   ├── brain_score.py
│   ├── rsa.py
│   ├── extract_representations.py
│   └── utils.py
├── scripts/
│   └── run_analysis.py
└── configs/
    └── models.yaml
```

## License

MIT License
