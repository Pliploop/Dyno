<div align="center">

# GD-Retriever: Controllable Generative Text-Music Retrieval with Diffusion Models

**Julien Guinot<sup>*,1,2</sup>, Elio Quinton<sup>2</sup>, György Fazekas<sup>1</sup>**  
<sup>1</sup> Centre for Digital Music, Queen Mary University of London, U.K.  
<sup>2</sup> Music & Audio Machine Learning Lab, Universal Music Group, London, U.K.  
<sup>*</sup>Correspondence to j.guinot@qmul.ac.uk

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-brightgreen.svg)](https://arxiv.org/abs/xxxx.xxxxx)

<p align="center">
  <img src="readme_/figs_/overview.png" alt="GD-Retriever Overview" width="500"/>
</p>

</div>

---

## Abstract

Multimodal contrastive models have achieved strong performance in text-audio retrieval and zero-shot settings, but improving joint embedding spaces remains an active research area. Less attention has been given to making these systems controllable and interactive for users. In text-music retrieval, the ambiguity of freeform language creates a many-to-many mapping, often resulting in inflexible or unsatisfying results.

We introduce **Generative Diffusion Retriever (GDR)**, a novel framework that leverages diffusion models to generate queries in a retrieval-optimized latent space. This enables controllability through generative tools such as negative prompting and denoising diffusion implicit models (DDIM) inversion, opening a new direction in retrieval control. GDR improves retrieval performance over contrastive teacher models and supports retrieval in audio-only latent spaces using non-jointly trained encoders. Finally, we demonstrate that GDR enables effective post-hoc manipulation of retrieval behavior, enhancing interactive control for text-music retrieval tasks.

---

## Highlights
- **Generative Retrieval**: Novel diffusion-based approach for text-music retrieval that generates queries in audio latent space
- **Controllable**: Enables negative prompting and DDIM inversion for interactive retrieval refinement
- **Flexible Encoders**: Works with non-jointly trained audio and text encoders
- **Improved Performance**: Outperforms contrastive teacher models on in-domain datasets
- **Interactive**: Enables post-hoc manipulation of retrieval behavior for better user control

---

## Overview

Instead of encoding text queries and audio keys in a joint embedding space (top), GD-Retriever generates queries in the audio space directly through conditioning on a text query (bottom). This approach enables generative controllability mechanisms that are not available in traditional contrastive retrieval systems.

<p align="center">
  <img src="readme_/figs_/GDRetrieval.png" alt="GD-Retriever Architecture" width="400"/>
  <img src="readme_/figs_/GDTraining.png" alt="GD-Retriever Training" width="400"/>
</p>

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/Diff-GAR.git
cd Diff-GAR
```

2. Create a conda environment:
```bash
conda create -n diffgar python=3.10
conda activate diffgar
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## Configuration

The codebase is designed to be modular and flexible. Most configuration is handled through YAML files in the `config/` directory. You only need to modify paths if you're using the same datasets as in the paper or if you want to use your own datasets.

### Dataset Paths (Optional)

If you're using the datasets from the paper (SongDescriber, MusicCaps, PrivateCaps), you'll need to update the placeholder paths in the configuration files. The main files to modify are:

- **Evaluation utilities**: `diffgar/evaluation/utils.py`
- **Data loaders**: `diffgar/dataloading/dataloaders.py`
- **Configuration files**: `config/train_ldm/data/local/`
- **Extract features config**: `config/extract_features/`

### Custom Datasets

For custom datasets, you can:
1. Create your own data loading scripts following the existing patterns
2. Modify the configuration files to point to your data
3. Use the existing evaluation framework with your own data

The codebase supports various audio-text encoders (CLAP, MULE, MusCALL) and can be easily extended to work with new datasets and encoders.

---

## Usage

### Training

To train a GD-Retriever model:

```bash
python train_ldm.py --config config/train_ldm/model/encoder_pair/clap/unet/train_ldm_sample_pred_base.yaml
```

### Evaluation

To evaluate retrieval performance:

```bash
python eval_retrieval.py --task song_describer --model_name your_model_name --model_step 50000
```

To evaluate fidelity and diversity:

```bash
python eval_fidelity.py --task song_describer --model_name your_model_name --model_step 50000
```

### Feature Extraction

To extract features using pre-trained encoders:

```bash
python extract_dataset.py --config config/extract_features/clap/extract_songdescriber.yaml
```

---

## Model Architecture

GD-Retriever consists of:

1. **Audio Encoder**: CLAP, MULE, or MusCALL for audio feature extraction
2. **Text Encoder**: T5 or CLAP text encoder for text feature extraction  
3. **Diffusion Model**: UNet or MLP-based diffusion model for generating audio embeddings
4. **Retrieval Head**: Cross-modal similarity computation

The model is trained on latent sequences of length T=64 (1 minute of audio) with a batch size of 256 for 100k steps using AdamW optimizer with linear warmup and cosine decay.

---

## Datasets

The model supports the following datasets:

| Dataset | #tracks | #captions | Hours | Training | Eval |
|---------|---------|-----------|-------|----------|------|
| SongDescriber | 0.7k | 1.1k | 23.3 | ❌ | ✅ |
| MusicCaps | 5.5k | 5.5k | 15.3 | ❌ | ✅ |
| PrivateCaps | 251k | 251k | 12.5k | ✅ | ✅ |

---

## Results

### Main Retrieval Results

GD-Retriever outperforms contrastive teacher models on in-domain datasets while maintaining competitive performance on out-of-domain evaluation sets.

#### Text-to-Audio Retrieval Performance

| Model | Metric | PC | SD | MC |
|-------|--------|----|----|----|
| **CLAP** | R@1 ↑ | 2.2 | 3.1 | **3.8** |
| | R@5 ↑ | 7.2 | 13.7 | **12.9** |
| | R@10 ↑ | 12.3 | 23.2 | **19.5** |
| | MedR (%) ↓ | 3.7 | 4.0 | **1.4** |
| **GDR-CLAP** | R@1 ↑ | **6.9** | **4.7** | 2.7 |
| | R@5 ↑ | **17.1** | **15.3** | 7.6 |
| | R@10 ↑ | **22.9** | **24.7** | 11.5 |
| | MedR (%) ↓ | **1.6** | **3.8** | 2.9 |
| **MusCALL** | R@1 ↑ | 10.1 | 3.6 | 1.0 |
| | R@5 ↑ | **26.2** | 13.6 | 3.9 |
| | R@10 ↑ | **35.1** | 22.0 | 7.0 |
| | MedR (%) ↓ | **0.4** | 4.2 | 5.1 |
| **GDR-MusCALL** | R@1 ↑ | **10.8** | **5.1** | **1.8** |
| | R@5 ↑ | 25.1 | **16.9** | **6.4** |
| | R@10 ↑ | 33.3 | **25.5** | **9.9** |
| | MedR (%) ↓ | 0.6 | **3.5** | **3.4** |

*PC: PrivateCaps, SD: SongDescriber, MC: MusicCaps*

### Controllability Results

#### Negative Prompting

GD-Retriever enables effective negative prompting for retrieval refinement, maintaining semantic similarity to the original query while distancing from negative attributes.

<p align="center">
  <img src="readme_/figs_/negativeprompting.png" alt="Negative Prompting Results" width="600"/>
</p>

#### DDIM Inversion

DDIM inversion allows fine-grained control over retrieval queries, enabling users to refine specific attributes while preserving semantic similarity to the original prompt.

<p align="center">
  <img src="readme_/figs_/ddim_inversion.png" alt="DDIM Inversion Example" width="600"/>
</p>

### Encoder Pair Variation

GD-Retriever can work with non-jointly trained encoders, demonstrating flexibility in encoder selection:

| Model | Text Encoder | R@1 | R@5 | R@10 | MR |
|-------|-------------|-----|-----|------|----|
| **GDR-CLAP** | T5 | **8.1** | **21.1** | **29.2** | **0.8** |
| | CLAP | 6.9 | 17.1 | 22.9 | 1.6 |
| **GDR-MULE** | T5 | 7.6 | 18.5 | 25.3 | 1.6 |

*Results on PrivateCaps dataset*

---

## Code Organization

This repository is organized as follows:

- `config/`: Configuration files for training, evaluation, and feature extraction
- `diffgar/`: Main package containing:
  - `dataloading/`: Data loading utilities and datasets
  - `evaluation/`: Evaluation scripts and metrics
  - `models/`: Model implementations (CLAP, MULE, MusCALL, diffusion models)
- `train_ldm.py`: Main training script
- `eval_retrieval.py`: Retrieval evaluation script
- `eval_fidelity.py`: Fidelity and diversity evaluation script
- `extract_dataset.py`: Feature extraction script

---

## Citation

If you use this code in your research, please cite our paper:


*Citation key coming soon - paper not yet published*

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

We thank the authors of CLAP and MusCALL for providing the pre-trained audio-text encoders used in this work. MULE weights and code were used from this repo : [MuLOOC](https://github.com/Pliploop/MuLOOC), from the paper [Leave-One-EquiVariant: Alleviating invariance-related information loss in contrastive music representations](https://arxiv.org/abs/2412.18955)

---

## Contact

For questions and issues, please open an issue on GitHub or contact [j.guinot@qmul.ac.uk](mailto:j.guinot@qmul.ac.uk).