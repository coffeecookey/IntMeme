# IntMeme: Demystifying Hateful Content - Full Documentation

## Overview

IntMeme is a comprehensive framework for hateful meme detection using large multimodal models with explainable decisions. The project leverages state-of-the-art multimodal transformers to classify memes across multiple datasets, focusing on hate speech detection in visual-textual content.

## Project Structure

```
intmeme/
├── main.py                 # Entry point using Hydra and PyTorch Lightning
├── requirements.txt        # Python dependencies
├── README.md              # Basic project description
├── configs/               # Hydra configuration files
│   ├── datamodule/        # Data module configurations
│   ├── dataset/           # Dataset configurations
│   ├── experiment/        # Experiment configurations
│   ├── hydra/            # Hydra settings
│   ├── model/            # Model configurations
│   ├── trainer/          # Trainer configurations
│   └── metric/           # Metric configurations
├── datamodules/           # PyTorch Lightning data modules
│   ├── flava.py          # FLAVA data module
│   ├── intmeme.py        # IntMeme data module
│   ├── roberta.py        # RoBERTa data module
│   ├── utils.py          # Utility functions
│   └── collators/        # Data collation functions
├── datasets/              # Dataset classes
│   ├── base.py           # Base dataset classes
│   ├── datasets.py       # Dataset utilities
│   ├── fhm_finegrained.py # FHM Fine-grained dataset
│   ├── harmemes.py       # Harmemes dataset
│   ├── mami.py           # MAMI dataset
│   └── utils.py          # Dataset utilities
├── models/                # PyTorch Lightning model modules
│   ├── flava.py          # FLAVA classification model
│   ├── intmeme.py        # IntMeme multimodal model
│   └── roberta.py        # RoBERTa text-only model
├── preprocessing/         # Data preprocessing scripts
│   └── inpainting/       # Text inpainting from images
└── scripts/               # Training and testing scripts
    ├── fhm-finegrained/
    ├── harmeme/
    └── mami/
```

## Dependencies

Key dependencies include:
- PyTorch Lightning (lightning>=2.0.0)
- Transformers (transformers)
- Hydra (hydra-core, omegaconf)
- OpenAI CLIP
- Torchmetrics
- PIL, NumPy, Pandas
- OpenCV for image processing

## Supported Models

### 1. FLAVA Model (`models/flava.py`)

**Architecture:**
- Uses Facebook's FLAVA (FLAVA: A Foundational Language And Vision Alignment) model
- Processes both text and image inputs jointly
- Extracts multimodal embeddings from [CLS] token
- Classification head: Linear layer on multimodal embeddings

**Input Processing:**
- Text: Tokenized with FLAVA processor
- Images: Processed with FLAVA image processor (224x224 RGB)
- Batch collation via `collators/flava.py`

**Training:**
- Cross-entropy loss on classification tasks
- Supports multiple classification heads for different labels

### 2. RoBERTa Model (`models/roberta.py`)

**Architecture:**
- Text-only model using RoBERTa
- Extracts [CLS] token embeddings
- Classification head: Linear layer on text embeddings

**Input Processing:**
- Text only (no images)
- Tokenized with AutoTokenizer
- Standard text classification pipeline

### 3. IntMeme Model (`models/intmeme.py`)

**Architecture:**
- **Novel multimodal fusion approach**
- Combines two encoders:
  - Multimodal encoder (e.g., FLAVA) for meme content
  - Text encoder (e.g., RoBERTa) for auxiliary passages
- Concatenates [CLS] embeddings from both encoders
- Classification head: Linear layer on concatenated embeddings

**Key Innovation:**
- Uses auxiliary text passages alongside meme content
- Enables incorporation of external knowledge or interpretations
- Higher dimensional feature space for better representation

**Input Processing:**
- Meme inputs: Text + Image (processed with multimodal processor)
- Passage inputs: Additional text (processed with text tokenizer)
- Custom collation in `datamodules/intmeme.py`

## Data Pipeline

### Datasets

The framework supports multiple hateful meme datasets:

1. **FHM (Fine-grained Hateful Memes)**
   - Binary classification: hateful vs non-hateful
   - Includes interpretations and captions as auxiliary data

2. **Harmemes**
   - Intensity classification: not harmful, somewhat harmful, very harmful
   - Target classification: individual, organization, community, society

3. **MAMI (Multimodal Misogyny)**
   - Focus on misogynistic content detection

### Data Loading Process

1. **Dataset Classes** (`datasets/*.py`)
   - Load annotations from JSON/JSONL files
   - Process images: resize to 224x224, convert to RGB
   - Handle auxiliary data (interpretations, captions, etc.)
   - Apply text templates for input formatting

2. **Data Modules** (`datamodules/*.py`)
   - PyTorch Lightning data modules
   - Handle train/val/test splits
   - Configure processors and tokenizers
   - Set up data loaders with custom collation

3. **Collators** (`datamodules/collators/*.py`)
   - Batch processing functions
   - Tokenize texts, process images
   - Handle variable-length sequences with padding

### Data Flow

```
Raw Data → Dataset Class → DataLoader → Collate Function → Model
    ↓           ↓              ↓           ↓              ↓
Annotations  Preprocessing  Batching   Tokenization   Forward Pass
+ Images     + Templates    + Padding   + Images
+ Auxiliary  + Labels       + Labels    + Labels
```

## Training Pipeline

### Configuration System

Uses Hydra for configuration management:
- **Experiment configs**: Define complete experiments (model + dataset + training)
- **Model configs**: Model architecture and hyperparameters
- **Dataset configs**: Data paths, preprocessing settings
- **Trainer configs**: Training settings (GPU, batch size, etc.)

### Training Actions

1. **fit**: Train + validate + test
2. **test**: Load checkpoint and test
3. **predict**: Generate predictions for inference

### Metrics and Logging

- **Metrics**: Accuracy, AUROC (configurable)
- **Logging**: PyTorch Lightning logging
- **Checkpointing**: Save best models based on validation metrics

### Multi-GPU Training

Supports:
- Single GPU training
- Multi-GPU training (DDP)
- Debug training (CPU, fast iteration)

## Preprocessing

### Text Inpainting

Located in `preprocessing/inpainting/`:
- **Purpose**: Remove overlaid text from meme images
- **Methods**:
  - OpenCV inpainting
  - MMEdit (MediaMind Edit) library
- **Process**:
  1. OCR detection using Keras-OCR
  2. Create mask over detected text
  3. Inpaint masked regions

## Experiment Configurations

### Supported Experiments

Examples from `configs/experiment/`:

- **FHM Fine-grained**:
  - FLAVA with InstructBLIP interpretations
  - IntMeme with mPLUG interpretations
  - RoBERTa baselines

- **Harmemes**:
  - Intensity and target classification
  - Multiple model variants

- **MAMI**:
  - Misogyny detection tasks

### Configuration Structure

Each experiment config includes:
```yaml
defaults:
  - /model: [model_type]
  - /dataset: [dataset_config]
  - /datamodule: [datamodule_type]
  - /trainer: [trainer_config]
  - /metric: [metrics]

model:
  cls_dict: {label: num_classes}
  optimizers: [optimizer_configs]

dataset:
  dataset_class: datasets.[dataset].[class]
  text_template: "{text} [SEP] {interpretation}"
  labels: [label_list]
  auxiliary_dicts: {train/val/test: {key: path}}
```

## Running Experiments

### Training

```bash
# Using scripts
bash scripts/fhm-finegrained/train/intmeme.sh

# Direct command
python main.py +experiment=fhm/intmeme/mPLUG.yaml action=fit
```

### Testing

```bash
python main.py +experiment=fhm/intmeme/mPLUG.yaml action=test model_checkpoint=path/to/checkpoint.ckpt
```

### Prediction

```bash
python main.py +experiment=fhm/intmeme/mPLUG.yaml action=predict model_checkpoint=path/to/checkpoint.ckpt
```

## Key Innovations

1. **IntMeme Architecture**: Novel fusion of multimodal and text encoders for enhanced hateful content detection

2. **Explainable Decisions**: Incorporation of interpretations and auxiliary text for better model understanding

3. **Modular Design**: Clean separation of models, datasets, and training logic via Hydra configs

4. **Multi-Dataset Support**: Unified framework for different hateful meme datasets

5. **Preprocessing Pipeline**: Text inpainting for robust visual feature extraction

## Model Performance

The framework achieves state-of-the-art results on hateful meme detection benchmarks by:
- Leveraging large pre-trained multimodal models
- Incorporating external knowledge through auxiliary passages
- Using appropriate fusion techniques for text-image understanding
- Providing explainable predictions through interpretation integration

## Future Extensions

The modular architecture supports easy extension to:
- New multimodal architectures
- Additional datasets
- Different fusion mechanisms
- Advanced preprocessing techniques
- Interpretability methods</content>
<parameter name="filePath">/Users/tanishaojha/Desktop/intmeme/DOC.md