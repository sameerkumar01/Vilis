# ViLiS — Visual Linguistic Sync

A Hindi lip-reading system that converts silent video of a speaker's face into Hindi text. ViLiS combines a Conformer-based visual encoder with CTC decoding and an optional Whisper-generated pseudo-label pipeline for training on unlabelled video.

## Quick Start

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the package in editable mode
pip install -e .
```

## Pipeline Steps

### 1. Data Preparation
Extract mouth ROIs and generate Whisper pseudo-labels:

```bash
python -m pipelines.prepare_data --video-dir data/raw --out-dir data/processed
```

### 2. Train Tokenizer
Build a SentencePiece BPE tokenizer on the Hindi transcripts:

```bash
python -m pipelines.train_tokenizer --corpus data/processed/transcripts.txt
```

### 3. Training
Train the Conformer CTC model:

```bash
python -m training.train --config config/config.py
```

### 4. Export
Export the trained model to ONNX (and optionally CoreML on macOS):

```bash
python -m pipelines.export --checkpoint checkpoints/best.pt --format onnx
```

### 5. Inference / Demo
Run real-time inference from a webcam or video file:

```bash
python -m inference.demo --source webcam
python -m inference.demo --source path/to/video.mp4
```

## Project Structure

```
vilis/
├── config/          # Configuration & hyper-parameters
├── model/           # Conformer encoder, CTC head
├── training/        # Training loop, schedulers
├── inference/        # ONNX / CoreML runtime, demo
├── pipelines/       # Data prep, tokenizer, export
├── utils/           # Shared helpers
├── tests/           # Unit & integration tests
├── requirements.txt
├── setup.py
└── README.md
```

## Device Support

ViLiS auto-detects the best available device:

| Priority | Device | Batch Size | Whisper Model | AMP |
|----------|--------|-----------|---------------|-----|
| 1        | CUDA   | 8         | large-v3      | ✅  |
| 2        | MPS    | 2         | medium        | ❌  |
| 3        | CPU    | 2         | medium        | ❌  |

## License

MIT
