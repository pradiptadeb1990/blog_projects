# Zero2Former Examples

This directory contains example applications using the Zero2Former transformer implementation.

## Grapheme to Phoneme (G2P) Conversion

The `g2p_conversion.py` script demonstrates how to use the Zero2Former transformer model for converting written text (graphemes) to their phonetic representations (phonemes).

### Dataset

The example uses the CMU Pronouncing Dictionary (CMUdict) which provides mappings from words to their phonetic transcriptions. The dataset is automatically downloaded using NLTK.

### Usage

1. Install required dependencies:
```bash
pip install nltk torch
```

2. Run the example:
```bash
python g2p_conversion.py
```

The script will:
- Download and prepare the CMUdict dataset
- Train a transformer model on the G2P task
- Save the trained model as 'g2p_model.pt'
- Test predictions on some sample words

### Model Architecture

The example uses a transformer with:
- 3 encoder layers
- 3 decoder layers
- 8 attention heads
- 256 model dimensions
- 1024 feedforward dimensions
- 0.1 dropout rate

### Example Output

After training, the model will make predictions like:
```
hello -> HH AH L OW
world -> W ER L D
transformer -> T R AE N S F AO R M ER
```
