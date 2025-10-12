# SarcasmDiffusion

SarcasmDiffusion is a portfolio project that showcases a custom text-to-image diffusion pipeline fine-tuned for sarcastic meme generation. It includes a Streamlit application for interactive inference, meme-style text overlays, and the assets needed to reproduce the dataset preprocessing and model export pipeline.

## Features
- Streamlit UI with explicit model loading flow, status messages, and generation spinner feedback.
- Fused SDXL text-to-image pipeline with CUDA/CPU auto selection and safe defaults for sarcastic meme prompts.
- Meme overlay utility that adds Impact-style top and bottom captions with automatic wrapping and stroke styling.
- Dataset preprocessing notebook (Phase A) plus exported LoRA and fused weights under `models/` for reproducibility.

## Repository Structure
```
SarcasmDiffusion/
|-- app.py                    # Streamlit inference interface and meme overlay helpers
|-- data/
|   |-- train.jsonl | dev.jsonl | test.jsonl
|   |-- processed/            # Artifacts created from the app (infer_app/)
|-- models/                   # Model outputs (LoRA and fused SDXL weights)
|-- Notebook - Fase A  Preprocesamiento del dataset.ipynb
|-- Project overview.docx
|-- LICENSE
`-- README.md
```

## Getting Started
1. Clone the repository and move into the project directory.
2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate     # Windows
   source .venv/bin/activate   # macOS/Linux
   ```
3. Install the runtime dependencies:
   ```bash
   pip install streamlit torch diffusers transformers accelerate safetensors pillow
   ```
4. Ensure the fused SDXL weights are available in `models/sdxl_fused_full/` (included for local runs). If you export new weights, update the folder path referenced in `app.py`.
5. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Using the App
- Press **Cargar modelo** to load the diffusion pipeline into memory. The status panel confirms success or surfaces any errors.
- Configure prompts, meme captions, and sampling parameters from the sidebar. Advanced negative prompts live inside an expander to keep the UI clean.
- Click **Generar** to trigger inference. A spinner labeled "Generando meme con SDXL..." stays visible until the image finishes rendering.
- Review the generated meme in the main panel, download it as PNG, or inspect the persisted copy under `data/processed/infer_app/`.

## Dataset and Training Artifacts
- The notebook `Notebook - Fase A  Preprocesamiento del dataset.ipynb` documents the data cleaning and augmentation workflow used to prepare sarcastic meme captions.
- The `models/` directory stores both the intermediate LoRA weights (`lora_only/`) and the merged SDXL pipeline (`sdxl_fused_full/`). These checkpoints are referenced by the app and can be swapped for future experiments.

## Roadmap Ideas
- Add prompt presets and gallery history within the Streamlit session state.
- Package the inference pipeline as a reusable Python module with CLI entry points.
- Publish a Hugging Face Space or Docker image for easier sharing.

## License
This project is released under the MIT License. See `LICENSE` for the full terms.
