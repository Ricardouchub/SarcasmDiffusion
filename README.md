# SarcasmDiffusion

<img src="https://img.shields.io/badge/Proyecto_Completado-%E2%9C%94-2ECC71?style=flat-square&logo=checkmarx&logoColor=white" alt="Proyecto Completado"/> <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/> <img src="https://img.shields.io/badge/Diffusers-0.35.1-orange?style=flat-square&logo=huggingface&logoColor=white" alt="Diffusers"/> <img src="https://img.shields.io/badge/LoRA-FineTuning-blue?style=flat-square&logo=openaichat&logoColor=white" alt="LoRA"/> <img src="https://img.shields.io/badge/Stable_Diffusion_XL-Model-9b59b6?style=flat-square&logo=ai&logoColor=white" alt="Stable Diffusion XL"/> <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"/>

**Generador de Memes Sarcásticos entrenado con Stable Diffusion XL + LoRA.**  
Proyecto que combina *deep learning generativo*, *procesamiento de emociones* y *fine-tuning visual* para crear memes controlados por texto.

---

## Descripción del Proyecto
**SarcasmDiffusion** es un modelo basado en **Stable Diffusion XL** ajustado mediante **LoRA (Low-Rank Adaptation)** para aprender el estilo visual de los memes irónicos y sarcásticos, utilizando un dataset derivado del *Hateful Memes Dataset* (Facebook AI).  
El objetivo es generar imágenes limpias y expresivas sin texto incrustado, sobre las cuales se superpone luego el caption estilo meme.

---

## Arquitectura y Técnicas Utilizadas
| Componente | Descripción |
|-------------|-------------|
| **Modelo base** | Stable Diffusion XL (SDXL Base 1.0) |
| **Fine-tuning** | LoRA sobre el UNet (fp16) |
| **Framework** | Hugging Face Diffusers + PEFT + Accelerate |
| **Dataset** | 10K memes balanceados (`humor`, `irony`, `neutral`) |
| **App** | Streamlit UI para inferencia con overlay estilo meme |
| **Preprocesamiento** | NLP con GoEmotions + RoBERTa-Irony para etiquetado semántico |
| **Formato final** | Modelo fusionado (`sdxl_fused_full`) + LoRA separado (`lora_only`) |

---

## Estructura del repositorio

```
SarcasmDiffusion/
├── app.py                              # Interfaz Streamlit para inferencia
├── data/
│   ├── img/                            # Imágenes originales del dataset
│   ├── processed/
│   │   ├── metadata_v3.csv             # Dataset enriquecido y balanceado
│   │   ├── train_lora_prompts.csv      # Prompts generados por tono
│   │   └── infer_samples/              # Resultados de inferencia
├── models/
│   ├── sdxl_fused_full/                # Modelo completo fusionado
│   └── lora_only/                      # Pesos LoRA del UNet
└── SarcasmDiffusion.ipynb              # Notebook principal del proyecto

```

---

## Hiperparámetros del entrenamiento

| Parámetro | Valor |
|------------|-------|
| Resolución | 1024 px |
| Batch Size | 1 (Grad Accum = 4) |
| Learning Rate | 1e-4 |
| Max Steps | 6,000 |
| LoRA r / α / dropout | 8 / 16 / 0.05 |
| Optimizer | AdamW8bit |
| Scheduler | Cosine + Warmup |

> Entrenamiento realizado con *fp16 mixed precision* en CUDA 12.4 (PyTorch 2.6).

---

## Flujo de trabajo

1. **Fase A — Preprocesamiento**  
   Limpieza del dataset, enriquecimiento con emociones e ironía, y balanceo de tonos.

2. **Fase B — Generación de prompts**  
   Creación automática de descripciones de entrenamiento (`humor`, `irony`, `neutral`).

3. **Fase C — Entrenamiento SDXL con LoRA**  
   Fine-tuning del UNet del modelo con parámetros congelados para preservar calidad base.

4. **Fase D — Fusión y Exportación**  
   Generación de dos versiones: `lora_only` y `sdxl_fused_full`.

5. **Fase E — Inferencia y UI**  
   Implementación de una app en Streamlit (`app.py`) con control de seed, steps, guidance y captions tipo meme.

---

## Ejemplo de generación
<p align="center">
  <img width="600" src="img/sample_meme.png" alt="Ejemplo de meme generado"/>
</p>

---

## Ejemplos de prompts

| Prompt | Caption arriba | Caption abajo |
|--------|----------------|---------------|
| "sarcastic meme about running out of GPU VRAM at 3am" | WHEN YOUR GPU SAYS | 'OUT OF MEMORY' AT 3AM |
| "meme sarcástico sobre intentar comer saludable pero viendo pizza" | CUANDO PROMETES COMER SANO | PERO LA PIZZA TE HABLA 🍕 |
| "funny meme about checking the fridge for the third time" | WHEN YOU CHECK THE FRIDGE | FOR THE THIRD TIME 😅 |

---

## Resultados y logros

✅ Entrenamiento exitoso de un modelo de difusión capaz de aprender el **estilo visual** de memes sarcásticos.  
✅ Dataset enriquecido y balanceado automáticamente.  
✅ Generación controlada con *negative prompts* (sin texto ni ruido).  
✅ App de inferencia funcional en Streamlit.  
✅ Overlay automático estilo meme con ajuste dinámico del texto.

---

## Licencia
Este proyecto se distribuye bajo licencia **MIT**.  
Dataset original: *Hateful Memes (Facebook AI)* bajo términos de uso de FAIR.

---

<p align="center">
  <b>© 2025 - Ricardo Urdaneta</b><br>
  <i>“Aprender, crear y compartir conocimiento con propósito.”</i>
</p>
