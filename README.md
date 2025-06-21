# 🧠 Fine-Tuning Gemma3-1B with Unsloth for Sentiment Analysis

This repository demonstrates how to fine-tune the **Gemma3-1B Instruction-Tuned** small language model (SLM) using [Unsloth](https://unsloth.ai/) for **financial sentiment analysis**. It frames the task as **supervised fine-tuning (SFT)** using prompt-completion pairs in a conversational format, leveraging PEFT (LoRA) for efficient training on limited hardware. The project serves as a general template for supervised fine-tuning with Unsloth and can be easily adapted to other tasks and datasets beyond sentiment analysis.

---

## 📌 Key Features

- 🔧 Fine-tuning using **Unsloth** with minimal boilerplate.
- 📈 Evaluates model **before and after** fine-tuning.
- ⚙️ Includes **PEFT/LoRA configuration** for memory-efficient training.
- 🧪 Evaluation metrics: Accuracy, Classification Report, Confusion Matrix.
- 📊 Visualizes training and validation loss curves.
- ✅ Runs on consumer-grade GPUs (e.g., 12GB VRAM, RTX 3060).

---

## 📝 Task Overview

We fine-tune `unsloth/gemma-3-1b-it` for **financial news sentiment classification**. The model is prompted in natural language, returning one of:
- `positive`
- `neutral`
- `negative`

---
we use the FinancialPhraseBank dataset sourced from: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news

## 🛠️ Setup Instructions

**Create Environment:**
```bash
conda create -n unsloth_env python=3.10 -y
conda activate unsloth_env
pip install unsloth scikit-learn matplotlib pandas datasets trl
```
Run the full notebook or training/testing scipts as appropriate
Note: A separate colab notebook is maintained to quickly try this out without worrying about envionment setup

