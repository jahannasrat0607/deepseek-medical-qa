# Fine-Tuning DeepSeek-R1-Distill-LLaMA-8B on Medical Reasoning Dataset

This project demonstrates how to fine-tune the [`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) model on the [`FreedomIntelligence/medical-o1-reasoning-SFT`](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) dataset using QLoRA and Hugging Face Trainer.

---

## Project Highlights

-  **Model**: DeepSeek-R1-Distill-LLaMA-8B
-  **Dataset**: Medical-O1 Reasoning SFT (Instruction-tuning for medical QA)
-  **Technique**: Parameter-efficient QLoRA fine-tuning
-  **Tracking**: Integrated with Weights & Biases (W&B)
-  **Use Case**: Enhancing instruction-following and reasoning in medical domain

---

## Folder Structure

```
.
├── medical-llama-qlora-finetune.ipynb   # Main training notebook
├── README.md                            # Project description and instructions
├── config/                              # Optional config files (if any)
└── outputs/                             # Model checkpoints, logs, etc.
```

---

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/jahannasrat0607/deepseek-medical-qa.git
cd deepseek-medical-qa
```

2. Install required packages

3. Set your secret tokens:
- In Kaggle, go to **Add-ons -> Secrets** and add your Hugging Face (`HF_TOKEN`) and Weights & Biases (`WANDB_API_KEY`).

---

## Training Details

- QLoRA configuration for memory-efficient fine-tuning.
- Batched dataset preprocessing with tokenization.
- Hugging Face Trainer integration for training, evaluation, and checkpointing.
- Logs and metrics tracked via W&B.

---

## Acknowledgements

- [DeepSeek-AI](https://huggingface.co/deepseek-ai)
- [FreedomIntelligence Dataset](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
- [Hugging Face](https://huggingface.co)
- [Weights & Biases](https://wandb.ai)

---

## License

This project is open-source and available under the [MIT License](LICENSE).
