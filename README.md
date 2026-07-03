# Production Pipeline: Fine-Tuning Qwen2.5 for Context-Aware Vedic Astrology Systems

An end-to-end Machine Learning Engineering pipeline for configuring, quantization-optimizing, and scaling domain-specific large language models. This repository contains the concrete architecture for fine-tuning Qwen2.5-1.5B-Instruct using Parameter-Efficient Fine-Tuning (PEFT/LoRA) alongside production blueprinting for high-throughput hosting via vLLM.

---

## 1. Project Directory Architecture
```text
vedic-astrology-llm-core/
│
├── data/
│   └── 5_handwritten_chats.json     # Custom target-schema multi-turn conversational data
│
├── notebooks/
│   └── finetuning.ipynb             # Quantized PEFT/LoRA fine-tuning execution pipeline
│
├── vllm_vps_hosting_guide.md        # Production-grade MLOps deployment documentation
└── README.md                        # Project executive overview and technical specs
