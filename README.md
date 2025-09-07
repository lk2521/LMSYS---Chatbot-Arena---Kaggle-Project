# LMSYS - Preference Prediction in Chatbot Arena Using Fine-Tuned LLMs  

## 📌 Overview  
This repository contains the solution to the **LMSYS – Chatbot Arena Human Preference Predictions** Kaggle competition. The task is to predict which of two chatbot responses users would prefer—or whether the result is a tie.  

Our approach leverages:  
- **Synthetic dataset generation** with LLaMA 3.1 405B.  
- **Parameter-Efficient Fine-Tuning (PEFT)** on Phi-3 Mini 4K and LLaMA 3.1 8B.  
- **Optimized inference** with KV caching, batching, dual-GPU, and multi-threading.  

We achieved a **log loss reduction from 11.892 → 0.949**, ranking competitively while remaining computationally efficient.  

---

## 🚀 Key Features  
- 10-criteria **discriminator prompt** for structured evaluation.  
- **20k+ synthetic samples** for training when labeled data was scarce.  
- Lightweight **Phi-3 Mini 4K** fine-tuned for efficiency.  
- **Dual-GPU inference pipeline** with batching & KV caching for speed.  
- Submission-ready probability predictions (`winner_model_a`, `winner_model_b`, `winner_tie`).  

---

## ⚙️ Methodology  

1. **Environment Setup**  
   - Dependencies installed from Kaggle offline wheels (`vLLM`, `PyTorch`, `LangChain`).  
   - CUDA-enabled GPUs with mixed precision.  

2. **Evaluation Framework**  
   - Responses compared across **10 criteria** (Relevance, Accuracy, Clarity, Conciseness, etc.).  
   - Model outputs structured preference judgments per criterion.  

3. **Synthetic Dataset**  
   - Generated 20k+ labeled pairs with **LLaMA 3.1 405B** to bootstrap training.  

4. **Fine-Tuning**  
   - Applied **PEFT** to Phi-3 Mini 4K and LLaMA 3.1 8B.  
   - Balanced efficiency and alignment with preference criteria.  

5. **Inference Optimization**  
   - KV caching (~40% faster).  
   - Batching (up to 50 samples).  
   - Dual-GPU parallelism & multi-threading for I/O.  

---

## 📊 Results  
- **Baseline log loss**: 11.892  
- **Final log loss**: 0.949  
- **Phi-3 Mini 4K**: Efficient and competitive despite small size.  
- **LLaMA 3.1 8B**: Higher capacity with improved alignment.  
