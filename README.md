<p align="center" width="100%">
<img src="./assets/logo.png" alt="842f5fe2-84ad-464a-83d6-408bf1a0d9fa.webp" width=30%>
</p>

# Croc: Pretraining Large Multimodal Models with Cross-Modal Comprehension



[![Arxiv](https://img.shields.io/badge/arXiv-2410.14332-red)](https://arxiv.org/abs/2410.14332) [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/collections/DeepGlint-AI/croc-670d1a34f0ad3dc15144585b)

<p align="center">
  <img src="https://github.com/user-attachments/assets/3ead6fce-2da1-4b9e-9bd9-f01447215f41" alt="842f5fe2-84ad-464a-83d6-408bf1a0d9fa.webp" width=70%>
</p>


Recent advances in Large Language Models (LLMs) have catalyzed the development of Large Multimodal Models(LMMs). However, existing research primarily focuses on tuning language and image instructions, ignoring the critical pretraining phase where models learn to process textual and visual modalities jointly. In this paper, we propose a new pretraining paradigm for LMMs to enhance the visual comprehension capabilities of LLMs by introducing a novel cross-modal comprehension stage. Specifically, we design a dynamically learnable prompt token pool and employ the Hungarian algorithm to replace part of the original visual tokens with the most relevant prompt tokens. Then, we conceptualize visual tokens as analogous to a ''foreign language'' for the LLMs and propose a mixed attention mechanism with bidirectional visual attention and unidirectional textual attention to comprehensively enhance the understanding of visual tokens. Meanwhile, we integrate a detailed caption generation task, leveraging rich descriptions to further facilitate LLMs in understanding visual semantic information. After pretraining on 1.5 million publicly accessible data, we present a new foundation model called **Croc**. Experimental results demonstrate that Croc achieves new state-of-the-art performance on massive vision-language benchmarks. To support reproducibility and facilitate further research, we will release the training code and pre-trained model weights.

## 📜 News
**[2024/10/21]** The [paper](https://arxiv.org/pdf/2410.14332) and [code](https://github.com/deepglint/Croc) are released!💥

## 👨‍💻 Todo
- [ ] Better model base on Croc
- [ ] Checkpoints of Croc-13B
- [x] Checkpoints of Croc-7B
- [x] Training code for Croc

## 🤖 Model Zoo

| Name | LLM | Checkpoint |  MMBench | MMBench-CN | SEED | MM-Vet | SQA-image | VQA-v2 | POPE | GQA | LLaVA-W |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Croc-7B | Vicuna-7B | [Croc-7B](https://huggingface.co/DeepGlint-AI/LLaVA-v1.5-Croc-7b) | 69.1 | 60.5 | 63.0 | 36.8 | 72.3 | 80.1 | 86.9 | 63.5 | 73.3 |



## Install

```bash
git clone https://github.com/deepglint/Croc.git
cd Croc
conda create -n croc python=3.10 -y
conda activate croc

pip install --upgrade pip
pip install -e .
pip install flash-attn --no-build-isolation
```

## Training

**Stage 1: Pretraining MLP**
```bash
bash scripts/pretrain_mlp.sh
```

**Stage 1.5: Pretraining Croc**
```bash
bash scripts/pretrain_croc.sh
```

**Stage 2: Instructional Finetuning**
```bash
bash scripts/finetune.sh
```

---


## Citation

```latex
@misc{xie2024crocpretraininglargemultimodal,
      title={Croc: Pretraining Large Multimodal Models with Cross-Modal Comprehension}, 
      author={Yin Xie and Kaicheng Yang and Ninghua Yang and Weimo Deng and Xiangzi Dai and Tiancheng Gu and Yumeng Wang and Xiang An and Yongle Zhao and Ziyong Feng and Jiankang Deng},
      year={2024},
      eprint={2410.14332},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.14332}, 
}
```

## Acknowledgement  

We extend our deepest gratitude to the creators and contributors of the following projects:  
1. [LLaVA](https://github.com/haotian-liu/LLaVA): The comprehensive codebase for training Vision-Language Models (VLMs).  

Their exceptional work has been instrumental to our research and development efforts.
