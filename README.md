# ViCToR: Improving Visual Comprehension via Token Reconstruction for Pretraining LMMs



[![Arxiv](https://img.shields.io/badge/arXiv-2410.14332-red)](https://arxiv.org/abs/2410.14332) [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/collections/DeepGlint-AI/croc-670d1a34f0ad3dc15144585b)

<p align="center">
  <img src="https://github.com/user-attachments/assets/3ead6fce-2da1-4b9e-9bd9-f01447215f41" alt="842f5fe2-84ad-464a-83d6-408bf1a0d9fa.webp" width=70%>
</p>


Large Multimodal Models (LMMs) often face a modality representation gap during pretraining: while language embeddings remain stable, visual representations are highly sensitive to contextual noise (e.g., background clutter). To address this issue, we introduce a visual comprehension stage, which we call \textbf{ViCToR}~(\textbf{Vi}sual \textbf{C}omprehension via \textbf{To}ken \textbf{R}econstruction), a novel pretraining framework for LMMs. ViCToR employs a learnable visual token pool and utilizes the Hungarian matching algorithm to select semantically relevant tokens from this pool for visual token replacement. Furthermore, by integrating a visual token reconstruction loss with dense semantic supervision, ViCToR can learn tokens which retain high visual detail, thereby enhancing the large language model’s (LLM’s) understanding of visual information. After pretraining on 3 million publicly accessible images and captions, \textbf{ViCToR} achieves state-of-the-art results, improving over LLaVA-NeXT-8B by $10.4\%$, $3.2\%$, and $7.2\%$ on the MMStar, SEED$^{I}$, and RealWorldQA benchmarks, respectively. We will release the code and model weights to facilitate reproducibility.

## 📜 News
**[2024/10/21]** The [paper](https://arxiv.org/pdf/2410.14332) and [code](https://github.com/deepglint/Croc) are released!💥

## 👨‍💻 Todo
- [ ] Better model base on ViCToR
- [ ] Checkpoints of ViCToR-13B
- [x] Checkpoints of ViCToR-7B
- [x] Training code for ViCToR

## 🤖 Model Zoo

| Name | LLM | Checkpoint |  MMBench | MMBench-CN | SEED | MM-Vet | SQA-image | VQA-v2 | POPE | GQA | LLaVA-W |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ViCToR-7B | Vicuna-7B | [ViCToR-7B](https://huggingface.co/DeepGlint-AI/LLaVA-v1.5-Croc-7b) | 69.1 | 60.5 | 63.0 | 36.8 | 72.3 | 80.1 | 86.9 | 63.5 | 73.3 |



## Install

```bash
git clone https://github.com/deepglint/Victor.git
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

**Stage 1.5: Pretraining ViCToR**
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
      title={ViCToR: Improving Visual Comprehension via Token Reconstruction for Pretraining LMMs}, 
      author={Yin Xie and Kaicheng Yang and Peirou Liang and Xiang An and Yongle Zhao and Yumeng Wang and Ziyong Feng and Roy Miles and Ismail Elezi and Jiankang Deng},
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
