# Dual-Encoder Based Natural Image Search Engine (MS-COCO + Phone Images)

This repository contains the code and report assets for a **dual-encoder natural image search engine** built on top of **OpenCLIP ViT-B/32**.  
The system retrieves images from a large collection using **free-form captions** and is evaluated on:

1. **TestSet-1:** Images and captions from the **MS-COCO 2017** dataset  
2. **TestSet-2:** **Phone-captured images** taken by the author with manually written captions  

The goal is to understand **how well a pretrained dual-encoder model generalizes** from a curated benchmark dataset to real-world personal imagery and how performance changes as the search space scales from 5k to 118k+ images.

---

## 📘 Code & Resources

- **Colab Notebook (full implementation):**  
  👉 [Dual Encoder Image Search Engine Colab](https://colab.research.google.com/drive/12QB3Xjmi5JjgYaV6OGQ0QvG-Wv54VcE-?usp=sharing)

- **Dataset (MS-COCO 2017 on Kaggle):**  
  👉 [COCO 2017 Dataset on Kaggle](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)

The notebook includes:

- OpenCLIP ViT-B/32 setup  
- Embedding generation for images and captions  
- FAISS indexing (cosine similarity)  
- Retrieval evaluation (Recall@K, MedR, MRR)  
- Visualization of top-K retrieved images for different queries  

---

## 🧠 Task Overview

Assignment requirements:

1. **Implement** a dual-encoder based natural image search engine using **MS-COCO**.  
2. **Prepare two test sets:**
   - **TestSet-1:** Images and captions drawn from MS-COCO 2017  
   - **TestSet-2:** Images captured by mobile phone + manually written captions  
3. **Compare** retrieval performance between **TestSet-1** and **TestSet-2**.

This repository provides exactly that: a CLIP-style dual encoder, a FAISS index over COCO images, and evaluation of phone-captured images under the same pipeline.

---

## 📦 Datasets

### TestSet-1 — MS-COCO 2017

Two COCO subsets are used:

- **Validation split**
  - **5,000 images**
  - **25,014 captions**
  - Clean, well-aligned human-written descriptions
- **Training split**
  - ~**118,000 images**
  - ~**590,000 captions**
  - Much larger, more redundant, and more noisy

All COCO images are processed through the **OpenCLIP image encoder**, and all captions through the **OpenCLIP text encoder**, producing L2-normalized 512-D embeddings stored in a **FAISS index**.

Example entries (validation split):

| ID    | File              | Caption                                                         |
|-------|-------------------|-----------------------------------------------------------------|
| 179765| 000000179765.jpg  | A black Honda motorcycle parked in front of a house            |
| 190236| 000000190236.jpg  | An office cubicle with four different types of workspaces      |
| 331352| 000000331352.jpg  | A small closed toilet in a cramped space                       |
| 517069| 000000517069.jpg  | Two women waiting at a bench next to a street                  |
| 182417| 000000182417.jpg  | A beautiful dessert waiting to be shared by two people         |

---

### TestSet-2 — Phone-Captured Images

- Small, realistic set of **10 mobile-phone photos**
- Scenes include:
  - A man holding a cat in a pet store
  - A person in front of an IEEE conference banner
  - A close-up portrait with PUMA glasses
  - Other everyday photos (vehicles, rooms, food, etc.)
- **4 images** passed strict path/validation checks and are used for metrics.

Each phone image has a manually written, detailed caption, e.g.:

- `"a smiling man holding a white cat inside a pet store with shelves of products behind him"`
- `"a young man standing in front of an IEEE COMPAS 2025 conference banner wearing a backpack"`
- `"a close up of a man wearing black framed PUMA glasses smiling outdoors"`

The phone images are encoded with **the same OpenCLIP pipeline**, and their embeddings are inserted into the same FAISS index as COCO, enabling **joint retrieval**.

---

## 🏗️ Model & Retrieval Pipeline

### Dual-Encoder (OpenCLIP ViT-B/32)

- **Image encoder:** Vision Transformer (ViT-B/32)  
- **Text encoder:** Transformer-based text model paired with the ViT  
- Both map inputs into a **shared 512-D embedding space**

#### Embedding generation

- **Images**
  - Resize & center crop to CLIP resolution
  - Normalize using CLIP mean/std
  - Encode with image encoder → 512-D vector
- **Captions**
  - Tokenize using OpenCLIP tokenizer
  - Encode with text encoder → 512-D vector
- **Normalize**: L2 normalization for both image and text embeddings

### FAISS Indexing

- Index type: exact **cosine-similarity** search (via normalized dot product)
- For a query caption embedding \(x_t\) and image embedding \(x_i\):

\[
s(x_t, x_i) = \frac{x_t \cdot x_i}{\|x_t\|_2 \|x_i\|_2}
\]

- Retrieval: given a caption, compute its embedding and query FAISS for **top-K nearest images**.

---

## 🔍 Evaluation Metrics

For each caption query, we compute ranking metrics:

- **Recall@K (R@K)** — fraction of queries where the correct image is in top-K  
- **Median Rank (MedR)** — median position of the correct image  
- **Mean Reciprocal Rank (MRR)** — average of \(1 / \text{rank}\)

Metrics are reported separately for:

- **TestSet-1** (COCO validation, and COCO training split)
- **TestSet-2** (phone images)

---

## 📈 Results

### 1. COCO Validation vs Phone Images

Using the **COCO 2017 validation split** as the search index (5k images) + 4 phone images:

| Set   | R@1  | R@5  | R@10 | MedR | MRR   | N (queries) |
|-------|------|------|------|------|-------|-------------|
| COCO  | 0.388| 0.648| 0.747| 2    | 0.509 | 25,014      |
| Phone | 0.250| 0.500| 0.500| 11   | 0.388 | 4           |

**Observations**

- On **COCO** (clean, in-distribution), the system is strong:
  - Correct image usually appears in **top-2 results**
  - R@10 almost **75%**
- On **phone images**, performance drops:
  - R@1 only **25%**
  - Median rank jumps to **11**
  - Shows clear **domain shift** between COCO and real phone photos

---

### 2. COCO Training Split vs Phone Images (Large-Scale Retrieval)

Using the **COCO 2017 training split** as the search index (~118k images, 590k captions):

| Set   | R@1  | R@5  | R@10 | MedR | MRR   | N (queries) |
|-------|------|------|------|------|-------|-------------|
| COCO  | 0.130| 0.267| 0.342| 36   | 0.202 | 590,313     |
| Phone | 0.250| 0.250| 0.250| 281  | 0.273 | 4           |

**Observations**

- With **118k images**, COCO performance naturally drops:
  - R@1 ≈ **13%**
  - R@10 ≈ **34%**
  - MedR climbs to **36**
- The **larger search space + caption redundancy** make ranking harder:
  - Many visually similar images compete for top ranks
  - Captions re-use common patterns (“a man standing…”, “a dog on the grass…”)
- Phone images remain challenging: ranks fluctuate widely and MedR is very high (281).

---

## 🖼️ Qualitative Examples

The report includes several visualizations (montages of top-10 results):

- **MS-COCO queries**
  - `boat_tunnel.png` — Query:  
    *“A boat in the water next to a rail in a tunnel.”*
  - `giraffe_fence.png` — Query:  
    *“A giraffe standing in a field next to a fence.”*

- **Phone-caption queries searching over COCO**
  - `cat_man_akif_photo.png` — Query:  
    *“a smiling man holding a white cat inside a pet store with shelves of products behind him”*  
  - `spectacles.png` — Query:  
    *“a close up of a man wearing black framed PUMA glasses smiling outdoors”*
  - `young_man.png` — Query:  
    *“a young man standing in front of an IEEE COMPAS 2025 conference banner wearing a backpack”*

These examples show that the search engine captures **broad semantics** (man, cat, glasses, conference-like scenes), but often fails to retrieve the **exact** phone image when it is embedded among thousands of similar COCO images.

---

## 🔧 How to Use (High-Level)

1. **Download COCO 2017** (or mount the Kaggle dataset)  
2. **Run the Colab notebook** (or adapt it locally):
   - Install dependencies: `open_clip`, `faiss`, `torch`, `tqdm`, etc.
   - Generate and save **image embeddings** for the chosen split
   - Build a **FAISS index** over image embeddings
   - Encode your **query captions** and perform retrieval
3. **Optional:**  
   - Add your own **phone images** + captions  
   - Encode and insert them into the index  
   - Re-run evaluation metrics and visualizations

---

## 💭 Discussion & Takeaways

- **Dual-encoder models like OpenCLIP are powerful zero-shot search engines** on well-curated, in-domain data (COCO validation).
- **Retrieval quality degrades with:**
  - **Scale** (5k → 118k images)
  - **Caption redundancy/noise** (training split vs validation split)
  - **Domain shift** (COCO → personal phone images)
- Even when the model “understands” rough semantics, it can struggle with:
  - Fine context (conference banner text, store interior)
  - Personal identity cues (the *same* person appearing)
- For **real-world search applications**, these experiments highlight the need for:
  - **Domain-aligned fine-tuning** (e.g., contrastive training on user data)
  - Better handling of **near-duplicates** and **caption variability**

---

## 👤 Author

**Akif Islam**  
Department of Computer Science & Engineering  
University of Rajshahi, Bangladesh  
📧 `iamakifislam@gmail.com`

---

## 📚 Citation

If you use this repository, any code snippet, or take help from the explanations, **please cite**:

```bibtex
@misc{ameen2025detectingaigeneratedimagesdiffusion,
      title={Detecting AI-Generated Images via Diffusion Snap-Back Reconstruction: A Forensic Approach}, 
      author={Mohd Ruhul Ameen and Akif Islam},
      year={2025},
      eprint={2511.00352},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.00352}, 
}
