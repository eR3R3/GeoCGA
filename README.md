<p align="center">
  <h1 align="center">GeoCGA: Geometry-Aware Cross-Modal Graph Alignment for Referring Segmentation in 3D Gaussian Splatting</h1>
 <p align="center">
    <a href="[https://arxiv.org/abs/2511.03992](https://openaccess.thecvf.com/content/CVPR2026/papers/Tao_Geometry-Aware_Cross-Modal_Graph_Alignment_for_Referring_Segmentation_in_3D_Gaussian_CVPR_2026_paper.pdf)" target="_blank" rel="noopener noreferrer">
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&' alt='arXiv PDF'>
    </a>
  </p>
</p>

---


Referring 3D segmentation seeks to localize and segment target objects in a 3D scene given a natural-language query, requiring joint reasoning over geometric structures and linguistic cues. Although recent progress using 3D Gaussian Splatting (3DGS) has improved rendering quality, existing methods still struggle to spatially ground textual references due to two fundamental limitations: (1) language encoders provide no explicit positional priors, weakening geometric relation modeling; and (2) cross-modal attention is self-reinforcing, causing spatial errors to propagate through the Gaussian field once misalignment occurs. To address this, we propose GeoCGA, a geometry-aware cross-modal graph alignment framework that bridges linguistic semantics with the 3DGS representation. GeoCGA introduces position-aware prompt expansion to build a semantic-spatial graph capturing relational structure in text, and constructs a Gaussian-based geometric graph encoding 3D topology. A cross-modal alignment module enforces geometric consistency between the two graphs, enabling stable and spatially grounded correspondence across views. GeoCGA consistently outperforms prior state-of-the-art methods, yielding relative mIoU improvements of 20.8% on Ref-LERF, 5.7% on LERF-OVS, and 1.0% on 3D-OVS.  

<img width="1572" height="577" alt="image" src="https://github.com/user-attachments/assets/534020b5-c562-433d-a200-ccdb6500f226" />

---





## Configuration
Please refer to the [Dr-Splat Codebase](https://github.com/kaist-ami/Dr-Splat) for configuration  
We found Dr-Splat to perform well for preprocessing and clustering

## Datasets
Our primary dataset, and the primary dataset for 3D Referring Segmentation is [Ref-Lerf](https://huggingface.co/datasets/FudanCVL/Ref-Lerf)
from ICML 2025 Oral Paper [ReferSplat](arxiv.org/abs/2508.08252). See their [codebase](https://github.com/heshuting555/ReferSplat) for complete dataset configuration.
Note that for some scene in dataset, you have to run model [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) to
generate pesuedo masks.

## Checkpoints
We upload our model checkpoint on [huggingface]().

## Preprocessing/Training/Inference
Here are a list of things you have to do for setup
- clone the Dr-Splat [codebase](https://github.com/kaist-ami/Dr-Splat)
- clone the [Ref-Lerf](https://huggingface.co/datasets/FudanCVL/Ref-Lerf) dataset from huggingface
- Follow their instruction to do preprocess/training
  - Note that if you have any problem training the Dr-Splat model or understand the logic under the hood,
  - check out [文件路径.txt](https://github.com/eR3R3/GeoCGA/blob/main/%E6%96%87%E4%BB%B6%E8%B7%AF%E5%BE%84.txt) and [模型原理.txt](https://github.com/eR3R3/GeoCGA/blob/main/%E6%A8%A1%E5%9E%8B%E5%8E%9F%E7%90%86.txt), they are specific instructions I followed 
- Understand and run render_binary.py file in precompute mode to do object-level Gaussian Clustering
  - After this, you should have a lot of pth files, including activation features/mean feature/mean feature of all objects
- try render them using rendering mode
- Expand the data in Ref-Lerf using LLM and leverage it to process them in tuples to extract relationships between objects
  - input.json is an example of processed data for scene waldo_kitchen
- train the model using gnn.py, note that for geometric consistency loss mentioned in paper, you have to render the objects as well and calculate the loss on render 2D images
- for inference, use LLM expansion and extraction, run pre-trained gnn to get the object, pass that precomputed pth file of that object to render_binary.py and get the binary masks



