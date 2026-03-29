# PLDet: Intra-layer Multi-scale Perception and Local Space Attention for Pulmonary Lesion Detection in CT Images

This paper has been accepted at Biomedical Signal Processing and Control (BSPC).  

## BibTex
```bibtex
@InProceedings{Chen_PLDet,
        author    = {Chen, Junren and Wang, Wei and Cheng, Junlong and Liang, Gang and Zhang, Lei and Chen, Liangyin},
        title     = {PLDet: Intra-layer Multi-scale Perception and Local Space Attention for Pulmonary Lesion Detection in CT Images},
        year      = {2026},
        # todo: add booktitle, publisher, volume, month, pages
}

## 📌 Abstract
Automated and precise detection of lesions in chest computed tomography (CT) images is essential for diagnosing pulmonary diseases.  However, existing multi-scale feature-based and visual attention-based methods struggle to achieve adequate multi-scale perception and local context focus in features, which hinders their ability to capture fine-grained representations, thereby limiting their effectiveness in addressing lesion scale variability and lesion instance locality, ultimately leading to suboptimal detection performance.  To this end, we propose a novel Pulmonary Lesion Detection (PLDet) approach.  Specifically, we propose an Intra-layer Multi-scale Perception (IMP) module that employs a novel multi-branch feature extraction strategy, which is orthogonal to the methods that utilize layer-wise operations to capture multi-scale information, enabling the extraction of more fine-grained information from different receptive fields within a single network layer.  Additionally, we propose a Local Spatial Attention (LSA) network that captures local context information from both row and column spaces of the feature maps using a few parameters, while circumventing the traditional attention mechanism's dependence on global computing and channel dimensionality reduction, thereby preserving the integrity of the original context.  PLDet outperforms advanced methods in pulmonary lesion detection tasks on chest CT images. Extensive experimental results suggest that our PLDet approach holds promise as an initial reading and/or screening tool in chest CT images to combat pulmonary diseases.

## 🔍 Methodology
![Overview of PLDet](./assets/Figure1_PLDet.png "")
The overview of our PLDet. The proposed PLDet adopts YOLO architecture, including three functional networks: the backbone, neck, and head networks.  Both the IMP module and LSA network are integrated into the backbone and neck networks.