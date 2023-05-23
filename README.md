## Restore Anything Pipeline: Segment Anything Meets Image Restoration

Jiaxi Jiang, Christian Holz

Sensing, Interaction & Perception Lab, ETH Zurich

>Recent image restoration methods have produced significant advancements using deep learning. However, existing methods tend to treat the whole image as a single entity, failing to account for the distinct objects in the image that exhibit individual texture properties. Existing methods also typically generate a single result, which may not suit the preferences of different users. In this paper, we introduce the Restore Anything Pipeline (RAP), a novel interactive and per-object level image restoration approach that incorporates a controllable model to generate different results that users may choose from. RAP incorporates image segmentation through the recent Segment Anything Model (SAM) into a controllable image restoration model to create a user-friendly pipeline for several image restoration tasks. We demonstrate the versatility of RAP by applying it to three common image restoration tasks: image deblurring, image denoising, and JPEG artifact removal. Our experiments show that RAP produces superior visual results compared to state-of-the-art methods. RAP represents a promising direction for image restoration, providing users with greater control, and enabling image restoration at an object level.

#### 1. Restore Anything Pipeline
![pipeline](https://github.com/eth-siplab/RAP/blob/main/figs/pipeline.png)
----------

#### 2. Flexible Blind Image Restoration
![interaction](https://github.com/eth-siplab/RAP/blob/main/figs/interaction.png)
----------

#### 3. Visual Comparisons
![comparison](https://github.com/eth-siplab/RAP/blob/main/figs/comparison.png)
----------

#### 4. More Examples
![examples](https://github.com/eth-siplab/RAP/blob/main/figs/examples.png)

We will release our codes in June.

#### BibTeX

```bibtex
@misc{jiang2023restore,
      title={Restore Anything Pipeline: Segment Anything Meets Image Restoration}, 
      author={Jiaxi Jiang and Christian Holz},
      year={2023},
      eprint={2305.13093},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
