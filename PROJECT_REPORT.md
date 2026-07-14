## Experiments

Three loss functions were evaluated while keeping the architecture,
optimizer, preprocessing pipeline, and training settings unchanged.

| Experiment | Loss | Test mIoU |
|------------|------|----------:|
| Baseline | Sparse Categorical Crossentropy | 71.6% |
| Experiment 1 | Dice Loss | **73.9%** |
| Experiment 2 | Weighted Dice + Cross Entropy | 73.0% |

---

### Key Findings

- Dice Loss achieved the highest quantitative performance, improving Test Mean IoU from 71.6% to 73.9%.
- Dice Loss produced noticeably better segmentation of object boundaries.
- The weighted Dice + Cross Entropy loss generated visually balanced predictions but did not outperform Dice Loss in Mean IoU.
- Low input resolution (96×96) remained the primary limitation, particularly for fine structures and pets with backgrounds of similar color.

---

## Lessons Learned

During this project I learned:

- How skip connections preserve spatial information in encoder-decoder architectures.
- Why Mean IoU is a more informative metric than pixel accuracy for semantic segmentation.
- How different loss functions influence segmentation behavior beyond what numerical metrics alone capture.
- How to implement and serialize custom Keras loss functions.
- The importance of combining quantitative evaluation with qualitative inspection of predictions.