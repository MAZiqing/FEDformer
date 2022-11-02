# FEDformer (ICML 2022 paper)

Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, Rong Jin, "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting," in Proc. 39th International Conference on Machine Learning (ICML 2022), Baltimore, Maryland, July 17-23, 2022

Frequency Enhanced Decomposed
Transformer (FEDformer) is more efficient than
standard Transformer with a linear complexity
to the sequence length [[paper](https://arxiv.org/abs/2201.12740)]. 

Our empirical studies
with six benchmark datasets show that compared
with state-of-the-art methods, FEDformer can
reduce prediction error by 14.8% and 22.6%
for multivariate and univariate time series,
respectively.

## Frequency Enhanced Attention
|![Figure1](https://user-images.githubusercontent.com/44238026/171341166-5df0e915-d876-481b-9fbe-afdb2dc47507.png)|
|:--:| 
| *Figure 1. Overall structure of FEDformer* |

|![image](https://user-images.githubusercontent.com/44238026/171343471-7dd079f3-8e0e-442b-acc1-d406d4a3d86a.png) | ![image](https://user-images.githubusercontent.com/44238026/171343510-a203a1a1-db78-4084-8c36-62aa0c6c7ffe.png)
|:--:|:--:|
| *Figure 2. Frequency Enhanced Block (FEB)* | *Figure 3. Frequency Enhanced Attention (FEA)* |


## Main Results
![image](https://user-images.githubusercontent.com/44238026/171345192-e7440898-4019-4051-86e0-681d1a28d630.png)


## Get Started

1. Install Python>=3.8, PyTorch 1.9.0.
2. Download data. You can obtain all the six benchmarks from [[Autoformer](https://github.com/thuml/Autoformer)] or [[Informer](https://github.com/zhouhaoyi/Informer2020)].
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the multivariate and univariate experiment results by separately:

```bash
bash ./scripts/run_M.sh
bash ./scripts/run_S.sh
```


## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{zhou2022fedformer,
  title={{FEDformer}: Frequency enhanced decomposed transformer for long-term series forecasting},
  author={Zhou, Tian and Ma, Ziqing and Wen, Qingsong and Wang, Xue and Sun, Liang and Jin, Rong},
  booktitle={Proc. 39th International Conference on Machine Learning (ICML 2022)},
  location = {Baltimore, Maryland},
  pages={},
  year={2022}
}
```

## Further Reading
Survey on Transformers in Time Series:

Qingsong Wen, Tian Zhou, Chaoli Zhang, Weiqi Chen, Ziqing Ma, Junchi Yan, and Liang Sun. "Transformers in time series: A survey." arXiv preprint arXiv:2202.07125 (2022). [paper](https://arxiv.org/abs/2202.07125)


## Contact

If you have any question or want to use the code, please contact tian.zt@alibaba-inc.com or maziqing.mzq@alibaba-inc.com .

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/thuml/Autoformer

https://github.com/zhouhaoyi/Informer2020

https://github.com/zhouhaoyi/ETDataset

https://github.com/laiguokun/multivariate-time-series-data

