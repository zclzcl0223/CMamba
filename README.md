# C-Mamba: Channel Correlation Enhanced State Space Models for Multivariate Time Series Forecasting

In this repository, we present the code of ["C-Mamba: Channel Correlation Enhanced State Space Models for Multivariate Time Series Forecasting"](https://arxiv.org/abs/2406.05316).

![CMamba](./fig/mainfig.png)

## Data

All the datasets are available at [Autoformer: Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). You only need to download `electricity`, `ETT-small`, `traffic`, and `weather`.

## Environment

We implement our code in `Python 3.8.13` and `CUDA 11.7`. See [requirments.txt](./requirements.txt) for other packages. For convenience, you can install using the following commands:
```
pip install https://download.pytorch.org/whl/cu117/torch-1.13.1%2Bcu117-cp38-cp38-linux_x86_64.whl

pip install https://download.pytorch.org/whl/cu117/torchvision-0.14.1%2Bcu117-cp38-cp38-linux_x86_64.whl

pip install -r requirements.txt
```

## Reproducibility

All the training scripts are provided in [scripts/long_term_forecast](./scripts/long_term_forecast). For instance, if you want to get the results for the `weather` dataset, you just need to run:
```
bash ./scripts/long_term_forecast/Weather_script/CMamba.sh
``` 
The default `seq_len` in this repository is 96. For other experimental settings, the hyperparameters that you can tune are:
```
--seq_len
--pred_len
--batch_size
--learning_rate
--e_layers        # the number of CMamba block
--dropout         # dropout in CMamba block
--head_dropout    # dropout before the final linear projection layer
--d_model         # dimension of patch embedding
--d_ff            # dimension of linear projection in Mamba
--sigma           # standard derivation for channel mixup
--reduction       # reduction rate for channel attention
```
It is recommended to tune `e_layers` in {2, 3, 4, 5}, `d_model=d_ff` in {128, 256}, `sigma` in {0.5, 1.0,..., 5.0}, and `reduction` in {2, 4, 8}.

We also provide the experimental scripts for Table 2, where we combine our proposed channel mixup and channel attention modules with state-of-the-art models. You can run the following command to reproduce the results:

```
bash ./scripts/long_term_forecast/ECL_script/PatchTST.sh
```

## Results

- Checkpoints for each model will be saved in `checkpoints/`;
- Training log will be saved in `log/`;
- Prediction for the testing set will be saved in `results/`;
- Visualization for the results of testing set will be saved in `test_results/`.

## Acknowledgement

We are grateful for the following github repositories that provide valuable datasets and code base:

https://github.com/thuml/Autoformer

https://github.com/zshicode/MambaStock

https://github.com/moskomule/senet.pytorch

https://github.com/thuml/Time-Series-Library

https://github.com/yuqinie98/PatchTST

https://github.com/kwuking/TimeMixer

https://github.com/luodhhh/ModernTCN

https://github.com/ts-kim/RevIN

## Citation

If you find this repo useful in your research, please consider citing our paper as follows:

```
@article{zeng2024c,
  title={C-Mamba: Channel Correlation Enhanced State Space Models for Multivariate Time Series Forecasting},
  author={Zeng, Chaolv and Liu, Zhanyu and Zheng, Guanjie and Kong, Linghe},
  journal={arXiv preprint arXiv:2406.05316},
  year={2024}
}
```