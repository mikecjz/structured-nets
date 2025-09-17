# Structered Low-Displacement Rank Layer for Multi-coil MR Operator

> [!NOTE]
> This repo uses infrastructure provided by [Thomas et al.
> (2019)](https://arxiv.org/abs/1810.02309), and is forked from
> [HazyResearch/structured-nets](https://github.com/HazyResearch/structured-nets).

[[Project Page]](https://junzhou.chen.engineer/LDR/)

This repo contains the source code for "Direct Inversion Formula of the Multi-coil MR Operator under
Arbitrary Trajectories" @ MICCAI 2025. 

This repo  contains the simulation and trainning code used in the paper that demonstrate the
feasibility of direct inversion of the multi-coil MR operator based on the theory of low
displacement rank (LDR). [see Kailath,
1979](https://www.sciencedirect.com/science/article/pii/0022247X79901240).


## Installation

Use the provided `environment.yaml` to create a conda environment.
```bash
conda env create -f environment.yaml -n LDR
```

## Running the Experiments
> [!NOTE]
> Example date used in the paper is provided in the `data` directory.


First activate the conda environment.

```bash
conda activate LDR
```

Currently, the repo supports $2 \times$ and $4 \times$ Cartesian sampling and non-Cartsian radial
sampling. Simply run the provided `run_sl_2x.sh`, `run_sl_4x.sh` for Cartesian sampling and
`run_sl_toep.sh` for non-Cartsian radial sampling to train the LDR layer parameters. For example, to
train the inverse multi-coil MR operator with LDR layer parameters for $2 \times$ Cartesian
sampling, run:

```bash
bash run_sl_2x.sh
```


Results will be saved in the `results` directory unless otherwise specified. Order of the saved
images will be:
<table border="1" cellspacing="0" cellpadding="5" style="border-collapse: collapse;">
  <tr>
    <th>Layer Input</th>
    <th>Training Target</th>
    <th>LDR Layer Output</th>
    <th>Difference</th>
    <th>Difference × 10</th>
  </tr>
</table>

### Forward Multi-coil MR Operator
Per the theory of LDR, the forward multi-coil MR operator shares the same LDR structure as the
inverse multi-coil MR operator. Train with `--type forward` flag to train the forward multi-coil MR
operator with LDR layer parameters. For example:

```bash
bash run_sl_2x.sh --type forward
```

### Generate GIFs
To generate GIFs of the training process, run:
```bash
./generate_gifs.sh --fps 50 <png_results_dir>
```

## Implementation Details

The LDR layer for the multi-coil MR operator is implemented as class  `ToeplitzLikeSymmetric` in
[pytorch/structure/toeplitz.py](pytorch/structure/layer.py#L210-L221). 

Under the hood, the Triangular Toeplitz operators are implemented using FFT with
[toeplitz_transpose_multiply_fft](pytorch/structure/toeplitz.py#L132) and
[toeplitz_multiply_fft](pytorch/structure/toeplitz.py#L185).

The specific ataset for loading MRI data is `get_MRI_dataset` in
[pytorch/dataset.py](pytorch/dataset.py#L80-L126).

## Bibtex

```bibtex
@InProceedings{CheJun_Direct_MICCAI2025,
          author    = {Chen, Junzhou and Christodoulou, Anthony G. and Fan, Zhaoyang},
          title     = {{Direct Inversion Formula of the Multi-coil MR Operator under Arbitrary Trajectories}},
          booktitle = {Proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
          year      = {2025},
          publisher = {Springer Nature Switzerland},
          volume    = {LNCS 15963},
          month     = {September}
        }
        

```

