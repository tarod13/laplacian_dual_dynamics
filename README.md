# Proper Laplacian Representation Learning

---

This repository contains the code used to generate the different experiments and plots contained in [the paper](https://arxiv.org/pdf/2310.10833.pdf) of the same title.

To learn the Laplacian representation of an environment, run the following code:

```
python train_laprepr.py
```

This will train an encoder whose input is the state and the output is the corresponding entry of the smallest $$d$$ eigenvectors of the Laplacian. Once training is done, a plot of each of the eigenvectors is stored in the folder `results`. 

By default, ALLO is used to train the Laplacian encoder. To change hyperparameters, including the optimization objective, you can either add arguments when running `train_laprepr.py`, or store them in a `.yaml` file in the folder `src/hyperparam` and set the `config_file`:

```
python train_laprepr.py --config_file=you_hypers_file.yaml
```

The code requires Jax, Haiku and a few such dependencies.
