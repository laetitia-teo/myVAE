# Variational Auto-Encoder

This is an implementation of the Variational Auto-Encoder. It is trained using the MNIST hand-written digit dataset.

To use :

```python
from main import *
```

This will create an instance of the VAE object which represents the models (encoder and decoder) and will train it for 10 epochs, with minibatch size 64.

Alternatively, use :

```python
from model import VAE

vae = VAE()
vae.train()
```

To sample 64 random images of hand-written digits, and display them alongside their reconstructed version from the auto-encoder, use :

```python
vae.recode_vis()
```

To sample a noise vector from which to generate new digits, use :

```python
vae.gen_vis()
```
