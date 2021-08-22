# Option-Pricing-Neural-Networks (OPNN)

Pricing options using deep neural networks: a comparative study of supervised and unsupervised learning. Unsupervised learning is based on optimally weighted loss functions for solving PDEs
with Neural Networks [1].

## Running

This was tested using python 3.7. First install dependencies with:

```bash
pip install -r requirements.txt
```

To train and plot the results of a model use:

```bash
python main.py -t -g MODEL_NAME
```

e.g.

```bash
python main.py -t -g BSSt
```

To see more options run:

```bash
python main.py -h
```

## Reference
[1] van der Meer, R., Oosterlee, C., & Borovykh, A. (2020). Optimally weighted loss functions for 
solving pdes with neural networks. _arXiv preprint arXiv:2002.06269_.
[[arXiv]](https://arxiv.org/pdf/2002.06269.pdf)
