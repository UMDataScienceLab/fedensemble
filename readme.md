# FedEnsemble

This repository contains some implementations of the paper [*Fed-ensemble: Ensemble Models in Federated Learning for Improved Generalization and Uncertainty Quantification*](https://ieeexplore.ieee.org/document/10113748)

Fed-ensemble trains several models instead of one and predicts by averaging them, improving generalization and uncertainty estimates.


## Getting Started

- Change the parameters in settings/cifar10.txt
- Run python main.py
- Results are generated in logs/ folder.


## Files

The implementation includes fedavg, fedensemble. They are implemented in server.py, federatedlearning.py, and client.py

## Contributing

Contributions are welcome! If you're interested in improving the multi-processing, or adding new algorithms, feel free to add a new pull request.

## Citation

[Paper](https://doi.org/10.1109/TASE.2023.3269639) (IEEE Transactions on Automation Science and Engineering, 2024). If you use this code, please cite:

```bibtex
@article{shi2024fedensemble,
  title={Fed-ensemble: Ensemble Models in Federated Learning for Improved Generalization and Uncertainty Quantification},
  author={Shi, Naichen and Lai, Fan and Al Kontar, Raed and Chowdhury, Mosharaf},
  journal={IEEE Transactions on Automation Science and Engineering},
  volume={21},
  number={3},
  pages={2792--2803},
  year={2024},
  doi={10.1109/TASE.2023.3269639}
}
```
