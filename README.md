<div align="center">
  <h1>dMel: Discretized Log Mel-Filterbanks</h1>

  [![Paper](https://img.shields.io/badge/Paper-Arxiv%20Link-green)](https://arxiv.org/pdf/2407.15835)
  [![Blog](https://img.shields.io/badge/Blog-Apple%20ML-blue)](https://apple.github.io/dmel-demo)
</div>

This software project accompanies the research paper, [dMel: Speech Tokenization Made Simple](https://arxiv.org/pdf/2407.15835) by *Bai, He and Likhomanenko, Tatiana and Zhang, Ruixiang and Gu, Zijin and Aldeneh, Zakaria and Jaitly, Navdeep* on speech tokenization for speech generation and speech recognition.



Repository contains the `dmel` pytorch-based package which performs discretization of the log mel-filterbanks for the given audio to prepare speech representations for decoder model training which will be generative model of speech.

## Installation
- from pypi
```bash
pip install dmel
```
- from source
```bash
pip install .
```

## Example of usage
We have a snipped of code to run feature extraction for both dMel and Mel and plotting their representations.
To run example:

```bash
pip install torchaudio matplotlib dmel
python run_example.py
```

The example will generate `example_mel.png` and `example_dmel.png`

## License
Repository is under [LICENSE](LICENSE).

## Citation

```
@article{bai2024dmel,
  title={dMel: Speech Tokenization Made Simple},
  author={Bai, He and Likhomanenko, Tatiana and Zhang, Ruixiang and Gu, Zijin and Aldeneh, Zakaria and Jaitly, Navdeep},
  journal={arXiv preprint arXiv:2407.15835},
  year={2024}
}
```