# NeRF-reproduce

A repo to reproduce NeRF.

## TODO

- [x] Build model and implement positional encoding( `model.py` )
- [x] Prepare and transform dataset( `datasets.py` )
- [x] Batchify the input rays
- [x] Transform the output of neural network to rgb color and weight.
- [x] Hierarchical Sampling
- [ ] Add render_only and use pretrained model to synthesize novol view
- [ ] Support more dataset
## Citation
Kudos to the authors for their amazing results:
```
@misc{mildenhall2020nerf,
    title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
    author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
    year={2020},
    eprint={2003.08934},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

However, if you find this implementation or pre-trained models helpful, please consider to cite:
```
@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
```
