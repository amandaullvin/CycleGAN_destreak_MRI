CycleGAN model for destreaking of radial MRI data

Reasonable parameters for training 
```python3 /workspace/pytorch-CycleGAN-and-pix2pix/train.py --dataroot /workspace/pytorch-CycleGAN-and-pix2pix/datasets/destreak --name destreak --model cycle_gan --display_freq 100 --loadSize 256 --fineSize 256 --lr 0.0002 --nepoch 200 --nepoch_decay 200 --which_model_netD n_layers --n_layers_D 2 --no_dropout --lambda_feat 1
```

## Acknowledgments
Code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [CycleGANwithPerceptionLoss](https://github.com/EliasVansteenkiste/CycleGANwithPerceptionLoss).
