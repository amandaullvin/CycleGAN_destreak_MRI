python train_wgan.py --dataroot ./datasets/RainRemoval --name RainRemoval50  --model cycle_wgan --no_dropout --batchSize 5 --pool_size 5 --no_html --adam --norm batch --which_model_netD dcgan --lr 0.00005 --nepoch 100 --nepoch_decay 150 --lambda_A 5.0 --lambda_B 5.0 --lambda_feat 0.0 --identity 0.0 --print_freq 1 --display_id -1
#--norm batch
#--which_model_netD dcgan
#--which_model_netG resnet_6blocks