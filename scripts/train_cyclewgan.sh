python3.5 train_wgan.py --dataroot ./datasets/RainRemoval --name rainremoval_cyclewgan01 --model cycle_wgan --no_dropout --batchSize 3 --display_id 0 --nepoch 10 --nepoch_decay 10 --lambda_A 10.0 --lambda_B 10.0 --lambda_feat 1.0 --print_freq 1
