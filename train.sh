python src/train_breed_strat_1.py
echo "Training breed strat 1 done"

python src/train_breed_strat_2.py
echo "Training breed strat 2 done"

# Possible arguments: 
# --lr X                [default: 0.00001]
# --num_epochs X        [default: 10]
# --batch_size X        [default: 64]
# --augment             [default: False]
# --layer_wise_lr       [default: False]
# --L2_reg              [default: False]
# --bn_layers           [default: True]
