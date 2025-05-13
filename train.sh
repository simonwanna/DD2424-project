# python src/train_breed_strat_1.py --lr 0.00001 --num_epochs 15 --batch_size 64 --augment --layer_wise_lr --L2_reg

# Possible arguments: 
# --lr X                [default: 0.00001]
# --num_epochs X        [default: 10]
# --batch_size X        [default: 64]
# --augment             [default: False]
# --layer_wise_lr       [default: False]
# --L2_reg              [default: False]
# --bn_layers           [default: True]


python src/train_breed_strat_2.py --lr 0.00001 --num_epochs 8 --batch_size 64 --augment --layer_wise_lr --L2_reg

# Possible arguments:
# --lr X                [default: 0.00001]
# --num_epochs X        [default: 10] NOTE: this is times layers to unfreeze
# --batch_size X        [default: 64]
# --augment             [default: False]
# --layer_wise_lr       [default: False]
# --L2_reg              [default: False]
# --bn_layers           [default: True]