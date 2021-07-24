export CUDA_VISIBLE_DEVICES=0
python  train.py --batch 4 ./dataset_3d/FAUST/  --edge_loss_setting 'rec_rec' --experi_path 'full_adap_GIH_Lap_rec_rec_1200_1600_2000_geo_005' --server 'local' --rec_epoch 1200 --geo_epoch 1600 --iter 2000 --geoloss 0.05 --n_crop 1 --ref_crop 2 --sampling_number 60 --limb_n 1 --limb_sampling 600  --sampling_pattern 'adaptive'
