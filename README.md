# ReliSoilNet
ReliSoilNet: A Reliability-Aware Hybrid Self-Supervised Multimodal Learning Framework for Soil Organic Carbon Prediction



### **Model Training**

Tip: For LUCAS, the random seeds were {1,42,86\}. For RaCA, \{1,43,86\} was used when Degraded-modality robust learning(DMRL) was enabled and \{1,42,86\}.

##### 0.Data Preparation:

Set the paths to the data in the `config.py` file.

```
train_l8_folder_path = 'ReliSoilNet/dataset/l8_images/train'
test_l8_folder_path = 'ReliSoilNet/dataset/l8_images/test'
val_l8_folder_path = 'ReliSoilNet/dataset/l8_images/val'
lucas_csv_path = 'ReliSoilNet/CSV/LUCAS_2015.csv'
climate_csv_folder_path = "ReliSoilNet/dataset/Climate/LUCAS_Climate_Data"

# if have pre-trained model:
# SIMCLR_PATH = "ReliSoilNet/results/RUN_LUCAS_HybridSSL_ViT_Trans_SelfSupervised_best.pth"
```



##### 1.pre-trained:

```
LUCAS:
python train_ssl.py -exp LUCAS_HybridSSL_ViT_Trans -ds LUCAS -nw 0 -trbs 32 -lr 0.0001 -ne 100 -lrs step -srtm -lstm -cnn ViT -rnn Transformer -s 1 42 86 --ssl_objective hybrid --temperature 0.5 --lambda_img_mask 0.25 --lambda_clim_mask 1.0 --img_mask_ratio 0.35 --clim_mask_ratio 0.25 --clim_mask_mode block --clim_block_size 3 --img_mask_patch_size 8 --img_decoder_hidden 256 --clim_decoder_hidden 256 --aux_dropout 0.1 --train_l8_dir "ReliSoilNet/dataset/l8_images/train" --test_l8_dir "ReliSoilNet/dataset/l8_images/test" --val_l8_dir "ReliSoilNet/dataset/l8_images/val" --main_csv_path "ReliSoilNet/CSV/LUCAS_2015.csv" --climate_csv_dir "ReliSoilNet/dataset/Climate/LUCAS_Climate_Data"


RaCA:
python train_ssl.py -exp RaCA_HybridSSL_ViT_Trans -ds RaCA -nw 0 -trbs 32 -lr 0.0001 -ne 100 -lrs step -srtm -lstm -cnn ViT -rnn Transformer -s 1 42 86 --ssl_objective hybrid --temperature 0.5 --lambda_img_mask 0.25 --lambda_clim_mask 1.0 --img_mask_ratio 0.35 --clim_mask_ratio 0.25 --clim_mask_mode block --clim_block_size 3 --img_mask_patch_size 8 --img_decoder_hidden 256 --clim_decoder_hidden 256 --aux_dropout 0.1 --train_l8_dir "ReliSoilNet/dataset/l8_images_us/train" --test_l8_dir "ReliSoilNet/dataset/l8_images_us/test" --val_l8_dir "ReliSoilNet/dataset/l8_images_us/val" --main_csv_path "ReliSoilNet/CSV/RaCA.csv" --climate_csv_dir "ReliSoilNet/dataset/Climate/RaCA_Climate_Data"
```



##### 2.Fine-tuning:

```
LUCAS:
python train.py -e Hybrid_LUCAS -d LUCAS -w 0 -simclr -trbs 64 -lr 0.0001 -ne 50 -ls step -srtm -lstm -log -seed 1 42 86 -fm gated -gh 64 -gd 0.10 -rm feature_fallback -rmod climate -rbp 0.30 -rdp 0.12 -rms 4 -rcs 0.35 -rsw 0.10 -rcw 0.04 -rgw 0.06 -rgm 0.08 -rtb 0.20 -rpm 0.92 -rwe 0


RaCA:
python train.py -e Hybrid_RaCA -d RaCA -w 0 -simclr -trbs 64 -lr 0.0001 -ne 50 -ls step -srtm -lstm -log -seed 1 43 86 -fm gated -gh 64 -gd 0.10 -rm feature_fallback -rmod climate -rbp 0.45 -rdp 0.18 -rms 6 -ris 0.25 -rcs 0.45 -rsw 0.14 -rcw 0.06 -rgw 0.10 -rgm 0.10 -rtb 0.20 -rpm 0.92 -rwe 0
```



**Help:**

For a detailed explanation of the arguments, you can run the following command:

```
python train_ssl.py --help
python train.py --help
```

##### This repository will be updated gradually.





























