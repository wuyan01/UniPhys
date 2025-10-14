mkdir sample_data
mkdir -p output output/HumanoidIm/ output/UniPhys/ output/HumanoidIm/phc_3 output/HumanoidIm/phc_comp_3 output/HumanoidIm/pulse_vae_iclr output/UniPhys/checkpoints
mkdir -p data
mkdir -p data/babel_state-action-text-pairs
gdown https://drive.google.com/uc?id=1bLp4SNIZROMB7Sxgt0Mh4-4BLOPGV9_U -O  sample_data/ # filtered shapes from AMASS
gdown https://drive.google.com/uc?id=1arpCsue3Knqttj75Nt9Mwo32TKC4TYDx -O  sample_data/ # all shapes from AMASS
gdown https://drive.google.com/uc?id=1uzFkT2s_zVdnAohPWHOLFcyRDq372Fmc -O  sample_data/ # amass_occlusion_v3
gdown https://drive.google.com/uc?id=1Vi-zrSIga9Da9alhiMDes-yf2pxkvObP -O sample_data/amass_isaac_standing_upright_slim_long.pkl
gdown https://drive.google.com/uc?id=1ztyljPCzeRwQEJqtlME90gZwMXLhGTOQ -O  output/HumanoidIm/pulse_vae_iclr/
gdown https://drive.google.com/uc?id=1S7_9LesLjfsFYqi4Ps6Sjzyuyun0Oaxi -O  output/HumanoidIm/pulse_vae_x/
gdown https://drive.google.com/uc?id=1JbK9Vzo1bEY8Pig6D92yAUv8l-1rKWo3 -O  output/HumanoidIm/phc_comp_3/
gdown https://drive.google.com/uc?id=1pS1bRUbKFDp6o6ZJ9XSFaBlXv6_PrhNc -O  output/HumanoidIm/phc_3/
gdown https://drive.google.com/uc?id=1g8Cf-V0wrFpFuZT2SO8wMauR19QN6MOi -O output/UniPhys/checkpoints/uniphys_T32.ckpt # UniPhys pretrained checkpoint
gdown https://drive.google.com/uc?id=1JdQFAbA_UkijiMLu5Q69KDePtbYZyO14 -O data/babel_state-action-text-pairs/text_embedding_dict_clip.pkl  # preprocessed text clip embedding
gdown https://drive.google.com/uc?id=1lIAXvQo8ALDd-alLJvSJgQzpuxKqueZC -O output/UniPhys/train_data_stats.npy # training data stats [mean,std]
