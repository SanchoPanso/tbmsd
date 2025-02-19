from ultralytics.data.split_dota import split_test, split_trainval

# split train and val set, with labels.
split_trainval(
    data_root="/mnt/d/datasets/dota/DOTAv1",
    save_dir="/mnt/d/datasets/dota/DOTAv1-split",
    crop_size=1024,
)

# split test set, without labels.
split_test(
    data_root="/mnt/d/datasets/dota/DOTAv1",
    save_dir="/mnt/d/datasets/dota/DOTAv1-split",
    crop_size=1024,
)