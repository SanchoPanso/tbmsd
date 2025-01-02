from ultralytics.data.split_dota import split_test, split_trainval

# split train and val set, with labels.
split_trainval(
    data_root="/mnt/c/Users/Alex/Downloads/DOTAv1",
    save_dir="/mnt/c/Users/Alex/Downloads/DOTAv1-split",
)

# split test set, without labels.
split_test(
    data_root="/mnt/c/Users/Alex/Downloads/DOTAv1",
    save_dir="/mnt/c/Users/Alex/Downloads/DOTAv1-split",
)