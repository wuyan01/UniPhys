from huggingface_hub import snapshot_download
import huggingface_hub.utils as hub_utils
import logging
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
hub_utils.logging.set_verbosity_warning()


if __name__ == "__main__":

    # # download the tracked data from huggingface
    babel_save_path = "./data"
    print("Downloading the tracked BABEL data from huggingface to {}...".format(babel_save_path))
    snapshot_download(
        repo_id="yan0116/SMPL_Humanoid_offline_dataset",
        repo_type="dataset",
        allow_patterns="babel_*/*.pkl",
        local_dir=babel_save_path,
        max_workers=4,
        tqdm_class=None,
    )
    print("Done!")

