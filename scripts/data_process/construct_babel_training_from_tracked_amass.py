import os
import json
from tqdm import tqdm
import joblib
import argparse
from huggingface_hub import snapshot_download
import huggingface_hub.utils as hub_utils
import logging
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
hub_utils.logging.set_verbosity_warning()

def have_overlap(seg1, seg2):
    if seg1[0] > seg2[1] or seg2[0] > seg1[1]:
        return False
    else:
        return True
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--babel_path", type=str, default="./data/babel_v1-0_release/babel_v1.0_release")
    parser.add_argument("--babel_split", type=str, default="val")
    args = parser.parse_args()


    # # download the tracked amass data from huggingface
    amass_save_path = "./data"
    print("Downloading the tracked AMASS data from huggingface to {}...".format(amass_save_path))
    snapshot_download(
        repo_id="yan0116/SMPL_Humanoid_offline_dataset",
        repo_type="dataset",
        allow_patterns="amass_*/*",
        local_dir=amass_save_path,
        max_workers=4,
        tqdm_class=None,
    )

    # construct and package the babel training set
    print("Constructing and packaging the babel {} set...".format(args.babel_split))
    babel = {}
    babel[args.babel_split] = json.load(open(os.path.join(args.babel_path, args.babel_split+'.json')))

    spl = args.babel_split

    if "extra" in spl:
        frame_ann = "frame_anns"
        seq_ann = "seq_anns"
    else:
        frame_ann = "frame_ann"
        seq_ann = "seq_ann"

    babel_motion_dict = {}
    babel_motion_dict["is_succ_all"] = []
    babel_motion_dict["root_state_all"] = []
    babel_motion_dict["dof_state_all"] = []
    babel_motion_dict["body_pos_all"] = []
    babel_motion_dict["action_all"] = []
    babel_motion_dict["z_all"] = []
    babel_motion_dict["frame_labels_all"] = []
    babel_motion_dict["motion_file"] = []

    for sid in tqdm(babel[spl]):
        data = babel[spl][sid]  #
        file_name = "/".join(data["feat_p"].split("/")[1:]).replace("npz", "pkl")

        if os.path.isfile(os.path.join(amass_save_path, "data", file_name)):
            tracking_result = joblib.load(os.path.join(amass_save_path, "data", file_name))
        else:
            # print(f"{os.path.join(amass_save_path, file_name)} does not exist, skipped")
            continue

        babel_motion_dict["is_succ_all"].append(tracking_result["is_succ"])
        babel_motion_dict["root_state_all"].append(tracking_result["root_state"])
        babel_motion_dict["dof_state_all"].append(tracking_result["dof_state"])
        babel_motion_dict["body_pos_all"].append(tracking_result["body_pos"])
        babel_motion_dict["action_all"].append(tracking_result["action"])
        babel_motion_dict["z_all"].append(tracking_result["pulse_z"])
        babel_motion_dict["motion_file"].append(file_name)

        N = tracking_result["pulse_z"].shape[0]
        process_transition = True
        fps = 30
        if frame_ann in babel[spl][sid] and babel[spl][sid][frame_ann] is not None:
                frame_labels = babel[spl][sid][frame_ann]['labels']
                # process the transition labels, concatenate it with the target action
                for seg in frame_labels:
                    if process_transition:
                        if seg['proc_label'] == 'transition':
                            for seg2 in frame_labels:
                                if seg2['start_t'] == seg['end_t']:
                                    seg['proc_label'] = 'transition to ' + seg2['proc_label']
                                    seg['act_cat'] = seg['act_cat'] + seg2['act_cat']
                                    break
                            if seg['proc_label'] == 'transition':
                                # print('no consecutive transition found, try to find overlapping segments')
                                for seg2 in frame_labels:
                                    if have_overlap([seg['start_t'], seg['end_t']], [seg2['start_t'], seg2['end_t']]) and seg2[
                                        'end_t'] > seg['end_t']:
                                        seg['proc_label'] = 'transition to ' + seg2['proc_label']
                                        seg['act_cat'] = seg['act_cat'] + seg2['act_cat']
                                        break
                                if seg['proc_label'] == 'transition':
                                    # print('the transition target action not found:')
                                    seg['proc_label'] = 'transition to another action'
                                    # print(sid, seg)
        else:  # the sequence has only sequence label, which means the sequence has only one action
            frame_labels = babel[spl][sid][seq_ann]['labels']  # onle one element
            frame_labels[0]['start_t'] = 0
            frame_labels[0]['end_t'] = N / fps

        babel_motion_dict["frame_labels_all"].append(frame_labels)

    babel_motion_dict["succ_idxes"] = [i for i, is_succ in enumerate(babel_motion_dict["is_succ_all"]) if is_succ]
    print("Saving the packaged babel {} set to data/babel_state-action-text-pairs/babel_{}.pkl".format(spl, spl))
    print("Total number of motion sequences: {}; Successfully tracking sequences: {}".format(len(babel_motion_dict["motion_file"]), len(babel_motion_dict["succ_idxes"])))

    os.makedirs("data/babel_state-action-text-pairs", exist_ok=True)
    joblib.dump(babel_motion_dict, "data/babel_state-action-text-pairs/babel_{}.pkl".format(spl), compress=True)

    
