import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import torch 
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import joblib
from tqdm import tqdm
import argparse
import json

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
def have_overlap(seg1, seg2):
    if seg1[0] > seg2[1] or seg2[0] > seg1[1]:
        return False
    else:
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--amass_path", type=str, default="/home/wuyan/euler/data/amass_raw")
    parser.add_argument("--babel_path", type=str, default="/home/wuyan/euler/data/babel_v1-0_release/babel_v1.0_release")
    parser.add_argument("--babel_split", type=str, default="train")
    args = parser.parse_args()

    process_transition = True

    upright_start = True
    robot_cfg = {
            "mesh": False,
            "rel_joint_lm": True,
            "upright_start": upright_start,
            "remove_toe": False,
            "real_weight": True,
            "real_weight_porpotion_capsules": True,
            "real_weight_porpotion_boxes": True, 
            "replace_feet": True,
            "masterfoot": False,
            "big_ankle": True,
            "freeze_hand": False, 
            "box_body": False,
            "master_range": 50,
            "body_params": {},
            "joint_params": {},
            "geom_params": {},
            "actuator_params": {},
            "model": "smpl",
        }

    smpl_local_robot = LocalRobot(robot_cfg,)

    ## amass occlusion
    amass_occlusion = joblib.load("sample_data/amass_copycat_occlusion_v3.pkl")

    ## load babel
    d_folder = args.babel_path # Data folder
    # l_babel_dense_files = ['train', ]  
    # l_babel_extra_files = []

    # # BABEL Dataset 
    # # babel_full_motion_dict = {}
    # for file in l_babel_dense_files + l_babel_extra_files:
    
    babel = {}
    babel[args.babel_split] = json.load(open(os.path.join(d_folder, args.babel_split+'.json')))

    for spl in babel:
        if "extra" in spl:
            frame_ann = "frame_anns"
            seq_ann = "seq_anns"
        else:
            frame_ann = "frame_ann"
            seq_ann = "seq_ann"
        babel_full_motion_dict = {}
        for sid in tqdm(babel[spl]):
            data = babel[spl][sid]  #
            file_name = "/".join(data["feat_p"].split("/")[1:])

            data_path = os.path.join(args.amass_path, file_name)

            splits = file_name.split("/")
            key_name_dump = "0-" + "_".join(splits).replace(".npz", "")

            bound = 0
            if key_name_dump in amass_occlusion:
                issue = amass_occlusion[key_name_dump]["issue"]
                if (issue == "sitting" or issue == "airborne") and "idxes" in amass_occlusion[key_name_dump]:
                    bound = amass_occlusion[key_name_dump]["idxes"][0]  # This bounded is calucaled assuming 30 FPS.....
                    if bound < 10:
                        print("bound too small", key_name_dump, bound)
                        continue
                else:
                    print("issue irrecoverable", key_name_dump, issue)
                    continue
                
            try:
                entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))
            except:
                print("skip {}".format(data_path))
                continue
                import ipdb; ipdb.set_trace()
            
            if not 'mocap_framerate' in  entry_data:
                print("missing mocap_framerate")
                continue
            framerate = entry_data['mocap_framerate']

            if "0-KIT_442_PizzaDelivery02_poses" == key_name_dump:
                bound = -2

            skip = int(framerate/30)
            root_trans = entry_data['trans'][::skip, :]

            pose_aa = np.concatenate([entry_data['poses'][::skip, :66], np.zeros((root_trans.shape[0], 6))], axis = -1)
            betas = entry_data['betas']
            gender = entry_data['gender']
            N = pose_aa.shape[0]
            
            if bound == 0:
                bound = N
                
            root_trans = root_trans[:bound]
            pose_aa = pose_aa[:bound]
            N = pose_aa.shape[0]
            if N < 10:
                print("N < 10")
                continue
        
            smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
            pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, smpl_2_mujoco]
            pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)

            beta = np.zeros((16))
            gender_number, beta[:], gender = [0], 0, "neutral"
            # print("using neutral model")
            smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
            smpl_local_robot.write_xml(f"phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
            skeleton_tree = SkeletonTree.from_mjcf(f"phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
            root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]

            new_sk_state = SkeletonState.from_rotation_and_root_translation(
                        skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here. 
                        torch.from_numpy(pose_quat),
                        root_trans_offset,
                        is_local=True)
            
            if robot_cfg['upright_start']:
                pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)  # should fix pose_quat as well here...

                new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
                pose_quat = new_sk_state.local_rotation.numpy()


            pose_quat_global = new_sk_state.global_rotation.numpy()
            pose_quat = new_sk_state.local_rotation.numpy()
            fps = 30

            """
            process text annotation
            """
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
                                print('no consecutive transition found, try to find overlapping segments')
                                for seg2 in frame_labels:
                                    if have_overlap([seg['start_t'], seg['end_t']], [seg2['start_t'], seg2['end_t']]) and seg2[
                                        'end_t'] > seg['end_t']:
                                        seg['proc_label'] = 'transition to ' + seg2['proc_label']
                                        seg['act_cat'] = seg['act_cat'] + seg2['act_cat']
                                        break
                                if seg['proc_label'] == 'transition':
                                    print('the transition target action not found:')
                                    seg['proc_label'] = 'transition to another action'
                                    print(sid, seg)
            else:  # the sequence has only sequence label, which means the sequence has only one action
                frame_labels = babel[spl][sid][seq_ann]['labels']  # onle one element
                frame_labels[0]['start_t'] = 0
                frame_labels[0]['end_t'] = root_trans.shape[0] / fps

            new_motion_out = {}
            new_motion_out['pose_quat_global'] = pose_quat_global
            new_motion_out['pose_quat'] = pose_quat
            new_motion_out['trans_orig'] = root_trans
            new_motion_out['root_trans_offset'] = root_trans_offset
            new_motion_out['beta'] = beta
            new_motion_out['gender'] = gender
            new_motion_out['pose_aa'] = pose_aa
            new_motion_out['fps'] = fps
            new_motion_out["frame_labels"] = frame_labels

            babel_full_motion_dict[key_name_dump] = new_motion_out

        joblib.dump(babel_full_motion_dict, "data/babel/babel_{}_upright.pkl".format(spl), compress=True)
        print("Saving {} sequences to {}".format(len(babel_full_motion_dict), "data/babel/babel_{}_upright.pkl".format(spl)))
