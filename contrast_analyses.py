from nilearn import datasets,surface,plotting
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nb
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm import compute_fixed_effects
import os 

def create_file_structure_dict(task_name, input_dir, sub_id_prefix = 'MSC'):
    all_files = []
    for sub in np.arange(1,10,1):
        for ses in np.arange(1,11,1):
            if sub < 10:
                if ses < 10:
                    curr_dir = f'{input_dir}/sub-{sub_id_prefix}0{sub}/ses-func0{ses}/func'
                else:
                    curr_dir = f'{input_dir}/sub-{sub_id_prefix}0{sub}/ses-func{ses}/func'
            else:
                if ses < 10:
                    curr_dir = f'{input_dir}/sub-{sub_id_prefix}{sub}/ses-func0{ses}/func'
                else:
                    curr_dir = f'{input_dir}/sub-{sub_id_prefix}{sub}/ses-func{ses}/func'

            if os.path.exists(curr_dir):
                file_list = os.listdir(curr_dir)
                for file in file_list:
                    if (file.split('.')[-1] == 'gz') and (file.split('_')[2].split('-')[1] == task_name):
                        all_files += [file]

    file_dict = {'sub':[], 'ses':[], 'task':[], 'run':[]}

    for bold_file in all_files:
        bold_file_split = bold_file.split('_')

        file_dict['sub'] += [bold_file_split[0].split('-')[1]]
        file_dict['ses'] += [bold_file_split[1].split('-')[1]]
        file_dict['task'] += [bold_file_split[2].split('-')[1]]

        if 'run' in bold_file_split[3]:
            file_dict['run'] += [bold_file_split[3].split('-')[1]]
        else:
            file_dict['run'] += ['no run']

    all_files_df = pd.DataFrame.from_dict(file_dict)

    return all_files_df

def run_task_baseline_contrasts(bold_file, events_file, sub, ses, task, run, tr=2.2):
    print(f'Running contrast effects for sub {sub}, session {ses}, run {run}')
    events_df = pd.read_csv(events_file,sep='\t')
    events = events_df[['onset', 'duration', 'trial_type']]

    unique_events = np.unique(events['trial_type'].values)

    contrasts = {f'{val} vs baseline': val for val in unique_events if val != 'rest'}

    lev1_glm = FirstLevelModel(tr,
                               slice_time_ref = 0.5,
                                subject_label=sub,
                                noise_model='ar1',
                                standardize=False,
                                drift_model='cosine'
                                )

    out = lev1_glm.fit(run_imgs = bold_file, 
                       events = events)

    if not os.path.exists('run_effect_size'):
        os.mkdir('run_effect_size')
    contrast_dir = 'run_effect_size'

    all_contrasts = []

    for con_name, con in contrasts.items():
        all_contrasts += [con_name]

        if run != 'no run':
            filename_base = (f'{contrast_dir}/contrast-{con_name}_sub_'
                f'{sub}_session_{ses}_task_{task}_run_{run}')
        else:
            filename_base = (f'{contrast_dir}/contrast_{con_name}_sub_'
                f'{sub}_session_{ses}_task_{task}_run_00')
        con_est_output  = out.compute_contrast(con, output_type = 'all')
        con_est_output['effect_size'].to_filename(f'{filename_base}_effect_size.nii.gz')
        con_est_output['effect_variance'].to_filename(f'{filename_base}_effect_variance.nii.gz')
    
    return all_contrasts 

def run_session_fe(contrast, ses, run, subjects):
    print(f'Running contrast average effects for contrast {contrast}, session {ses}')
    contrast_imgs = []
    variance_imgs = []

    # delete inclusion of all subjects if want subject-specific maps
    for sub in subjects:
        run_level_img_base = f'run_effect_size/contrast-{contrast}_sub'
        curr_contrast_img = f'{run_level_img_base}_{sub}_session_{ses}_task_motor_run_{run}_effect_size.nii.gz'
        curr_var_img = f'{run_level_img_base}_{sub}_session_{ses}_task_motor_run_{run}_effect_variance.nii.gz'
        # only include the run if a contrast image exists for it
        if os.path.isfile(curr_contrast_img):
            contrast_imgs += [curr_contrast_img]
            variance_imgs += [curr_var_img]

    if not os.path.exists('session_contrast_maps'):
        os.mkdir('session_contrast_maps')
    ses_contrast_dir = 'session_contrast_maps'

    filename_base = f'{ses_contrast_dir}/ses_{ses}_contrast_{contrast}'

    if len(contrast_imgs) > 0:
        fixed_fx_contrast_img, fixed_fx_variance_img, fixed_fx_stat_img, fixed_fx_z_score_img = compute_fixed_effects(contrast_imgs, 
                                                                                                                    variance_imgs, 
                                                                                                                    mask=None, 
                                                                                                                    precision_weighted=False, 
                                                                                                                    return_z_score=True)
        fixed_fx_contrast_img.to_filename(f'{filename_base}_effect_size.nii.gz')
        fixed_fx_variance_img.to_filename(f'{filename_base}_variance.nii.gz')
        fixed_fx_stat_img.to_filename(f'{filename_base}_effect_stat.nii.gz')
        fixed_fx_z_score_img.to_filename(f'{filename_base}_z_effect_size.nii.gz')
    else:
        print(f'no contrast imgs: {contrast_imgs}')

def run_contrast_effects(contrast):
    print(f'Running contrast average effects for contrast {contrast}')
    contrast_imgs = []
    variance_imgs = []

    for session in np.arange(1,11,1):
        if session >= 10:
            ses = f'func{session}'
        else:
            ses = f'func0{session}'
            
        curr_contrast_img = f'session_contrast_maps/ses_{ses}_contrast_{contrast}_effect_size.nii.gz'
        curr_var_img = f'session_contrast_maps/ses_{ses}_contrast_{contrast}_variance.nii.gz'
        if os.path.isfile(curr_contrast_img):
            contrast_imgs += [curr_contrast_img]
            variance_imgs += [curr_var_img]
    
    if not os.path.exists('average_contrast_maps'):
        os.mkdir('average_contrast_maps')
    average_contrast_dir = 'average_contrast_maps'

    filename_base = f'{average_contrast_dir}/contrast_{contrast}'

    fixed_fx_contrast_img, fixed_fx_variance_img, fixed_fx_stat_img, fixed_fx_z_score_img = compute_fixed_effects(contrast_imgs, 
                                                                                                                  variance_imgs, 
                                                                                                                  mask=None, 
                                                                                                                  precision_weighted=False, 
                                                                                                                  return_z_score=True)
    fixed_fx_contrast_img.to_filename(f'{filename_base}_effect_size.nii.gz')
    fixed_fx_variance_img.to_filename(f'{filename_base}_variance.nii.gz')
    fixed_fx_stat_img.to_filename(f'{filename_base}_effect_stat.nii.gz')
    fixed_fx_z_score_img.to_filename(f'{filename_base}_z_effect_size.nii.gz')