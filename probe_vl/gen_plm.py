from probe import AutoProberConfig, AutoProber
from experiment_config import DATASET_PARAS,LANG_CONFIG
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Extract Pixel LM predictors.")

    parser.add_argument(
        "--pixel_lm_right_blur",
        action='store_true'
    )
    parser.add_argument(
        "--pixel_whole_ia_masking",
        action='store_true'
    )
    parser.add_argument(
        "--is_m4",
        action='store_true'
    )
    parser.add_argument(
        "--is_denoiser",
        action='store_true'
    )
    parser.add_argument(
        "--pixel_preview_window",
        default=0,
        type=int
    )
    parser.add_argument(
        "--pixel_remove_white",
        action='store_true'
    )
    parser.add_argument(
        "--pixel_blur_kernel_size",
        default=5,
        type=int
    )
    parser.add_argument(
        "--pixel_lm_right_cut",
        action='store_true'
    )
    return parser.parse_args()

args = parse_args()


for dataset_name in ['es','es_ch','ge_po','ge_zu','ko','simp','trad','tr','ge_uk','persian2','en']:
    if not args.is_m4:
        path_to_model = "Team-PIXEL/pixel-base"
    else:
        print(f"dataset_name={dataset_name}")
        path_to_model = LANG_CONFIG[dataset_name]['pixel']

    print("Loading model",path_to_model)
    config = AutoProberConfig(
        path_to_model = path_to_model,
        path_to_processor= path_to_model,
        path_to_plot_paras = None,
        dataset_paras = '../scripts/probe_paras.yml',
        dataset_name = dataset_name,
        is_pixel_lm=True,
        is_vl=False,
        feat_idx_for_sent_idx = -1,
        is_spaced=LANG_CONFIG[dataset_name]['is_spaced'],
        probe_group_by=['stimuli_id','sent_idx'] if dataset_name in ['ge_zu','ko','potec','tr','da','ge_po','ge_uk','es','he','tr_w1','ko_2','ge_zu_2','ge_po_2','es_2'] else ['stimuli_id'],
        use_force_segmentation=LANG_CONFIG[dataset_name]['use_force_segmentation'],
        pixel_lm_right_blur = args.pixel_lm_right_blur,
        pixel_whole_ia_masking = args.pixel_whole_ia_masking,
        is_pixel_lm_m4=args.is_m4,
        pixel_use_arabic_renderer=True if dataset_name in ['persian','persian2'] else False,
        is_hebrew=True if dataset_name in ['he'] else False,
        is_pixel_denoiser=args.is_denoiser,
        ia_post_process_func = lambda x:x.replace("fs","").replace("g","").replace("f","").replace("h",""),
        pixel_preview_window = args.pixel_preview_window,
        pixel_remove_white = args.pixel_remove_white,
        pixel_blur_kernel_size = args.pixel_blur_kernel_size,
        pixel_lm_right_cut = args.pixel_lm_right_cut
    )
    prober = AutoProber(config)
    prober.load_precomputed(overwrite=False)
    #if 1 == 2:
    group_name='pixel' if not config.pixel_lm_right_blur else 'pixel_uni'
    if config.pixel_whole_ia_masking:
        group_name += "_fm"
    if not config.is_pixel_lm_m4:
        group_name += "_v1"
    if args.is_denoiser:
        group_name = "dn_" + group_name
    if args.pixel_preview_window > 0:
        group_name += f"pv_{args.pixel_preview_window}"
    if args.pixel_remove_white:
        group_name += "_nw"
    if args.pixel_blur_kernel_size != 5:
        group_name += f"_k{args.pixel_blur_kernel_size}"
    if args.pixel_lm_right_cut:
        group_name += "_cut"
    df_token_metrics, df_ia_metrics = prober.annotate(sigma = -2, 
                                                    filter_conditions=[(prober.colname_of_subj_id, lambda x:x>1)], 
                                                    lm_only=True,
                                                    group_name=group_name
                                                    )
    
