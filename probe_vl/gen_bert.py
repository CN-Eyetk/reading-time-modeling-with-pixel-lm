from probe import AutoProberConfig, AutoProber
from experiment_config import DATASET_PARAS,LANG_CONFIG
for dataset_name in ['es','es_ch','ge_po','ge_zu','ko','simp','trad','tr','ge_uk','persian2','en']:#,'ge_zu','ko','potec','tr','zuco']:#'beijing','beijing2','simp','trad','hkc_s','hkc_p'

    models = [
            (LANG_CONFIG[dataset_name]['bert'], 'bert', LANG_CONFIG[dataset_name]['force_special_tokens_for_bert']),
            ]
    for model_name, group_name, fst in models:
        config = AutoProberConfig(
            path_to_model = model_name,
            path_to_processor= model_name,
            path_to_plot_paras = None,
            dataset_paras = '../scripts/probe_paras.yml',
            dataset_name = dataset_name,
            is_vl=False,
            force_special_tokens=fst,
            feat_idx_for_sent_idx = -1,
            is_spaced=LANG_CONFIG[dataset_name]['is_spaced'],
            probe_group_by=['stimuli_id','sent_idx'] if dataset_name in ['ge_zu','ko','potec','tr','da','ge_po','ge_uk','he','es','tr_w1'] else ['stimuli_id'],
            ia_post_process_func = LANG_CONFIG[dataset_name]['ia_post_process_func'],
            is_bidir=True
            
            
        )
        prober = AutoProber(config)
        prober.load_precomputed(overwrite=False)
        #if 1 == 2:
        df_token_metrics, df_ia_metrics = prober.annotate(sigma = -2, 
                                                        filter_conditions=[(prober.colname_of_subj_id, lambda x:x>1)], 
                                                        lm_only=True,
                                                        group_name=group_name
                                                        )
    #for model_name, group_name, fst in models:
