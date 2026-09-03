from plmd.polyeye.probe_vl.probe import AutoProberConfig, AutoProber
from plmd.polyeye.probe_vl.experiment_config import DATASET_PARAS,LANG_CONFIG
for dataset_name in ['simp_l2','trad_l2','jp_l2']:#'beijing','beijing2','simp','trad','hkc_s','hkc_p'


    models = [
            
            ('google/gemma-3-4b-it', 'gemma', {"bos_token":'<bos>', "eos_token":'<eos>'}),
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
            probe_group_by=['stimuli_id','sent_idx'] if dataset_name in ['ge_zu','ko','potec','tr','da','ge_po','ge_uk','he','es'] else ['stimuli_id'],
            ia_post_process_func = LANG_CONFIG[dataset_name]['ia_post_process_func']
            
            
        )
        prober = AutoProber(config)
        prober.load_precomputed(overwrite=False)

        df_token_metrics, df_ia_metrics = prober.annotate(sigma = -2, 
                                                        filter_conditions=[(prober.colname_of_subj_id, lambda x:x>1)], 
                                                        lm_only=True,
                                                        group_name=group_name
                                                        )
    #for model_name, group_name, fst in models:
