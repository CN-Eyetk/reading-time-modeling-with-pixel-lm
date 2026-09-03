import numpy as np
import pandas as pd

def get_passes(ia_ids, bid = False):
    passes = np.zeros(len(ia_ids))
    init_ia_id = -1
    for t,ia_id in enumerate(ia_ids):
        if bid:
            if ia_id != init_ia_id:
                passes[t] +=1
        else:
            if ia_id > init_ia_id:
                passes[t] +=1
        init_ia_id = ia_id
    sum_passes = np.zeros(len(ia_ids))
    for i in range(len(passes)):
        sum_passes[i] = sum(passes[:i+1])
    return passes, sum_passes


def get_normal_measures(ia_ids, measures, ia_number = None):
    """
        For non durative measures
    """
    ia_ids = [int(x) for x in ia_ids]
    if ia_number is None:
        ia_number = max(ia_ids) + 1
    ia_ids = np.array(ia_ids)
    measures = np.array(measures)
    pass_ids = get_passes(ia_ids, bid = True)[1]
    ff = np.zeros(ia_number)
    fqs = np.zeros(ia_number)
    fp_mean = np.zeros(ia_number)
    #tp_sum = np.zeros(ia_number)
    tp_mean = np.zeros(ia_number)
    gpt_mean = np.zeros(ia_number)
    for idx in range(ia_number):
        fixation_mask = ia_ids == idx
        if fixation_mask.sum() > 0:
            first_pass_id = pass_ids[fixation_mask].min()
            first_pass_mask = pass_ids == first_pass_id
            first_pass_measure = measures[first_pass_mask].mean()
            fp_mean[idx] = first_pass_measure
    for i, (ia_id, measure) in enumerate(zip(ia_ids, measures)):
        tp_mean[ia_id] = ((tp_mean[ia_id] * fqs[ia_id]) + measure)/ (fqs[ia_id] + 1)
        fqs[ia_id] += 1
        #tp_sum[ia_id] += measure
        if fqs[ia_id] == 1:
            ff[ia_id] = measure        
        go_pating_measures = []
        cur_idx = i
        go_pasting = True
        while go_pasting:
            go_pating_measures.append(measures[cur_idx])
            cur_idx += 1
            if cur_idx >= len(measures):
                go_pasting = False
            elif ia_ids[cur_idx] >= ia_id:
                go_pasting = False
        if len(go_pating_measures) > 0:
            gpt_mean[ia_id] += sum(go_pating_measures) / len(go_pating_measures)
    gpt_mean[gpt_mean==0] = -1
    tp_mean[tp_mean==0] = -1
    ff[ff==0] = -1
    fp_mean[fp_mean==0] = -1
    data = {
        "FIRST_PASS_MEAN_MEASURE":fp_mean,
        "FIRST_FIXATION_MEASURE":ff,
        "TOTAL_PASS_MEAN_MEASURE":tp_mean,
        "GO_PAST_MEAN_MEASURE":gpt_mean,
    }
    
    df = pd.DataFrame(data)
    return data, df
def get_durations_and_counts(ia_ids, durations, ia_number = None):
    """
        ia_ids: the first ia should be 0
    """
    ia_ids = [int(x) for x in ia_ids]

    if ia_number is None:
        ia_number = max(ia_ids) + 1
    ia_ids = np.array(ia_ids)
    durations = np.array(durations)
    pass_ids = get_passes(ia_ids, bid = True)[1]

    fqs = np.zeros(ia_number) #Fixation Frequency
    pqs = np.zeros(ia_number) #Pass Frequency
    ffds = np.zeros(ia_number) - 1 #First Fixation Duration
    fpfds = np.zeros(ia_number) #Sum of Fixation Duration in First Pass
    fp_skip = np.zeros(ia_number) - 1
    tfds = np.zeros(ia_number)
    gpts = np.zeros(ia_number)
    
    skpfqs = np.zeros(ia_number)
    #rrdfqs = np.zeros(ia_number)
    rofqs = np.zeros(ia_number)
    rifqs = np.zeros(ia_number)
    last_ia_id = -1
    
    for idx in range(ia_number):
        #print("idx",idx)
        fixation_mask = ia_ids == idx
        #print("fm1",np.int16(fixation_mask))
        if fixation_mask.sum() > 0:
            first_pass_id = pass_ids[fixation_mask].min()
            first_pass_mask = pass_ids == first_pass_id
            if first_pass_mask.sum() > 0:
                first_fixation_step = np.argmax(first_pass_mask)
                ia_ids_before_first_fixation = ia_ids[:first_fixation_step]
                if sum(ia_ids_before_first_fixation > idx) > 0:
                    fp_skip[idx] = 1
                else:
                    fp_skip[idx] = 0
            else:
                fp_skip[idx] = 1
            
            first_pass_duration = durations[first_pass_mask].sum()
            fpfds[idx] = first_pass_duration
            #print("fm2",np.int16(first_pass_mask))

    for i, (ia_id, duration) in enumerate(zip(ia_ids, durations)):
        fqs[ia_id] += 1
        tfds[ia_id] += duration
        
        if fqs[ia_id] == 1:
            ffds[ia_id] = duration
        for j in range(last_ia_id + 1, ia_id + 1):
            
            pqs[j] += 1
            if j >= last_ia_id + 1 and j < ia_id:
                skpfqs[j] += 1
        
        go_pating_durs = []
        
        cur_idx = i
        
        if i == 0 or max(ia_ids[:i]) < ia_id:
            go_pasting = True
        while go_pasting:

            go_pating_durs.append(durations[cur_idx])

            cur_idx += 1
            if cur_idx >= len(durations):
                go_pasting = False
            elif ia_ids[cur_idx] > ia_id:
                go_pasting = False
                
        gpts[ia_id] += sum(go_pating_durs)
        last_ia_id = ia_id
        if i < len(ia_ids) - 1:
            if ia_ids[i+1] < ia_ids[i]:
                rofqs[ia_id] += 1
        if i > 0:
            if ia_ids[i-1] > ia_ids[i]:
                rifqs[ia_id] += 1
    gpts[gpts==0] = -1
    tfds[tfds==0] = -1
    fp_skip[fp_skip==-1] = 1
    fpfds[fp_skip==1] = -1
    rifqs[fqs == 0] = -1
    rofqs[fqs == 0] = -1
    data = {
        "FIXATION_FREQUENCY":fqs,
        "PROGRESSIVE_PASS_FREQUENCY":pqs,
        "FIRST_FIXATION_DURATION":ffds,
        "FIRST_PASS_DURATION":fpfds,
        "TOTAL_FIXATION_DURATION":tfds,
        "GO_PAST_DURATION":gpts,
        "PROGRESSIVE_SKIP_FREQUENCY":skpfqs,
        "REGRESSION_OUT_FREQUENCY":rofqs,
        "REGRESSION_IN_FREQUENCY":rifqs,
        "IF_SKIP":np.array(fqs == 0, dtype=np.int16),#np.(fqs == 0),
        "IF_FIRST_SKIP":np.array(fp_skip, dtype=np.int16)
    }
    
    df = pd.DataFrame(data)
    return data, df
            
            
        
        
        
if __name__ == "__main__":
    ia_ids = [0,0,0,1,0,0,1,3,2,2,5,3]
    durations = [32,31,35,81,32,31,35,81,31,33,44,11]
    print(get_durations_and_counts(ia_ids,durations)[0])

    
    #[0,  0,  0,  1,  0,  1,  3,  2]
    #[1., 1., 1., 2., 2., 3., 4., 4.]