"""Analysis for creativity task
BEHAVIOR RELATED FUNCTIONS
"""


import numpy as np
from neo import io
import quantities as pq
import glob   
import matplotlib.pyplot as plt
import pandas as pd
import mat73


import sys
sys.path.append('../')

import DMN_LFP_funcs

import scipy

import brpylib


def contains_nan_nested(nested_list):
    for inner_list in nested_list:
        if any(pd.isna(item) for item in inner_list):
            return True
    return False


def trial_type_subset_df(trial_info_df, trial_type, practice_prompts = [['key'], ['10 + 1 = ']]):
    """ Find where practice trials end and real session starts. 
        (Practice AUT and OCT trials start with promot "key"; BAT trials start with '10 + 1 = ')
        Trial type is AUT, OCT, or BAT
        practice_trial_prompts is 'key' or '10 + 1 = '
    """
    practice_trials = trial_info_df['stimVal'].apply(lambda x: x[0] in practice_prompts)
    trial_info_df_cut = trial_info_df[~practice_trials]

    # Filter out rows with NaNs in respRepStartNIP (e.g. if trial got paused and have stimuli in df without start times)
    cleaned_trial_info_df = trial_info_df_cut[~trial_info_df_cut['respRepStartNIP'].apply(contains_nan_nested)].reset_index(drop=True)



    if trial_type=='all':
        trial_type_subset_df = cleaned_trial_info_df
    else:
        trial_type_subset_df = cleaned_trial_info_df[cleaned_trial_info_df['stimType']==trial_type] #just the AUT trials
    return(trial_type_subset_df)


def get_trial_info_df(data_loc):    #get trial info from matlab file
    matfile = glob.glob(data_loc + '/TrialInfo*')
    try:
        trial_info = scipy.io.loadmat(matfile[0])
        trial_info_df = pd.DataFrame(trial_info['TrialInfo'][0])
    except:
        trial_info = mat73.loadmat(matfile[0])
        trial_info_df = pd.DataFrame(trial_info['TrialInfo'])
    trial_info_df.drop('keyPressTimesPTB',axis=1,inplace=True)
    trial_info_df.dropna(inplace=True)

    #patient UIC202503 was distracted lots... remove a bunch of trials
    if '20250130-153652' in data_loc:
        trial_info_df = trial_info_df.loc[:12]
    if '20250131-150132' in data_loc:
        trial_info_df.drop([9,11,19], inplace=True)

    return (trial_info_df)


def get_start_end_times_ms(trial_info_df, trial_type, pre_trial_buffer=0, data_loc=0):
    ''' Get the start and end times in ms of each trial for given trial type. 
    End times are end of stimulus period. End of response period would be end times + 6000ms
    '''
    
    ####Find where practice trials end and real session starts. Practice AUT and OCT trials start with promot "key"; BAT trials start with '10 + 1 = '
    trial_subset_df = trial_type_subset_df(trial_info_df, trial_type, practice_prompts=[['key'], ['10 + 1 = ']]) 

    ###find time slices when AUT (or whichever) trials are happening
    trial_start_times_30khz = trial_subset_df['respDurStartNIP']
    trial_end_times_30khz = trial_subset_df['respDurFinishNIP']
    trial_start_times_ms = trial_start_times_30khz.values/30.
    trial_end_times_ms = trial_end_times_30khz.values/30.
    trial_start_times_ms = np.array(trial_start_times_ms, dtype=float).flatten()
    trial_end_times_ms = np.array(trial_end_times_ms, dtype=float).flatten()

    ###get times for each sub-trial (3 items per trial)
    start_end_info = trial_subset_df['respRepStartNIP']
    mask = start_end_info.apply(lambda x: not any(np.isnan(val) for sublist in x for val in sublist)) #remove nans if any
    start_end_info = start_end_info[mask]

    start_times_30khz = np.concatenate([start_end_info.iloc[i][:,0] for i in np.arange(len(start_end_info))])
    end_times_30khz = np.concatenate([start_end_info.iloc[i][:,1] for i in np.arange(len(start_end_info))])
    start_times_ms = start_times_30khz/30. - pre_trial_buffer #subtract pre_trial_buffer if want to calculate a little before the prompt appears
    end_times_ms = end_times_30khz/30.

    # if (patient=='UIC202522') | (patient=='UIC202521'): #new system: times not 30KHz but rather 1e9 sampling rate
    if int(data_loc[-15:-7]) >= 20251213:#new system: times not 30KHz but rather 1e9 sampling rate
        ns2_file = glob.glob(data_loc + '/Hub1' + '*.ns2')[0] #
        ns2_file_loaded = brpylib.NsxFile(ns2_file)
        ns2_data = ns2_file_loaded.getdata()
        timestamp = ns2_data['data_headers'][0]['Timestamp']

        corrected_start_times = (start_times_30khz - timestamp)/1e9 #in seconds
        corrected_end_times = (end_times_30khz - timestamp)/1e9
        corrected_trial_start_times = (trial_start_times_30khz - timestamp)/1e9

        start_times_ms = corrected_start_times * 1000 - pre_trial_buffer
        end_times_ms = corrected_end_times * 1000
        trial_start_times_ms = corrected_trial_start_times * 1000

    return(start_times_ms, end_times_ms, trial_start_times_ms)


####choose start and end of window (which can be stimulus, response, or whole trial)
def get_window_times(window,start_times_ms,end_times_ms):
    if window=='stimulus':
        win_start = start_times_ms
        win_end = end_times_ms
    elif window=='response':
        win_start = end_times_ms
        win_end = end_times_ms + (6*1000) + 1 ##add +1 because for some reason end - start needs to be odd, or else frequencies in psd don't match...
    else: #plot whole trial--stim & resp
        win_start = start_times_ms
        win_end = end_times_ms + (6*1000)
    return(win_start,win_end)


def create_csv_template(patient, data_loc, trial_type):
    """Create session specific CSV template for this patient's prompts, and start/end times for each subtrial
        Will still need to listen to .wav file to get responses and manually enter those, as well as creativity scores"""
    
    import csv
    csv_file = 'scored_responses/' + trial_type + '_' + patient + '_items_responses_template.csv'
    
    trial_info_df = get_trial_info_df(data_loc)
    df = trial_type_subset_df(trial_info_df, trial_type, practice_prompts=[['key'], ['10 + 1 = ']]) #just the AUT trials

    # prompts = np.concatenate(df.stimVal.values) 
    # prompts = [item[0][0] for item in np.concatenate(df.stimVal.values)] 
    prompts = [item[0] for item in np.concatenate(df.stimVal.values)] 
    rep_times = np.concatenate(df.respRepStartNIP.values) #[start,end] times for each AUT sub-trial
    originality_template = np.tile(np.nan,len(prompts))
    elaboration_template = np.tile(np.nan,len(prompts))
    response_template = np.tile(np.nan,len(prompts))
    row_list_for_csv = list(zip(prompts, response_template, originality_template, elaboration_template, rep_times[:,0],rep_times[:,1]))

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        field = ['Item', 'response', 'originality', 'elaboration', 'Start', 'End']
        writer.writerow(field)
        writer.writerows(row_list_for_csv)
    
    contents_df = pd.DataFrame(row_list_for_csv,columns = field)
    return(contents_df)
 

#this is actually the same as above... 
def create_csv_template_from_ripple(patient, data_loc, trial_type):
    """Create session specific CSV template for this patient's prompts, and start/end times for each subtrial
        Will still need to listen to .wav file to get responses and manually enter those, as well as creativity scores"""
    
    import csv
    csv_file = 'scored_responses/' + trial_type + '_' + patient + '_items_responses_template.csv'
    
    trial_info_df = get_trial_info_df(data_loc)
    df = trial_type_subset_df(trial_info_df, trial_type, practice_prompts=[['key'], ['10 + 1 = ']]) #just the AUT trials

    # prompts = np.concatenate(df.stimVal.values) 
    prompts = [item[0] for item in np.concatenate(df.stimVal.values)] 
    rep_times = np.concatenate(df.respRepStartNIP.values) #[start,end] times for each AUT sub-trial
    originality_template = np.tile(np.nan,len(prompts))
    elaboration_template = np.tile(np.nan,len(prompts))
    response_template = np.tile(np.nan,len(prompts))
    row_list_for_csv = list(zip(prompts, response_template, originality_template, elaboration_template, rep_times[:,0],rep_times[:,1]))

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        field = ['Item', 'response', 'originality', 'elaboration', 'Start', 'End']
        writer.writerow(field)
        writer.writerows(row_list_for_csv)
    
    contents_df = pd.DataFrame(row_list_for_csv,columns = field)
    return(contents_df)


def create_csv_template_old(df,patient):
    """Create session specific CSV template for this patient's prompts, and start/end times for each subtrial
        Will still need to listen to .wav file to get responses and manually enter those, as well as creativity scores"""
    
    import csv
    csv_file = 'scored_responses/' + patient + '_items_responses_template.csv'

    # prompts = np.concatenate(df.stimVal.values) 
    prompts = [item[0] for item in np.concatenate(df.stimVal.values)] 
    rep_times = np.concatenate(df.respRepStartNIP.values) #[start,end] times for each AUT sub-trial
    row_list_for_csv = list(zip(prompts,rep_times[:,0],rep_times[:,1]))

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        field = ['Item', 'Start', 'End']
        writer.writerow(field)
        writer.writerows(row_list_for_csv)
    
    contents_df = pd.DataFrame(row_list_for_csv,columns = field)
    return(contents_df)


####Get the photodiode trace from the ns5 file.
def get_phototrace(data_loc):
    ns5_file = glob.glob(data_loc + '/' + '*.ns5')[0]
    print('loading NS5 file')
    read_ns5 = io.BlackrockIO(ns5_file)
    print('reading')
    block_ns5 = read_ns5.read_block()
    photo_trace = block_ns5.segments[0].analogsignals[2].as_array()[:,0]
    thresh = 3500
    on_times = np.where(photo_trace>thresh)[0]
    on_starts_inds =  np.where(on_times[1:] - on_times[:-1] > 5)[0] #greater than 5 (rather than 1) in case of random wobbles(?)
    on_start_times = on_times[on_starts_inds+1] #these are the times the trials start!



# def count_syllables(AUT_response_df):
#     '''    
#     COUNT NUMBER OF SYLLABLES IN EACH RESPONSE IN DF 
#     RETURN LIST WHERE EACH ENTRY IS NUMBER OF SYLLABLES FOR EACH RESPONSE
#     '''

#     import spacy 
#     nlp = spacy.load('en_core_web_lg')
#     import nltk
#     nltk.download('cmudict')
#     d = cmudict.dict()

#     syllables_list_for_df = []

#     for response in AUT_response_df.response:
#         doc = nlp(response)
#         num_syllables = 0
#         num_phonemes = 0
#         for token in doc:
#             try:
#                 processed_word = d[str(token).lower()][0]  
#                 num_syllables += len([y for y in processed_word if y[-1].isdigit()])
#                 num_phonemes += len(processed_word)

#             except: #not in dictionary
#                 num_syllables += int(input('Manually enter number of syllables in "' + str(token) +'": '))
#         # print(response + '\nSyllables: ' + str(num_syllables) + '\nPhonemes: ' + str(num_phonemes) + '\n\n')
#         syllables_list_for_df.append(num_syllables)
#     return(syllables_list_for_df)


#remove stopwords, remove punctuation, lemmatize
def clean_phrases(phrases):
    import spacy 
    import spacy_cleaner
    from spacy_cleaner import processing, Cleaner
    from spacy_cleaner.processing import removers, replacers, mutators
    nlp = spacy.load('en_core_web_lg')

    nlp.Defaults.stop_words.add('use')  #remove the word "use" as well
    nlp.Defaults.stop_words.add('things')
    nlp.Defaults.stop_words.add('stuff')

    try:
        nlp.Defaults.stop_words.remove('put')
    except:
        print('put already removed')

    # pipeline = spacy_cleaner.Pipeline(nlp, removers.remove_stopword_token,
    # removers.remove_punctuation_token,
    # mutators.mutate_lemma_token)
    # pipeline2 = spacy_cleaner.Pipeline(nlp, removers.remove_stopword_token,
    # removers.remove_punctuation_token) #no lemmatization

    pipeline = spacy_cleaner.Cleaner(nlp, removers.remove_stopword_token,
    removers.remove_punctuation_token,
    mutators.mutate_lemma_token)

    pipeline2 = spacy_cleaner.Cleaner(nlp, removers.remove_stopword_token,
    removers.remove_punctuation_token) #no lemmatization

    phrases = phrases.astype(str)
    cleaned_phrases = [pipeline2.clean([phrase]) for phrase in phrases]
    #make it so that "cleaned_phrases" is a list of strings, not a list of lists
    import itertools as it
    cleaned_phrases = list(it.chain.from_iterable(cleaned_phrases))

    return(cleaned_phrases)
