
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import nltk
import re
from datetime import datetime
get_ipython().magic('matplotlib inline')

essays = pd.read_csv('opendata_essays000.gz', escapechar='\\', names=['_projectid', '_teacherid', 'title', 'short_description', 'need_statement', 'essay', 'thankyou_note', 'impact_letter'])
projects = pd.read_csv('opendata_projects000.gz', escapechar='\\', names=['_projectid', '_teacher_acctid', '_schoolid', 'school_ncesid', 'school_latitude', 'school_longitude', 'school_city', 'school_state', 'school_zip', 'school_metro', 'school_district', 'school_county', 'school_charter', 'school_magnet', 'school_year_round', 'school_nlns', 'school_kipp', 'school_charter_ready_promise', 'teacher_prefix', 'teacher_teach_for_america', 'teacher_ny_teaching_fellow', 'primary_focus_subject', 'primary_focus_area' ,'secondary_focus_subject', 'secondary_focus_area', 'resource_type', 'poverty_level', 'grade_level', 'vendor_shipping_charges', 'sales_tax', 'payment_processing_charges', 'fulfillment_labor_materials', 'total_price_excluding_optional_support', 'total_price_including_optional_support', 'students_reached', 'total_donations', 'num_donors', 'eligible_double_your_impact_match', 'eligible_almost_home_match', 'funding_status', 'date_posted', 'date_completed', 'date_thank_you_packet_mailed', 'date_expiration'])


# In[5]:

#for testing purposes
essays0 = essays[:4000]
projects0 = projects[:4000]


# In[6]:

def clean_data(projects, essays):

    def convert(DataFrame):  #convert categorical columns
        for i in range(len(DataFrame.columns)):   
            if(DataFrame[DataFrame.columns[i]].dtype=='O' and DataFrame.columns[i] != 'school_state' and DataFrame.columns[i] != 'essay'):
                DataFrame[DataFrame.columns[i]]=DataFrame[DataFrame.columns[i]].astype('category')
                length=len(DataFrame[DataFrame.columns[i]].unique())
                DataFrame[DataFrame.columns[i]]=DataFrame[DataFrame.columns[i]].cat.rename_categories([j for j in range(length)])
        return DataFrame

    def tokenize_only(text):
        # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        return filtered_tokens
    
    projects = pd.merge(projects, essays, how='inner', left_on='_projectid', right_on = '_projectid')
    projects = projects.drop(['_teacher_acctid','_schoolid','school_latitude','school_ncesid','school_latitude','school_longitude',
                  'school_city','school_zip','school_district', 'school_county','school_charter','school_magnet',
                   'school_year_round','school_nlns','school_kipp','school_charter_ready_promise',
                   'teacher_ny_teaching_fellow','vendor_shipping_charges', 'sales_tax','payment_processing_charges','fulfillment_labor_materials','total_price_excluding_optional_support',
                 'eligible_double_your_impact_match','eligible_almost_home_match','date_completed','date_thank_you_packet_mailed',
                'total_donations', 'num_donors', 'teacher_teach_for_america', 'primary_focus_subject',
                                  'secondary_focus_subject','secondary_focus_area', '_projectid',
                             '_teacherid', 'thankyou_note', 'impact_letter', 'title', 'short_description', 'need_statement'], axis = 1)

    projects_dropna = projects.dropna()
    p_noLive = projects_dropna[projects_dropna['funding_status'] != 'live']
    #convert reallocated to expired
    p_noLive['funding_status'] = p_noLive['funding_status'].replace(to_replace = 'reallocated', value = 'expired')
    times_post = pd.DatetimeIndex(p_noLive.date_posted)
    p_noLive['date_posted'] = times_post
    times_expiration = pd.DatetimeIndex(p_noLive.date_expiration)
    p_noLive['date_expiration'] = times_expiration
    p_noLive['duration'] = p_noLive['date_expiration'] - p_noLive['date_posted']
    p_noLive['duration'] = pd.Series(pd.to_timedelta(p_noLive['date_expiration'].values - p_noLive['date_posted'].values).days)
    p_noLive = p_noLive.drop(['date_expiration','date_posted'], axis = 1)
    
    p_noLive = convert(p_noLive)

    split = ['school_metro', 'teacher_prefix','primary_focus_area', 'resource_type', 'poverty_level', 'grade_level']
    for s in split:
        for i in range(len(p_noLive[s].unique())):
            p_noLive[s + str(i)] = p_noLive[s].apply(lambda x:1 if x == i else 0)
    state = pd.read_csv("state_gdp.csv", header = None)
    del state[0]
    state['stgreater'] = state[1].apply(lambda x: 1 if int(x) >= 53000 else 0)
    state['staverage'] = state[1].apply(lambda x: 1 if int(x) < 53000 and int(x) > 45000 else 0)
    state['stlower'] = state[1].apply(lambda x: 1 if int(x) < 45000 else 0)
    del state[1]
    state = state.rename(columns = ({2: 'school_state'}))
    data = pd.merge(p_noLive,state, how='inner', left_on='school_state', right_on = 'school_state')
    totalvocab_tokenized = []
    for i in data['essay']:
        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.append(len(allwords_tokenized))

    vocab = totalvocab_tokenized
    data['vocab'] = totalvocab_tokenized
    data['essaylarge'] = data['vocab'].apply(lambda x: 1 if (x) >= 350 else 0)
    data['essayave'] = data['vocab'].apply(lambda x: 1 if (x) < 350 and (x) > 200 else 0)
    data['essaylow'] = data['vocab'].apply(lambda x: 1 if (x) <= 200 else 0)

    data = data.drop(['school_state', 'school_metro',
           'primary_focus_area', 'resource_type', 'poverty_level',
           'grade_level','teacher_prefix','primary_focus_area','vocab','essay'], axis = 1)
    
    return data


# In[11]:

data = clean_data(projects0, essays0)
data

