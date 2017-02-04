import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt
#import nltk
#import re
from datetime import datetime

essays = pd.read_csv('opendata_essays000', escapechar='\\', names=['_projectid', '_teacherid', 'title', 'short_description', 'need_statement', 'essay', 'thankyou_note', 'impact_letter'],encoding='utf-8')
projects = pd.read_csv('opendata_projects000', escapechar='\\', names=['_projectid', '_teacher_acctid', '_schoolid', 'school_ncesid', 'school_latitude', 'school_longitude', 'school_city', 'school_state', 'school_zip', 'school_metro', 'school_district', 'school_county', 'school_charter', 'school_magnet', 'school_year_round', 'school_nlns', 'school_kipp', 'school_charter_ready_promise', 'teacher_prefix', 'teacher_teach_for_america', 'teacher_ny_teaching_fellow', 'primary_focus_subject', 'primary_focus_area' ,'secondary_focus_subject', 'secondary_focus_area', 'resource_type', 'poverty_level', 'grade_level', 'vendor_shipping_charges', 'sales_tax', 'payment_processing_charges', 'fulfillment_labor_materials', 'total_price_excluding_optional_support', 'total_price_including_optional_support', 'students_reached', 'total_donations', 'num_donors', 'eligible_double_your_impact_match', 'eligible_almost_home_match', 'funding_status', 'date_posted', 'date_completed', 'date_thank_you_packet_mailed', 'date_expiration'],encoding='utf-8')

pdmerge = pd.merge(projects, essays, on = '_projectid')
projects1 = pdmerge.drop(['_teacher_acctid','_schoolid','school_latitude','school_ncesid','school_latitude','school_longitude',
                  'school_city','school_zip','school_district', 'school_county','school_charter','school_magnet',
                   'school_year_round','school_nlns','school_kipp','school_charter_ready_promise',
                   'teacher_ny_teaching_fellow','vendor_shipping_charges', 'sales_tax','payment_processing_charges','fulfillment_labor_materials','total_price_excluding_optional_support',
                'eligible_double_your_impact_match','eligible_almost_home_match','date_completed','date_thank_you_packet_mailed',
                'total_donations', 'num_donors', 'teacher_teach_for_america', 'primary_focus_subject',
                                  'secondary_focus_subject','secondary_focus_area', '_projectid',
                             '_teacherid', 'thankyou_note', 'impact_letter', 'title', 'short_description', 'need_statement'], axis = 1)

projects1_dropna = projects1.dropna()
p_noLive = projects1_dropna[projects1_dropna['funding_status'] != 'live']
p_noLive['funding_status'] = p_noLive['funding_status'].replace(to_replace = 'reallocated', value = 'expired')
times_post = pd.DatetimeIndex(p_noLive.date_posted)
p_noLive['date_posted'] = times_post
times_expiration = pd.DatetimeIndex(p_noLive.date_expiration)
p_noLive['date_expiration'] = times_expiration
p_noLive['duration'] = p_noLive['date_expiration'] - p_noLive['date_posted']
p_noLive['duration'] = pd.to_timedelta(p_noLive['date_expiration'].values - p_noLive['date_posted'].values).days
p_noLive = p_noLive.drop(['date_expiration','date_posted'], axis = 1)

def convert(DataFrame):  #convert categorical columns
    for i in range(len(DataFrame.columns)):   
        if(DataFrame[DataFrame.columns[i]].dtype=='O' and DataFrame.columns[i] != 'school_state' and DataFrame.columns[i] != 'essay'):
            DataFrame[DataFrame.columns[i]]=DataFrame[DataFrame.columns[i]].astype('category')
            length=len(DataFrame[DataFrame.columns[i]].unique())
            DataFrame[DataFrame.columns[i]]=DataFrame[DataFrame.columns[i]].cat.rename_categories([j for j in range(length)])
    return DataFrame

p_noLive = convert(p_noLive)

split = ['school_metro', 'teacher_prefix','primary_focus_area', 'resource_type', 'poverty_level', 'grade_level']
for s in split:
    for i in range(len(p_noLive[s].unique())):
        p_noLive[s + str(i)] = p_noLive[s].apply(lambda x:1 if x == i else 0)
state = pd.read_csv("state_gdp.csv", header = None, thousands=',')
del state[0]
state['stgreater'] = state[1].apply(lambda x: 1 if int(x) >= 53000 else 0)
state['staverage'] = state[1].apply(lambda x: 1 if int(x) < 53000 and int(x) > 45000 else 0)
state['stlower'] = state[1].apply(lambda x: 1 if int(x) < 45000 else 0)
del state[1]
state = state.rename(columns = ({2: 'school_state'}))
data = pd.merge(p_noLive,state, how='inner', left_on='school_state', right_on = 'school_state')

#count essay len
count_word = []
for i in data['essay']:
    count_word.append(sum(1 for w in i.lower().split(' ')))
data['vocab'] = count_word
data['essaylarge'] = data['vocab'].apply(lambda x: 1 if (x) >= 350 else 0)
data['essayave'] = data['vocab'].apply(lambda x: 1 if (x) < 350 and (x) > 200 else 0)
data['essaylow'] = data['vocab'].apply(lambda x: 1 if (x) <= 200 else 0)

data = data.drop(['school_state', 'school_metro',
           'primary_focus_area', 'resource_type', 'poverty_level',
           'grade_level','teacher_prefix','primary_focus_area','vocab','essay'], axis = 1)

data = data.rename(columns = {'school_metro0':'school_metro_rural',
                        'school_metro1':'school_metro_suburban',
                       'school_metro2':'school_metro_urban',
                       'primary_focus_area3':'primary_focus_area_LL',
                       'primary_focus_area4':'primary_focus_area_MS',
                       'primary_focus_area5':'primary_focus_area_MA',
                       'primary_focus_area0':'primary_focus_area_AL',
                       'primary_focus_area6':'primary_focus_area_SN',
                       'primary_focus_area2':'primary_focus_area_HS',
                       'primary_focus_area1':'primary_focus_area_HC',
                       'grade_level3':'grade_level_P2',
                       'grade_level0':'grade_level_35',
                       'grade_level1':'grade_level_68',
                       'grade_level2':'grade_level_912',
                       'teacher_prefix0':'teacher_prefix_Dr',
                       'teacher_prefix1':'teacher_prefix_Mr',
                       'teacher_prefix2':'teacher_prefix_MrMrs',
                       'teacher_prefix3':'teacher_prefix_Mrs',
                       'teacher_prefix4':'teacher_prefix_Ms',
                       'teacher_prefix5':'teacher_prefix_Teacher',
                       'resource_type0':'resource_type_Books',
                       'resource_type1':'resource_type_Other',
                       'resource_type2':'resource_type_Supplies',
                       'resource_type3':'resource_type_Technology',
                       'resource_type4':'resource_type_Trips',
                       'resource_type5':'resource_type_Visitors',
                       'poverty_level0':'poverty_level_High',
                       'poverty_level1':'poverty_level_Highest',
                       'poverty_level2':'poverty_level_Low',
                       'poverty_level3':'poverty_level_Moderate',
                       'essaylarge':'essaylong'})

data.to_csv('data_final.csv',encoding='utf-8')
