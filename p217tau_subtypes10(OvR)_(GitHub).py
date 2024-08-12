# -*- coding: utf-8 -*-

""" notes:
    This is to analyse data for biological PET staging of AD using Janssen 
        pTau217
"""

#%% Set working directory and import/clean data

# Import Libraries-------------------------------------------------------------
import os
os.chdir('/YOUR_DIRECTORY/PythonCode')
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from scipy import stats
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from scipy.stats import iqr
# import scikit_posthocs as sp


# This is our own written functions or Github ones. keep in the same directory
from OurLabCodes import (ROC_plot, Youden_thresh, Sens_thresh, Spec_thresh, 
                        PPV_thresh, NPV_thresh, demog, cohend, ci_bootstraps, 
                        prob_curve_dist, get_asterisks, get_hashtag)
import compare_auc_delong_xu


# Import data------------------------------------------------------------------
MK1 = pd.read_csv('Blood_dataset.csv') # this is MK1 dataset
MK2 = pd.read_csv('intermediate_dataset_2.csv') # this is MK2 dataset
ATN = pd.read_csv('MK6240_ATN_all_CPU.csv') # retrieve MK6240 data here
Missing_APOE = pd.read_csv('22092023_ Missing_APOE.csv') # retrieve missing APOE here
ATN_ADNET = pd.read_csv("MK6240_ATN_full_19Jul2024.csv") # retrieve missing MMSE here  


# Clean data-------------------------------------------------------------------
# MK1 cohort
MK1.columns = (MK1.columns.str.replace(' ', '_', regex=True)
                                .str.replace('\.', '_', regex=True)
                                .str.replace('\(','', regex=True)
                                .str.replace('\)','', regex=True)
                                .str.replace('-','_', regex=True)
                                .str.replace('+','PLUS', regex=True)
                                .str.replace("/","", regex=True))
MK1.APOE = (MK1.APOE.str.replace("/","_", regex=True))
MK1 = MK1.loc[:,~MK1.columns.duplicated()]
MK1 = MK1.rename(columns={'plasma_p217PLUStau_fgml__avg': 'pTau_217'})


# MK2 cohort
MK2.columns = (MK2.columns.str.replace(' ', '_', regex=True)
                                .str.replace('\.', '_', regex=True)
                                .str.replace('\(','', regex=True)
                                .str.replace('\)','', regex=True)
                                .str.replace('-','_', regex=True)
                                .str.replace('+','PLUS', regex=True)
                                .str.replace("/","", regex=True))
MK2.APOE = (MK2.APOE.str.replace("/","_", regex=True))
MK2 = MK2.loc[:,~MK2.columns.duplicated()]
MK2.pTau_217 = MK2.pTau_217*1000 # converting pg/ml to fg/ml
MK2 = MK2.rename(columns={'Blood_Processing_Processing_Date': 'Blood_date'})
"""In MK2 drop duplicate rows for when blood assay is repeated multiple times due to having had multiple aliquots for the same participants
# MK1 has unique AIBL IDs so this is not a problem for MK1"""
"""Comment on some p217+tau assays: "came as 2 aliquots, tested both in p217 assay mistakenly. See NUPI440437"""
MK2 = MK2.groupby(['AIBL_ID', 'Coll_#']).first()\
                  .reset_index()  \
                  .rename_axis(None, axis=1)

# ATN
ATN.columns = (ATN.columns.str.replace(' ', '_', regex=True)
                          .str.replace('\.', '_', regex=True))
ATN_ADNET.columns = (ATN_ADNET.columns.str.replace(' ', '_', regex=True)
                              .str.replace('\.', '_', regex=True))

# Missing_APOE
Missing_APOE.columns = Missing_APOE.columns.str.replace(' ', '', regex=True)
Missing_APOE.APOE = (Missing_APOE.APOE.str.replace("/","_", regex=True))



# Find uncommon ID's in MK1 and add to MK2
Only_in_MK1 = MK1[~MK1["AIBL_ID"].isin(MK2["AIBL_ID"])]


MK2 = MK2[['AIBL_ID', 'AIBL_TP', 'Collection', 'Gender', 'APOE', 'YoE', 'pTau_217', 
          'Blood_date']]
Only_in_MK1 = Only_in_MK1[['AIBL_ID', 'AIBL_TP', 'Collection', 'Gender', 'APOE', 
                           'YoE', 'pTau_217', 'Blood_date']]
BloodData = MK2.append(Only_in_MK1)


# Previously reported Thresholds-----------------------------------------------
Estab_thresh_pT = 126.7
Estab_thresh_A = 25
Estab_thresh_MetaT = 1.19 
Estab_thresh_Me = 1.18
Estab_thresh_Te = 1.24
Estab_thresh_Te2 = 2.68
Estab_thresh_R = 1.08


# Add missing APOE-------------------------------------------------------------
BloodData = BloodData.merge(Missing_APOE,  
                            on=["AIBL_ID"], 
                            how='left', suffixes=('', '_Missing'))
BloodData['APOE'] = BloodData['APOE'].fillna(BloodData['APOE_Missing'])
BloodData.drop(columns='APOE_Missing', inplace=True)


# Add tau, CL, Dx, cog scores in from ATN--------------------------------------
BloodData = BloodData.merge(ATN[["AIBL_ID", "AIBL_TP", "Tau_Age", "Centiloid", "Ab_Acquisition_Date",
                                 "Tau_Me_SUVR", "Tau_Te_SUVR", "Tau_R_SUVR", "Tau_Acquisition_Date",
                                 "Tau_Meta_Temp_SUVR", "ADNeT_Diag", "Diagnosis", 
                                 "Composite_memory", "MMSE", "CDR_SoB"]],  
                            on=["AIBL_ID", "AIBL_TP"], 
                            how='left')

# Fix Dx missing cells --------------------------------------------------------
# For diagnosis, first check "ADNeT_Diag", if nan, grab Dx from Diagnosis which is AIBL Dx
BloodData.ADNeT_Diag[BloodData.ADNeT_Diag.isnull()] = BloodData.Diagnosis


# Some MK2 IDs have missing Dx. Lucy: "All of the IDs you sent me have a classification of MCI"
Miss_ID = pd.DataFrame({'AIBL_ID': [2900, 2946, 3005, 3035, 3129, 3168, 3177]})
for i in Miss_ID.AIBL_ID:
        BloodData.loc[BloodData.AIBL_ID == i, 'ADNeT_Diag'] = "MCI"


#  Remove rows with na in ADNeT_Diag otherwise np.where assigns "CN" to nan!!! 
BloodData = BloodData[BloodData.ADNeT_Diag.notna()]

BloodData["simple_Diag"] = np.where(BloodData["ADNeT_Diag"].str.contains("Healthy"), "HC",
                                    np.where(BloodData["ADNeT_Diag"].str.contains("Normal"), "HC",
                                             np.where(BloodData["ADNeT_Diag"].str.contains("MCI"), "MCI",
                                                      np.where(BloodData["ADNeT_Diag"].str.contains("AD"), "AD",
                                                               pd.NA))))

# Some IDs were flagged as "Excluded" under ADNeT_Diag. It is ok to add them 
# back for the purpose of this analysis
""" the following data for Excluded participants is from Chris F
email subject: [RE: Questions about data received from Janssen]"""
Ex_IDs = pd.DataFrame({'AIBL_ID': [2629, 2683, 2705, 2786, 2797, 2862, 2865, 2869],
                       'simple_Diag': ["MCI", "HC", "HC", "MCI", "MCI", "MCI", "MCI", "HC"]})
for i in Ex_IDs.AIBL_ID:
    temp = BloodData.loc[BloodData.AIBL_ID == i, 'simple_Diag']
    if len(temp) == 1:
        temp.iloc[0] = Ex_IDs.loc[Ex_IDs.AIBL_ID == i, 'simple_Diag'].values[0]
        BloodData.loc[BloodData.AIBL_ID == i, 'simple_Diag'] = temp.iloc[0]
    elif len(temp) > 1:
        warnings.warn("This Participants has 2 or more points. Adjust code") 
                

# Add missing MMSE-------------------------------------------------------------
BloodData.Tau_Acquisition_Date = pd.to_datetime(BloodData.Tau_Acquisition_Date)
ATN_ADNET.Tau_Acquisition_Date = pd.to_datetime(ATN_ADNET.Tau_Acquisition_Date)
BloodData = BloodData.merge(ATN_ADNET[["AIBL_ID", "Tau_Acquisition_Date", "MMSE"]],  
                            on=["AIBL_ID", "Tau_Acquisition_Date"], 
                            how='left', suffixes=('', '_Missing'))
BloodData['MMSE'] = BloodData['MMSE'].fillna(BloodData['MMSE_Missing'])
BloodData.drop(columns='MMSE_Missing', inplace=True)


# Only include AD continuum as there are other diseases too
Data = BloodData[BloodData.simple_Diag.isin(["HC", "MCI", "AD"])]


# Add AT profiles--------------------------------------------------------------
Data['Profile_new'] = np.where((Data['Centiloid']<Estab_thresh_A) & (Data['Tau_Me_SUVR']< Estab_thresh_Me) & (Data['Tau_Te_SUVR']< Estab_thresh_Te) & (Data['Tau_R_SUVR']< Estab_thresh_R), "A-Me-Te-R-",
                      np.where((Data['Centiloid']<Estab_thresh_A) & ((Data['Tau_Me_SUVR']>= Estab_thresh_Me) & (Data['Tau_Te_SUVR']< Estab_thresh_Te) & (Data['Tau_R_SUVR']< Estab_thresh_R)), "A-Me+Te-R-",
                      np.where((Data['Centiloid']<Estab_thresh_A) & (Data['Tau_Me_SUVR']< Estab_thresh_Me) & (Data['Tau_Te_SUVR']>= Estab_thresh_Te) & (Data['Tau_R_SUVR']< Estab_thresh_R), "A-Me-Te+R-",
                      np.where((Data['Centiloid']<Estab_thresh_A) & (Data['Tau_Me_SUVR']< Estab_thresh_Me) & (Data['Tau_Te_SUVR']< Estab_thresh_Te) & (Data['Tau_R_SUVR']>= Estab_thresh_R), "A-Me-Te-R+",
                      np.where((Data['Centiloid']<Estab_thresh_A) & ((Data['Tau_Me_SUVR']>= Estab_thresh_Me) & (Data['Tau_Te_SUVR']>= Estab_thresh_Te) & (Data['Tau_R_SUVR']< Estab_thresh_R)), "A-Me+Te+R-",
                      np.where((Data['Centiloid']<Estab_thresh_A) & ((Data['Tau_Me_SUVR']< Estab_thresh_Me) & (Data['Tau_Te_SUVR']>= Estab_thresh_Te) & (Data['Tau_R_SUVR']>= Estab_thresh_R)), "A-Me-Te+R+",
                      np.where((Data['Centiloid']<Estab_thresh_A) & ((Data['Tau_Me_SUVR']>= Estab_thresh_Me) & (Data['Tau_Te_SUVR']>= Estab_thresh_Te) & (Data['Tau_R_SUVR']>= Estab_thresh_R)), "A-Me+Te+R+",
                      np.where((Data['Centiloid']>=Estab_thresh_A) & (Data['Tau_Me_SUVR']< Estab_thresh_Me) & (Data['Tau_Te_SUVR']< Estab_thresh_Te) & (Data['Tau_R_SUVR']< Estab_thresh_R), "A+Me-Te-R-",
                      np.where((Data['Centiloid']>=Estab_thresh_A) & (Data['Tau_Me_SUVR']>= Estab_thresh_Me) & (Data['Tau_Te_SUVR']< Estab_thresh_Te) & (Data['Tau_R_SUVR']< Estab_thresh_R), "A+Me+Te-R-",
                      np.where((Data['Centiloid']>=Estab_thresh_A) & (Data['Tau_Me_SUVR']>= Estab_thresh_Me) & (Data['Tau_Te_SUVR']>= Estab_thresh_Te) & (Data['Tau_R_SUVR']< Estab_thresh_R), "A+Me+Te+R-",
                      np.where((Data['Centiloid']>=Estab_thresh_A) & (Data['Tau_Me_SUVR']< Estab_thresh_Me) & (Data['Tau_Te_SUVR']>= Estab_thresh_Te) & (Data['Tau_R_SUVR']< Estab_thresh_R), "A+Me-Te+R-",
                      np.where((Data['Centiloid']>=Estab_thresh_A) & (Data['Tau_Me_SUVR']>= Estab_thresh_Me) & (Data['Tau_Te_SUVR']>= Estab_thresh_Te) & (Data['Tau_R_SUVR']>= Estab_thresh_R), "A+Me+Te+R+",
                      np.where((Data['Centiloid']>=Estab_thresh_A) & (Data['Tau_Me_SUVR']>= Estab_thresh_Me) & (Data['Tau_Te_SUVR']< Estab_thresh_Te) & (Data['Tau_R_SUVR']>= Estab_thresh_R), "A+Me+Te-R+",
                      np.where((Data['Centiloid']>=Estab_thresh_A) & (Data['Tau_Me_SUVR']< Estab_thresh_Me) & (Data['Tau_Te_SUVR']>= Estab_thresh_Te) & (Data['Tau_R_SUVR']>= Estab_thresh_R), "A+Me-Te+R+",
                      np.where((Data['Centiloid']>=Estab_thresh_A) & (Data['Tau_Me_SUVR']< Estab_thresh_Me) & (Data['Tau_Te_SUVR']< Estab_thresh_Te) & (Data['Tau_R_SUVR']>= Estab_thresh_R), "A+Me-Te-R+",
                      pd.NA)))))))))))))))

                              
# Add biological PET stages----------------------------------------------------                   
Data['Profile_newer'] = np.where((Data['Centiloid']<Estab_thresh_A) & (Data['Tau_Me_SUVR']< Estab_thresh_Me) & (Data['Tau_Te_SUVR']< Estab_thresh_Te) & (Data['Tau_R_SUVR']< Estab_thresh_R), "A-T-",
                      np.where((Data['Centiloid']>=Estab_thresh_A) & (Data['Tau_Me_SUVR']< Estab_thresh_Me) & (Data['Tau_Te_SUVR']< Estab_thresh_Te) & (Data['Tau_R_SUVR']< Estab_thresh_R), "A+T-",
                      np.where((Data['Centiloid']>=Estab_thresh_A) & (Data['Tau_Me_SUVR']>= Estab_thresh_Me) & (Data['Tau_Te_SUVR']< Estab_thresh_Te) & (Data['Tau_R_SUVR']< Estab_thresh_R), "A+MTL",
                      np.where((Data['Centiloid']>=Estab_thresh_A) & (Data['Tau_Te_SUVR'].between(Estab_thresh_Te, Estab_thresh_Te2)), "A+MOD",
                      np.where((Data['Centiloid']>=Estab_thresh_A) & (Data['Tau_Te_SUVR']> Estab_thresh_Te2), "A+HIGH",
                      pd.NA)))))


# Manual overwrite of 3 A+ nans based on visual read:
manual_ID = pd.DataFrame({'AIBL_ID': [2703, 3080, 3213],
                       'Profile_newer': ["A+MTL", "A+HIGH", "A+T-"]})
for i in manual_ID.AIBL_ID:
    temp = Data.loc[Data.AIBL_ID == i, 'Profile_newer']
    if len(temp) == 1:
        temp.iloc[0] = manual_ID.loc[manual_ID.AIBL_ID == i, 'Profile_newer'].values[0]
        Data.loc[Data.AIBL_ID == i, 'Profile_newer'] = temp.iloc[0]
    elif len(temp) > 1:
        warnings.warn("This Participants has 2 or more points. Adjust code")  
    
    
# Final data-------------------------------------------------------------------
# ALL
Data = Data[Data.simple_Diag.isin(["MCI", "AD", "HC"])]
# 

# Data are available from multiple Coll_#'s. Only include one data point per AIBL_ID.
Data = Data.groupby(['AIBL_ID']).last()\
    .reset_index()  \
    .rename_axis(None, axis=1)     
# Data.to_csv("Data.csv")



#%% Show between-laboratory variation between common IDs' pTau value from MK1 vs MK2
common = MK1[MK1["AIBL_ID"].isin(MK2["AIBL_ID"])]
common = common.rename(columns={'pTau_217' : 'pTau_217_MK1'})
common2 = common.merge(MK2, on = ["AIBL_ID", "AIBL_TP"], how = "left").dropna(subset=['pTau_217_MK1', 'pTau_217'])
plt.scatter(common2.pTau_217_MK1, common2.pTau_217, color = "purple", alpha=0.3)
plt.xlabel("Janssen R&D (cohort 1)")
plt.ylabel("Quanterix Corp (cohort 2)")
plt.plot([0, 700], [0, 700], color='gray', linestyle='--')
plt.annotate(("cor: " + str(round(stats.spearmanr(common2.pTau_217_MK1, common2.pTau_217)[0], 2))), xy=(0.6, 0.80), xycoords='axes fraction', fontsize=14)
plt.title("Same plasma samples, tested in two labs")
# plt.savefig('/coh1_coh2.PNG', dpi=300, bbox_inches='tight')



#%% Difference between biomarker acquisition dates
Data.Blood_date = pd.to_datetime(Data.Blood_date)
Data.Ab_Acquisition_Date = pd.to_datetime(Data.Ab_Acquisition_Date)
Data.Tau_Acquisition_Date = pd.to_datetime(Data.Tau_Acquisition_Date)

Data["tau_minus_blood"] = Data.Tau_Acquisition_Date - Data.Blood_date
round(Data["tau_minus_blood"].mean(numeric_only=False).days/30, 1)
round(Data["tau_minus_blood"].std(numeric_only=False).days/30, 1)

Data["AB_minus_blood"] = (Data.Ab_Acquisition_Date - Data.Blood_date)
round(Data["AB_minus_blood"].mean(numeric_only=False).days/30, 1)
round(Data["AB_minus_blood"].std(numeric_only=False).days/30, 1)



#%% Supp Table 1: Baseline Demographics for the ones that did not fit NIA-AA categories

# Including only atypical biological PET stages (assigned na)
Med = Data[~Data["Profile_newer"].notna()].reset_index(drop=True)
Med["Clinical_stage"] = np.where(Med["simple_Diag"].isin(["MCI", "AD"]), 1, 0)

one = Med[Med['Profile_new'] == 'A-Me+Te-R-']
one.name= 'A-Me+Te-R-'

two = Med[Med['Profile_new'] == 'A-Me-Te+R-']
two.name= 'A-Me-Te+R-'

three = Med[Med['Profile_new'] == 'A-Me-Te-R+']
three.name= 'A-Me-Te-R+'

four = Med[Med['Profile_new'] == 'A-Me-Te+R+']
four.name= 'A-Me-Te+R+'

five = Med[Med['Profile_new'] == 'A-Me+Te+R-']
five.name= 'A-Me+Te+R-'

six = Med[Med['Profile_new'] == 'A-Me+Te+R+']
six.name= 'A-Me+Te+R+'

mylist = [one, two, three, four, five, six]
row_names = ["Number", "Tau_Age", "YoE", "Gender", "MMSE", "CDR_SoB", "Tau_Meta_Temp_SUVR", "Centiloid",  "pTau_217", "Clinical_stage"]

SuppTable1 = demog(mylist, row_names)
# SuppTable1.to_csv("/SuppT1_Atypicals.csv")



#%% Supp Table 2: Baseline Demographics CU vs MCI vs AD

# excluding atypical biological PET stages (assigned na)
Med = Data[Data["Profile_newer"].notna()].reset_index(drop=True)
Med["Centiloid_status"] = np.where(Med["Centiloid"] > Estab_thresh_A, 1, 0)

CU = Med[Med.simple_Diag == 'HC']
CU.name= 'CU'
MCI = Med[Med.simple_Diag == 'MCI']
MCI.name= 'MCI'
AD = Med[Med.simple_Diag == 'AD']
AD.name= 'AD'

mylist = [CU, MCI, AD]
row_names = ["Number", "Tau_Age", "YoE", "Gender", "APOE", "MMSE", "CDR_SoB", "Tau_Meta_Temp_SUVR", "Centiloid",  "Centiloid_status", "pTau_217", "Profile_newer"]

SuppTable2 = demog(mylist, row_names)
# SuppTable2.to_csv("/SuppT2_CUMCIAD.csv")



#%% Main Table 2: Baseline Demographics: biological PET stages

# excluding atypical biological PET stages (assigned na)
Med = Data[Data["Profile_newer"].notna()].reset_index(drop=True)
Med["Clinical_stage"] = np.where(Med["simple_Diag"].isin(["MCI", "AD"]), 1, 0)

AnTn = Med[Med.Profile_newer == 'A-T-']
AnTn.name = 'A-T-'

ApTn = Med[Med.Profile_newer == 'A+T-']
ApTn.name = 'A+T-'

MTL = Med[Med.Profile_newer == 'A+MTL']
MTL.name = 'A+MTL'

MOD = Med[Med.Profile_newer == 'A+MOD']
MOD.name = 'A+MOD'

HIGH = Med[Med.Profile_newer == 'A+HIGH']
HIGH.name = 'A+HIGH'

mylist = [AnTn, ApTn, MTL, MOD, HIGH]
row_names = ["Number", "Tau_Age", "YoE", "Gender", "APOE", "MMSE", "CDR_SoB", "Centiloid", "Tau_Me_SUVR", "Tau_Te_SUVR", "pTau_217", "Clinical_stage"]

MainTable2 = demog(mylist, row_names)

# Stats
### Data type conversions
# APOE
Med.APOE = Med.APOE.replace('E3_E3', 0)
Med.APOE = Med.APOE.replace('E3_E2', 0)
Med.APOE = Med.APOE.replace('E2_E2', 0)
Med.APOE = Med.APOE.replace('E4_E2', 1)
Med.APOE = Med.APOE.replace('E4_E3', 1)
Med.APOE = Med.APOE.replace('E3_E4', 1)
Med.APOE = Med.APOE.replace('E4_E4', 1)

Med.simple_Diag = Med.simple_Diag.replace('MCI', 'CI')
Med.simple_Diag = Med.simple_Diag.replace('AD', 'CI')


# 2 by 2 MainTable2 of contingency for chi square test
CrossTab_Gender = pd.crosstab(Med.Gender, Med.Profile_newer)
CrossTab_APOE = pd.crosstab(Med.APOE, Med.Profile_newer)
CrossTab_Dx = pd.crosstab(Med.simple_Diag, Med.Profile_newer)


### MainTable2 p values
MainTable2["p"] = pd.NA

# Age
model_ols = ols('Tau_Age ~ Profile_newer', data=Med).fit() 
model_ols.resid.skew()
plt.hist(model_ols.resid)
""" we consider the residuals normally distributed so anova is ok"""
stats_Table = sm.stats.anova_lm(model_ols, typ=2)
MainTable2["p"][1] = round(stats_Table['PR(>F)']['Profile_newer'], 3)

# YoE with nans
model_ols = ols('YoE ~ Profile_newer', data=Med).fit()
model_ols.resid.skew()
plt.hist(model_ols.resid)
""" the residuals are non-normally distributed so kruskal wallis is better"""
stats_Table = stats.kruskal(AnTn.YoE.dropna(), ApTn.YoE.dropna(), MTL.YoE.dropna(), MOD.YoE.dropna(), HIGH.YoE.dropna())
MainTable2["p"][2] = stats_Table.pvalue


# Gender
MainTable2["p"][3] = round(stats.chi2_contingency(CrossTab_Gender, correction=False)[1], 3) # dichotomous categorical: Pearson's chi-squared test for independence without Yates correction
"""only turn on Yate's correction for small (n<=5) samples"""

# APOE
MainTable2["p"][4] = round(stats.chi2_contingency(CrossTab_APOE, correction=True)[1], 3) # categorical: Pearson's chi-squared test for independence without Yates correction
""" turned on Yale's correction as some cell frequencies are 0 or below 10 but this didn't change p value"""

# MMSE with nans
model_ols = ols('MMSE ~ Profile_newer', data=Med).fit() 
model_ols.resid.skew()
plt.hist(model_ols.resid)
""" the residuals are non-normally distributed so kruskal wallis is better"""
stats_Table = stats.kruskal(AnTn.MMSE.dropna(), ApTn.MMSE.dropna(), MTL.MMSE.dropna(), MOD.MMSE.dropna(), HIGH.MMSE.dropna())
MainTable2["p"][5] = stats_Table.pvalue

# CDR-SoB with nans
model_ols = ols('CDR_SoB ~ Profile_newer', data=Med).fit() 
model_ols.resid.skew()
plt.hist(model_ols.resid)
""" the residuals are non-normally distributed so kruskal wallis is better"""
stats_Table = stats.kruskal(AnTn.CDR_SoB.dropna(), ApTn.CDR_SoB.dropna(), MTL.CDR_SoB.dropna(), MOD.CDR_SoB.dropna(), HIGH.CDR_SoB.dropna())
MainTable2["p"][6] = stats_Table.pvalue

# CL
model_ols = ols('Centiloid ~ Profile_newer', data=Med).fit() 
model_ols.resid.skew()
plt.hist(model_ols.resid)
""" we consider the residuals normally distributed so anova is ok"""
stats_Table = sm.stats.anova_lm(model_ols, typ=2)
MainTable2["p"][7] = round(stats_Table['PR(>F)']['Profile_newer'], 3)

# Me
model_ols = ols('Tau_Me_SUVR ~ Profile_newer', data=Med).fit() 
model_ols.resid.skew()
plt.hist(model_ols.resid)
""" we consider the residuals normally distributed so anova is ok"""
stats_Table = sm.stats.anova_lm(model_ols, typ=2)
MainTable2["p"][8] = round(stats_Table['PR(>F)']['Profile_newer'], 3)

# Te
model_ols = ols('Tau_Te_SUVR ~ Profile_newer', data=Med).fit() 
model_ols.resid.skew()
plt.hist(model_ols.resid)
""" we consider the residuals normally distributed so anova is ok"""
stats_Table = sm.stats.anova_lm(model_ols, typ=2)
MainTable2["p"][9] = round(stats_Table['PR(>F)']['Profile_newer'], 3)

# pTau
model_ols = ols('pTau_217 ~ Profile_newer', data=Med).fit() 
model_ols.resid.skew()
plt.hist(model_ols.resid)
""" the residuals seem non-normally distributed so kruskal wallis is better"""
stats_Table = stats.kruskal(AnTn.pTau_217.dropna(), ApTn.pTau_217.dropna(), MTL.pTau_217.dropna(), MOD.pTau_217.dropna(), HIGH.pTau_217.dropna())
MainTable2["p"][10] = stats_Table.pvalue

# Diagnosis
MainTable2["p"][11] = round(stats.chi2_contingency(CrossTab_Dx, correction=False)[1], 3) # dichotomous categorical: Pearson's chi-squared test for independence without Yates correction

# MainTable2.to_csv("/T2_BiolPET.csv")



#%% Main Table 2 post-hoc pairwise comparisons

# Age
"""normal residual, we do Tukey HSD"""
print("Age: \n" + str(pairwise_tukeyhsd(Med['Tau_Age'], Med['Profile_newer'], alpha=0.05)))

# YoE with nans
"non-normal residual, we do Dunn's test"
sub = Med[Med.YoE.notna()]
print("YOE: \n" + str(sp.posthoc_dunn(Med, val_col="YoE", group_col="Profile_newer", p_adjust='holm')))

# Gender
""" comparison of 5 groups was ns, so need to do pair-wise comaprison"""

# APOE
# comparsion 1 ----
sub = []
CrossTab =[]
sub = Med[(Med.Profile_newer=="A-T-") | (Med.Profile_newer=="A+T-")]
CrossTab = pd.crosstab(sub.APOE, sub.Profile_newer)
print("APOE A-T- vs A+T- \n" + str((stats.chi2_contingency(CrossTab, correction=False)[1])*4)) # 4 is the number of comparisons we're repeating across the 5 groups. multiplication by 4 is for Bonferroni correction
# comparison 2 ----
sub = []
CrossTab =[]
sub = Med[(Med.Profile_newer=="A-T-") | (Med.Profile_newer=="A+MTL")]
CrossTab = pd.crosstab(sub.APOE, sub.Profile_newer)
print("APOE A-T- vs A+MTL \n" + str((stats.chi2_contingency(CrossTab, correction=False)[1])*4)) 
# comparison 3 ----
sub = []
CrossTab =[]
sub = Med[(Med.Profile_newer=="A-T-") | (Med.Profile_newer=="A+MOD")]
CrossTab = pd.crosstab(sub.APOE, sub.Profile_newer)
print("APOE A-T- vs A+MOD \n" + str((stats.chi2_contingency(CrossTab, correction=False)[1])*4)) 
# comparison 4 ----
sub = []
CrossTab =[]
sub = Med[(Med.Profile_newer=="A-T-") | (Med.Profile_newer=="A+HIGH")]
CrossTab = pd.crosstab(sub.APOE, sub.Profile_newer)
print("APOE A-T- vs A+HIGH \n" + str((stats.chi2_contingency(CrossTab, correction=False)[1])*4))

# MMSE with nans
"non-normal residual, we do Dunn's test"
print("MMSE: \n" + str(round(sp.posthoc_dunn(Med, val_col="MMSE", group_col="Profile_newer", p_adjust='holm'), 3)))

# CRD with nans
"non-normal residual, we do Dunn's test"
print("CDR: \n" + str(round(sp.posthoc_dunn(Med, val_col="CDR_SoB", group_col="Profile_newer", p_adjust='holm'), 3)))

# CL
"normal residual, we do Tukey HSD"
print("Centiloid: \n" + str(pairwise_tukeyhsd(Med['Centiloid'], Med['Profile_newer'], alpha=0.05)))

# Me tau
"normal residual, we do Tukey HSD"
print("Me tau: \n" + str(pairwise_tukeyhsd(Med['Tau_Me_SUVR'], Med['Profile_newer'], alpha=0.05)))

# Te tau
"normal residual, we do Tukey HSD"
print("Te tau: \n" + str(pairwise_tukeyhsd(Med['Tau_Te_SUVR'], Med['Profile_newer'], alpha=0.05)))

# pTau
"non-normal residual, we do Dunn's test"
print("P217: \n" + str(round(sp.posthoc_dunn(Med, val_col="pTau_217", group_col="Profile_newer", p_adjust='holm'), 3)))

# Diagnosis
# Comparison 1 ----
sub = []
CrossTab =[]
sub = Med[(Med.Profile_newer=="A-T-") | (Med.Profile_newer=="A+T-")]
CrossTab = pd.crosstab(sub.simple_Diag, sub.Profile_newer)
print("Dx A-T- vs A+T- \n" + str((stats.chi2_contingency(CrossTab, correction=False)[1])*4)) 

# Comparison 2 ----
sub = []
CrossTab =[]
sub = Med[(Med.Profile_newer=="A-T-") | (Med.Profile_newer=="A+MTL")]
CrossTab = pd.crosstab(sub.simple_Diag, sub.Profile_newer)
print("Dx A-T- vs A+MTL \n" + str((stats.chi2_contingency(CrossTab, correction=False)[1])*4)) 

# Comparison 3 ----
sub = []
CrossTab =[]
sub = Med[(Med.Profile_newer=="A-T-") | (Med.Profile_newer=="A+MOD")]
CrossTab = pd.crosstab(sub.simple_Diag, sub.Profile_newer)
print("Dx A-T- vs A+MOD \n" + str((stats.chi2_contingency(CrossTab, correction=False)[1])*4)) 

# Comparison 4 ----
sub = []
CrossTab =[]
sub = Med[(Med.Profile_newer=="A-T-") | (Med.Profile_newer=="A+HIGH")]
CrossTab = pd.crosstab(sub.simple_Diag, sub.Profile_newer)
print("Dx A-T- vs A+HIGH \n" + str((stats.chi2_contingency(CrossTab, correction=False)[1])*4 ))



#%% Figure 1: Box plot

# excluding atypical biological PET stages (assigned na)
Med = Data[Data["Profile_newer"].notna()].reset_index(drop=True)

# colors
palette = {'A-T-': "lightblue", 'A+T-':"#0ABAB5", 'A+MTL':"orange", 'A+MOD':"#ff4c4c", 'A+HIGH':"#cc0000"}

# box plot
plt.figure()
sns.set_context("talk", font_scale=1.6)
ax = sns.catplot(x="Profile_newer", y="pTau_217", kind='box', height = 10, 
                     data=Med, order=["A-T-", 'A+T-', 'A+MTL', 'A+MOD', 'A+HIGH'],
                     color = 'white')
    
ax = sns.swarmplot(x="Profile_newer", y="pTau_217", data=Med, color=".25",
                    order=["A-T-", 'A+T-', 'A+MTL', 'A+MOD', 'A+HIGH'],
                    palette= palette,
                    size =9,
                    alpha = 0.9)
ax.set_xticklabels([r'A-T-', r'A+T-', r'A+T$_{\mathrm{MTL}}$+', r'A+T$_{\mathrm{MOD}}$+', r'A+T$_{\mathrm{HIGH}}$+'], fontsize=25)
ax.set_xticklabels(ax.get_xticklabels(),rotation=25)
plt.ylabel("Plasma p217+tau (fg/ml)")
ax.set_yticks([0, 200, 400, 600, 800, 1000])
ax.set_yticklabels(("0", "200", "400", 
                        "600", "800", "1000"))
# plt.savefig('/boxplot_BlankBox.PNG', dpi=300)
# plt.savefig('/boxplot_BlankBox.PNG', dpi=300, bbox_inches='tight')


# Concentrations
print(Med \
      .groupby("Profile_newer")["pTau_217"] \
          .agg([np.median, lambda x: np.nanpercentile(x, 25), lambda x: np.nanpercentile(x, 75)]) \
              .round(1))
    
    

#%% Supp Table 3, cohen's d and median fold change
# Pairwise comparison of pT between groups 
print("P217: \n" + str(round(sp.posthoc_dunn(Med, val_col="pTau_217", group_col="Profile_newer", p_adjust='holm'), 3)))

# Cohen's d between groups
boxes = ["A-T-", "A+T-", "A+MTL", "A+MOD", "A+HIGH"]
pairwise_comparisons = []
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        group1 = boxes[i]
        group2 = boxes[j]
        comparison_result = f"{group1} vs {group2}: " + str(round(cohend(Med[Med.Profile_newer== group1].pTau_217, Med[Med.Profile_newer == group2].pTau_217), 1))
        pairwise_comparisons.append(comparison_result)
        
for comparison in pairwise_comparisons:
    print(comparison)
    

# Median fold chnage between groups
boxes = ["A-T-", "A+T-", "A+MTL", "A+MOD", "A+HIGH"]
pairwise_comparisons = []
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        group1 = boxes[i]
        group2 = boxes[j]
        # Fetch median values directly
        median_group1 = Med.loc[Med.Profile_newer == group1, 'pTau_217'].median()
        median_group2 = Med.loc[Med.Profile_newer == group2, 'pTau_217'].median()
        # Calculate fold change
        fold_change = median_group2 / median_group1
        comparison_result = f"{group1} vs {group2}: {fold_change:.1f}"
        pairwise_comparisons.append(comparison_result)
        
for comparison in pairwise_comparisons:
    print(comparison)

     
    
#%% Supp Table 4: ROC All
# Youden T, sens, spec, ppv, npv and 95% sens T, 95% spec T
Med = []
Med = Data[Data["Profile_newer"].notna()].reset_index(drop=True)


# Binarize the profiles
Med['0vs1_4'] = np.where(Med["Profile_newer"]=="A-T-", 0, 1)
Med['02vs34']  = np.where((Med["Profile_newer"]=="A-T-") | (Med["Profile_newer"]=="A+T-") | (Med["Profile_newer"]=="A+MTL"), 0, 1)
Med['0_3vs4'] = np.where(Med["Profile_newer"]=="A+HIGH", 1, 0)


Comp1 = pd.DataFrame(ci_bootstraps(Med['pTau_217'], Med['0vs1_4'], alpha=0.95, n_bootstraps=1000, rng_seed=42)).T
Comp1.rename(columns={0:"AUC", 1:"Youden", 2:"Sensitivity", 3:"Specificity", 4:"PPV", 5:"NPV"}, inplace=True)


Comp2 = pd.DataFrame(ci_bootstraps(Med['pTau_217'], Med['02vs34'], alpha=0.95, n_bootstraps=1000, rng_seed=42)).T
Comp2.rename(columns={0:"AUC", 1:"Youden", 2:"Sensitivity", 3:"Specificity", 4:"PPV", 5:"NPV"}, inplace=True)


Comp3 = pd.DataFrame(ci_bootstraps(Med['pTau_217'], Med['0_3vs4'], alpha=0.95, n_bootstraps=1000, rng_seed=42)).T
Comp3.rename(columns={0:"AUC", 1:"Youden", 2:"Sensitivity", 3:"Specificity", 4:"PPV", 5:"NPV"}, inplace=True)


Comp = pd.concat([Comp1, Comp2, Comp3])
# Comp.to_csv("/ROC_Youden_all.csv")



#%% Comp1 if we don't exclude the 31 A-T+ (for supplementary)
Med = []
Med = Data

# Binarize the profiles
Med['0vs1_4'] = np.where(Med["Centiloid"]<25, 0, 1)

Comp1a = pd.DataFrame(ci_bootstraps(Med['pTau_217'], Med['0vs1_4'], alpha=0.95, n_bootstraps=1000, rng_seed=42)).T
Comp1a.rename(columns={0:"AUC", 1:"Youden", 2:"Sensitivity", 3:"Specificity", 4:"PPV", 5:"NPV"}, inplace=True)



#%% Supp Table 5: ROC CI only
Med = []
Med = Data[Data["Profile_newer"].notna()].reset_index(drop=True)
Med = Med[Med.simple_Diag.isin(["MCI", "AD"])].reset_index(drop=True)


# Binarize the profiles
Med['0vs1_4'] = np.where(Med["Profile_newer"]=="A-T-", 0, 1)
Med['02vs34']  = np.where((Med["Profile_newer"]=="A-T-") | (Med["Profile_newer"]=="A+T-") | (Med["Profile_newer"]=="A+MTL"), 0, 1)
Med['0_3vs4'] = np.where(Med["Profile_newer"]=="A+HIGH", 1, 0)


Comp1 = pd.DataFrame(ci_bootstraps(Med['pTau_217'], Med['0vs1_4'], alpha=0.95, n_bootstraps=1000, rng_seed=42)).T
Comp1.rename(columns={0:"AUC", 1:"Youden", 2:"Sensitivity", 3:"Specificity", 4:"PPV", 5:"NPV"}, inplace=True)


Comp2 = pd.DataFrame(ci_bootstraps(Med['pTau_217'], Med['02vs34'], alpha=0.95, n_bootstraps=1000, rng_seed=42)).T
Comp2.rename(columns={0:"AUC", 1:"Youden", 2:"Sensitivity", 3:"Specificity", 4:"PPV", 5:"NPV"}, inplace=True)


Comp3 = pd.DataFrame(ci_bootstraps(Med['pTau_217'], Med['0_3vs4'], alpha=0.95, n_bootstraps=1000, rng_seed=42)).T
Comp3.rename(columns={0:"AUC", 1:"Youden", 2:"Sensitivity", 3:"Specificity", 4:"PPV", 5:"NPV"}, inplace=True)


Comp = pd.concat([Comp1, Comp2, Comp3])
# Comp.to_csv("/ROC_Youden_CI.csv")



#%% pTau 177.4+ participants
# Discrimination between disease stages of enrolled CI participants
Med = []
Med = Data[Data["Profile_newer"].notna()]
Med = Med[Med.simple_Diag.isin(["MCI", "AD"])]
Med = Med[Med.pTau_217 > 177.38].reset_index(drop=True)


# T = 177.38
Med['onlyHigh']  = np.where(Med["Profile_newer"]=="A+HIGH", 1, 0)

Comp4 = pd.DataFrame(ci_bootstraps(Med['pTau_217'], Med['onlyHigh'], alpha=0.95, n_bootstraps=1000, rng_seed=42)).T
Comp4.rename(columns={0:"AUC", 1:'Youden threshold', 2:"Sensitivity", 3:"Specificity", 4:"PPV", 5:"NPV"}, inplace=True)



#%% Supp Figure 1a: Spec-based threshold for A+High_ All
Med = []
Med = Data[Data["Profile_newer"].notna()].reset_index(drop=True)

# Binarize the profile
Med['0_3vs4'] = np.where(Med["Profile_newer"]=="A+HIGH", 1, 0)

# Thresholds
High_sens_90 = Sens_thresh(Med['pTau_217'], Med['0_3vs4'], 0.90)
High_spec_90 = Spec_thresh(Med['pTau_217'], Med['0_3vs4'], 0.90)
High_spec_95 = Spec_thresh(Med['pTau_217'], Med['0_3vs4'], 0.95)

# colours
palette = {'A-T-': "lightblue", 'A+T-':"#0ABAB5", 'A+MTL':"orange", 'A+MOD':"#ff4c4c", 'A+HIGH':"#cc0000"}


# Fig 1 - 90% sens
Med['color'] = np.where(Med["pTau_217"]<High_sens_90, "Included", "Excluded")
plt.figure()
sns.set_context("talk", font_scale=1.6)
ax = sns.catplot(x='Profile_newer', y='pTau_217', kind="box", height=10, 
            order=['A-T-', 'A+T-', 'A+MTL', 'A+MOD', 'A+HIGH'], color='white', linewidth=2, 
            data=Med,
            zorder=3)
ax = sns.swarmplot(x='Profile_newer', y='pTau_217', order=['A-T-', 'A+T-', 'A+MTL', 'A+MOD', 'A+HIGH'],
                    data=Med[Med.color == "Included"],
                   palette=palette, 
                  zorder=10, size=9, alpha = 0.9)
ax = sns.swarmplot(x='Profile_newer', y='pTau_217', order=['A-T-', 'A+T-', 'A+MTL', 'A+MOD', 'A+HIGH'],
                    data=Med[Med.color == "Excluded"],
                  color = 'grey',
                  zorder=10, size=9, alpha = 0.9)
plt.xlabel('')
ax.set_xticklabels([r'A-T-', r'A+T-', r'A+T$_{\mathrm{MTL}}$+', r'A+T$_{\mathrm{MOD}}$+', r'A+T$_{\mathrm{HIGH}}$+'], fontsize=25)
ax.set_xticklabels(ax.get_xticklabels(),rotation=25)
plt.ylabel("Plasma p217+tau (fg/ml)")
ax.set_yticks([0, 200, 400, 600, 800, 1000])
ax.set_yticklabels(("0", "200", "400", 
                        "600", "800", "1000"))
plt.axhline(High_sens_90, linestyle = ":", color = 'k') # Youden threshold in All for A+HIGH+
plt.text(4.6, High_sens_90, '90% sens', fontsize = 25)
plt.text(-0.5, -400, 
         "A+T$_{\mathrm{HIGH}}$+ excluded: " + str(round(len(Med[(Med.Profile_newer == 'A+HIGH') & (Med.pTau_217 > High_sens_90)])/len(Med[Med.Profile_newer == 'A+HIGH'])*100)) + "%", 
          fontsize = 25)
plt.text(-0.5, -500, 
         "PPV: " + str(round(len(Med[(Med.Profile_newer == 'A+HIGH') & (Med.pTau_217 > High_sens_90)])/len(Med[Med.pTau_217 > High_sens_90])*100)) + "%", 
          fontsize = 25)
plt.text(-0.5, -600, 
         "NPV: " + str(round(len(Med[(Med.Profile_newer != 'A+HIGH') & (Med.pTau_217 < High_sens_90)])/len(Med[Med.pTau_217 < High_sens_90])*100)) + "%", 
          fontsize = 25)
plt.tight_layout()
# plt.savefig('/boxplot_All_Sens90.PNG', dpi=300, bbox_inches='tight')


# Fig 2 - 90% spec
Med['color'] = np.where(Med["pTau_217"]<High_spec_90, "Included", "Excluded")
plt.figure()
sns.set_context("talk", font_scale=1.6)
ax = sns.catplot(x='Profile_newer', y='pTau_217', kind="box", height=10, 
            order=['A-T-', 'A+T-', 'A+MTL', 'A+MOD', 'A+HIGH'], color='white', linewidth=2, 
            data=Med,
            zorder=3)
ax = sns.swarmplot(x='Profile_newer', y='pTau_217', order=['A-T-', 'A+T-', 'A+MTL', 'A+MOD', 'A+HIGH'],
                    data=Med[Med.color == "Included"],
                   palette=palette, 
                  zorder=10, size=9, alpha = 0.9)
ax = sns.swarmplot(x='Profile_newer', y='pTau_217', order=['A-T-', 'A+T-', 'A+MTL', 'A+MOD', 'A+HIGH'],
                    data=Med[Med.color == "Excluded"],
                  color = 'grey',
                  zorder=10, size=9, alpha = 0.9)
plt.xlabel('')
ax.set_xticklabels([r'A-T-', r'A+T-', r'A+T$_{\mathrm{MTL}}$+', r'A+T$_{\mathrm{MOD}}$+', r'A+T$_{\mathrm{HIGH}}$+'], fontsize=25)
ax.set_xticklabels(ax.get_xticklabels(),rotation=25)
plt.ylabel("Plasma p217+tau (fg/ml)")
ax.set_yticks([0, 200, 400, 600, 800, 1000])
ax.set_yticklabels(("0", "200", "400", 
                        "600", "800", "1000"))
plt.axhline(High_spec_90, linestyle = ":", color = 'k')  # 90% spec threshold in All for A+THIGH+
plt.text(4.6, High_spec_90, '90% spec', fontsize = 25)
plt.text(-0.5, -400, 
         "A+T$_{\mathrm{HIGH}}$+ excluded: " + str(round(len(Med[(Med.Profile_newer == 'A+HIGH') & (Med.pTau_217 > High_spec_90)])/len(Med[Med.Profile_newer == 'A+HIGH'])*100)) + "%", 
          fontsize = 25)
plt.text(-0.5, -500, 
         "PPV: " + str(round(len(Med[(Med.Profile_newer == 'A+HIGH') & (Med.pTau_217 > High_spec_90)])/len(Med[Med.pTau_217 > High_spec_90])*100)) + "%", 
          fontsize = 25)
plt.text(-0.5, -600, 
         "NPV: " + str(round(len(Med[(Med.Profile_newer != 'A+HIGH') & (Med.pTau_217 < High_spec_90)])/len(Med[Med.pTau_217 < High_spec_90])*100)) + "%", 
          fontsize = 25)
plt.tight_layout()
# plt.savefig('/boxplot_All_Spec90.PNG', dpi=300, bbox_inches='tight')


# Fig 3 - 95% spec
Med['color'] = np.where(Med["pTau_217"]<High_spec_95, "Included", "Excluded")
plt.figure()
sns.set_context("talk", font_scale=1.6)
ax = sns.catplot(x='Profile_newer', y='pTau_217', kind="box", height=10, 
            order=['A-T-', 'A+T-', 'A+MTL', 'A+MOD', 'A+HIGH'], color='white', linewidth=2, 
            data=Med,
            zorder=3)
ax = sns.swarmplot(x='Profile_newer', y='pTau_217', order=['A-T-', 'A+T-', 'A+MTL', 'A+MOD', 'A+HIGH'],
                    data=Med[Med.color == "Included"],
                   palette=palette, 
                  zorder=10, size=9, alpha = 0.9)
ax = sns.swarmplot(x='Profile_newer', y='pTau_217', order=['A-T-', 'A+T-', 'A+MTL', 'A+MOD', 'A+HIGH'],
                    data=Med[Med.color == "Excluded"],
                  color = 'grey',
                  zorder=10, size=9, alpha = 0.9)
plt.xlabel('')
ax.set_xticklabels([r'A-T-', r'A+T-', r'A+T$_{\mathrm{MTL}}$+', r'A+T$_{\mathrm{MOD}}$+', r'A+T$_{\mathrm{HIGH}}$+'], fontsize=25)
ax.set_xticklabels(ax.get_xticklabels(),rotation=25)
plt.ylabel("Plasma p217+tau (fg/ml)")
ax.set_yticks([0, 200, 400, 600, 800, 1000])
ax.set_yticklabels(("0", "200", "400", 
                        "600", "800", "1000"))
plt.axhline(High_spec_95, linestyle = ":", color = 'k')  # 90% spec threshold in All for A+THIGH+
plt.text(4.6, High_spec_95, '95% spec', fontsize = 25)
plt.text(-0.5, -400, 
         "A+T$_{\mathrm{HIGH}}$+ excluded: " + str(round(len(Med[(Med.Profile_newer == 'A+HIGH') & (Med.pTau_217 > High_spec_95)])/len(Med[Med.Profile_newer == 'A+HIGH'])*100)) + "%", 
          fontsize = 25)
plt.text(-0.5, -500, 
         "PPV: " + str(round(len(Med[(Med.Profile_newer == 'A+HIGH') & (Med.pTau_217 > High_spec_95)])/len(Med[Med.pTau_217 > High_spec_95])*100)) + "%", 
          fontsize = 25)
plt.text(-0.5, -600, 
         "NPV: " + str(round(len(Med[(Med.Profile_newer != 'A+HIGH') & (Med.pTau_217 < High_spec_95)])/len(Med[Med.pTau_217 < High_spec_95])*100)) + "%", 
          fontsize = 25)
plt.tight_layout()
# plt.savefig('/boxplot_All_Spec95.PNG', dpi=300, bbox_inches='tight')



#%% Supp Figure 1b: Spec-based threshold for A+High_ CI only
Med = []
Med = Data[Data["Profile_newer"].notna()].reset_index(drop=True)
Med = Med[Med.simple_Diag.isin(["MCI", "AD"])]

# Binarize the profiles
Med['0_3vs4'] = np.where(Med["Profile_newer"]=="A+HIGH", 1, 0)

# Thresholds
High_Sens_90 = Sens_thresh(Med['pTau_217'], Med['0_3vs4'], 0.90)
High_spec_90 = Spec_thresh(Med['pTau_217'], Med['0_3vs4'], 0.90)
High_spec_95 = Spec_thresh(Med['pTau_217'], Med['0_3vs4'], 0.95)

# colours
palette = {'A-T-': "lightblue", 'A+T-':"#0ABAB5", 'A+MTL':"orange", 'A+MOD':"#ff4c4c", 'A+HIGH':"#cc0000"}


# Fig 1 - 90% sens
Med['color'] = np.where(Med["pTau_217"]<High_Sens_90, "Included", "Excluded")
plt.figure()
sns.set_context("talk", font_scale=1.6)
ax = sns.catplot(x='Profile_newer', y='pTau_217', kind="box", height=10, 
            order=['A-T-', 'A+T-', 'A+MTL', 'A+MOD', 'A+HIGH'], color='white', linewidth=2, 
            data=Med,
            zorder=3)
ax = sns.swarmplot(x='Profile_newer', y='pTau_217', order=['A-T-', 'A+T-', 'A+MTL', 'A+MOD', 'A+HIGH'],
                    data=Med[Med.color == "Included"],
                   palette=palette, 
                  zorder=10, size=9, alpha = 0.9)
ax = sns.swarmplot(x='Profile_newer', y='pTau_217', order=['A-T-', 'A+T-', 'A+MTL', 'A+MOD', 'A+HIGH'],
                    data=Med[Med.color == "Excluded"],
                  color = 'grey',
                  zorder=10, size=9, alpha = 0.9)
plt.xlabel('')
ax.set_xticklabels([r'A-T-', r'A+T-', r'A+T$_{\mathrm{MTL}}$+', r'A+T$_{\mathrm{MOD}}$+', r'A+T$_{\mathrm{HIGH}}$+'], fontsize=25)
ax.set_xticklabels(ax.get_xticklabels(),rotation=25)
plt.ylabel("Plasma p217+tau (fg/ml)")
ax.set_yticks([0, 200, 400, 600, 800, 1000])
ax.set_yticklabels(("0", "200", "400", 
                        "600", "800", "1000"))
plt.axhline(High_Sens_90, linestyle = ":", color = 'k') # Youden threshold in All for A+HIGH+
plt.text(4.6, High_Sens_90, '90% sens', fontsize = 25)
plt.text(-0.5, -400, 
         "A+T$_{\mathrm{HIGH}}$+ excluded: " + str(round(len(Med[(Med.Profile_newer == 'A+HIGH') & (Med.pTau_217 > High_Sens_90)])/len(Med[Med.Profile_newer == 'A+HIGH'])*100)) + "%", 
          fontsize = 25)
plt.text(-0.5, -500, 
         "PPV: " + str(round(len(Med[(Med.Profile_newer == 'A+HIGH') & (Med.pTau_217 > High_Sens_90)])/len(Med[Med.pTau_217 > High_Sens_90])*100)) + "%", 
          fontsize = 25)
plt.text(-0.5, -600, 
         "NPV: " + str(round(len(Med[(Med.Profile_newer != 'A+HIGH') & (Med.pTau_217 < High_Sens_90)])/len(Med[Med.pTau_217 < High_Sens_90])*100)) + "%", 
          fontsize = 25)
plt.tight_layout()
# plt.savefig('/boxplot_CI_Sens90.PNG', dpi=300, bbox_inches='tight')


# Fig 2 - 90% spec
Med['color'] = np.where(Med["pTau_217"]<High_spec_90, "Included", "Excluded")
plt.figure()
sns.set_context("talk", font_scale=1.6)
ax = sns.catplot(x='Profile_newer', y='pTau_217', kind="box", height=10, 
            order=['A-T-', 'A+T-', 'A+MTL', 'A+MOD', 'A+HIGH'], color='white', linewidth=2, 
            data=Med,
            zorder=3)
ax = sns.swarmplot(x='Profile_newer', y='pTau_217', order=['A-T-', 'A+T-', 'A+MTL', 'A+MOD', 'A+HIGH'],
                    data=Med[Med.color == "Included"],
                   palette=palette, 
                  zorder=10, size=9, alpha = 0.9)
ax = sns.swarmplot(x='Profile_newer', y='pTau_217', order=['A-T-', 'A+T-', 'A+MTL', 'A+MOD', 'A+HIGH'],
                    data=Med[Med.color == "Excluded"],
                  color = 'grey',
                  zorder=10, size=9, alpha = 0.9)
plt.xlabel('')
ax.set_xticklabels([r'A-T-', r'A+T-', r'A+T$_{\mathrm{MTL}}$+', r'A+T$_{\mathrm{MOD}}$+', r'A+T$_{\mathrm{HIGH}}$+'], fontsize=25)
ax.set_xticklabels(ax.get_xticklabels(),rotation=25)
plt.ylabel("Plasma p217+tau (fg/ml)")
ax.set_yticks([0, 200, 400, 600, 800, 1000])
ax.set_yticklabels(("0", "200", "400", 
                        "600", "800", "1000"))
plt.axhline(High_spec_90, linestyle = ":", color = 'k')  # 90% spec threshold in All for A+THIGH+
plt.text(4.6, High_spec_90, '90% spec', fontsize = 25)
plt.text(-0.5, -400, 
         "A+T$_{\mathrm{HIGH}}$+ excluded: " + str(round(len(Med[(Med.Profile_newer == 'A+HIGH') & (Med.pTau_217 > High_spec_90)])/len(Med[Med.Profile_newer == 'A+HIGH'])*100)) + "%", 
          fontsize = 25)
plt.text(-0.5, -500, 
         "PPV: " + str(round(len(Med[(Med.Profile_newer == 'A+HIGH') & (Med.pTau_217 > High_spec_90)])/len(Med[Med.pTau_217 > High_spec_90])*100)) + "%", 
          fontsize = 25)
plt.text(-0.5, -600, 
         "NPV: " + str(round(len(Med[(Med.Profile_newer != 'A+HIGH') & (Med.pTau_217 < High_spec_90)])/len(Med[Med.pTau_217 < High_spec_90])*100)) + "%", 
          fontsize = 25)
plt.tight_layout()
# plt.savefig('/boxplot_CI_Spec90.PNG', dpi=300, bbox_inches='tight')


# Fig 3 - 95% spec
Med['color'] = np.where(Med["pTau_217"]<High_spec_95, "Included", "Excluded")
plt.figure()
sns.set_context("talk", font_scale=1.6)
ax = sns.catplot(x='Profile_newer', y='pTau_217', kind="box", height=10, 
            order=['A-T-', 'A+T-', 'A+MTL', 'A+MOD', 'A+HIGH'], color='white', linewidth=2, 
            data=Med,
            zorder=3)
ax = sns.swarmplot(x='Profile_newer', y='pTau_217', order=['A-T-', 'A+T-', 'A+MTL', 'A+MOD', 'A+HIGH'],
                    data=Med[Med.color == "Included"],
                   palette=palette, 
                  zorder=10, size=9, alpha = 0.9)
ax = sns.swarmplot(x='Profile_newer', y='pTau_217', order=['A-T-', 'A+T-', 'A+MTL', 'A+MOD', 'A+HIGH'],
                    data=Med[Med.color == "Excluded"],
                  color = 'grey',
                  zorder=10, size=9, alpha = 0.9)
plt.xlabel('')
ax.set_xticklabels([r'A-T-', r'A+T-', r'A+T$_{\mathrm{MTL}}$+', r'A+T$_{\mathrm{MOD}}$+', r'A+T$_{\mathrm{HIGH}}$+'], fontsize=25)
ax.set_xticklabels(ax.get_xticklabels(),rotation=25)
plt.ylabel("Plasma p217+tau (fg/ml)")
ax.set_yticks([0, 200, 400, 600, 800, 1000])
ax.set_yticklabels(("0", "200", "400", 
                        "600", "800", "1000"))
plt.axhline(High_spec_95, linestyle = ":", color = 'k')  # 90% spec threshold in All for A+THIGH+
plt.text(4.6, High_spec_95, '95% spec', fontsize = 25)
plt.text(-0.5, -400, 
         "A+T$_{\mathrm{HIGH}}$+ excluded: " + str(round(len(Med[(Med.Profile_newer == 'A+HIGH') & (Med.pTau_217 > High_spec_95)])/len(Med[Med.Profile_newer == 'A+HIGH'])*100)) + "%", 
          fontsize = 25)
plt.text(-0.5, -500, 
         "PPV: " + str(round(len(Med[(Med.Profile_newer == 'A+HIGH') & (Med.pTau_217 > High_spec_95)])/len(Med[Med.pTau_217 > High_spec_95])*100)) + "%", 
          fontsize = 25)
plt.text(-0.5, -600, 
         "NPV: " + str(round(len(Med[(Med.Profile_newer != 'A+HIGH') & (Med.pTau_217 < High_spec_95)])/len(Med[Med.pTau_217 < High_spec_95])*100)) + "%", 
          fontsize = 25)
plt.tight_layout()
# plt.savefig('/boxplot_CI_Spec95.PNG', dpi=300, bbox_inches='tight')



#%% Figure 3a: Probability curve All
""" here we remove ptau > 500 fg/ml as logisitic regression is sensitive to outliers and there are only a few values above 500
So we cannot fit a line """

################################ 0 vs 1-4  ################################
# Probability Score 

# 95% se & sp thresholds ------------------------------------------------------
Med = []
Med = Data[(Data["Profile_newer"].notna()) & (Data["pTau_217"]<500)].reset_index(drop=True)
# Med = Data[Data["Profile_newer"].notna()].reset_index(drop=True)

Med['0vs1_4'] = np.where(Med["Profile_newer"]=="A-T-", "neg", "pos")

A_prob_boot = prob_curve_dist(Med, "pTau_217", "0vs1_4", "Plasma p217+tau (fg/ml)", "Probability score", "sens_spec", 0.95, 0, 460, 50)
plt.close()


################################ 0-2 vs 3-4   ################################
# Probability Score 

# 95% se & sp thresholds ------------------------------------------------------
Med = []
Med = Data[(Data["Profile_newer"].notna()) & (Data["pTau_217"]<500)].reset_index(drop=True)
Med['02vs34']  = np.where((Med["Profile_newer"]=="A-T-") | (Med["Profile_newer"]=="A+T-") | (Med["Profile_newer"]=="A+MTL"), "neg", "pos")

A_prob_boot1 = prob_curve_dist(Med, "pTau_217", "02vs34", "Plasma p217+tau (fg/ml)", "Probability score", "sens_spec", 0.95, 0, 460, 50)
plt.close()


################################ 0-3 vs 4   ################################
# Probability Score 

# 95% se & sp thresholds ------------------------------------------------------
Med = []
Med = Data[(Data["Profile_newer"].notna()) & (Data["pTau_217"]<500)].reset_index(drop=True)
Med['0_3vs4'] = np.where(Med["Profile_newer"]=="A+HIGH", "pos", "neg")

A_prob_boot2 = prob_curve_dist(Med, "pTau_217", "0_3vs4", "Plasma p217+tau (fg/ml)", "Probability score", "sens_spec", 0.95, 0, 460, 50)
plt.close()


########################## All 4 curves in one plot ###########################
# Merge T- and MTL
Med.Profile_newer = Med.Profile_newer.replace('A+T-', "A+T-MTL+")
Med.Profile_newer = Med.Profile_newer.replace('A+MTL', "A+T-MTL+")

# the curves based on bootstrap means:
Med["y1"] = 1 - A_prob_boot.Mean
Med["y2"] = A_prob_boot.Mean - A_prob_boot1.Mean
Med["y3"] = A_prob_boot1.Mean - A_prob_boot2.Mean
Med["y4"] = A_prob_boot2.Mean

# start Figure
sns.set_context("talk", font_scale=0.9)
Med = Med.sort_values("pTau_217")

#### Moving average
# plot    
plt.figure(figsize=(8, 5))
Data_SAC = pd.DataFrame(columns=["pT_anchor", "Prv_AnTn", "Prv_TnegMTL", "Prv_MOD", "Prv_HIGH"], index=range(0, 401))
for i in range(40,440):
    sub=Med[Med["pTau_217"].between(i-40, i+40)]
    Data_SAC["pT_anchor"][i-40] = i
    if len(sub)>1:
        Data_SAC["Prv_AnTn"][i-40] = len(sub[sub["Profile_newer"] == 'A-T-'])/(len(sub))
        Data_SAC["Prv_TnegMTL"][i-40] = len(sub[sub["Profile_newer"] == 'A+T-MTL+'])/(len(sub))
        Data_SAC["Prv_MOD"][i-40] = len(sub[sub["Profile_newer"] == 'A+MOD'])/(len(sub))
        Data_SAC["Prv_HIGH"][i-40] = len(sub[sub["Profile_newer"] == 'A+HIGH'])/(len(sub))
    else:
        continue

for col in Data_SAC.columns:
      Data_SAC[col] = Data_SAC[col].astype(float)          

# from moving average
plt.scatter(Data_SAC['pT_anchor'], Data_SAC['Prv_AnTn'], color='lightblue', alpha=0.4, s=1, marker='.')
plt.scatter(Data_SAC['pT_anchor'], Data_SAC['Prv_TnegMTL'], color='orange', alpha=0.4, s=1, marker='.')
plt.scatter(Data_SAC['pT_anchor'], Data_SAC['Prv_MOD'], color='red', alpha=0.4, s=1, marker='.')
plt.scatter(Data_SAC['pT_anchor'], Data_SAC['Prv_HIGH'], color='darkred', alpha=0.4, s=1, marker='.')

# from regression
line1, = plt.plot(Med['pTau_217'], Med['y1'], color='lightblue')
line2, = plt.plot(Med['pTau_217'], Med['y2'], color='orange')
line3, = plt.plot(Med['pTau_217'], Med['y3'], color='red')
line4, = plt.plot(Med['pTau_217'], Med['y4'], color='darkred')

plt.xlim(0, 460)
plt.ylim(-0.02, 1.02)
plt.xticks(np.arange(50, 500, 50))
plt.xlabel('Plasma p217+tau (fg/ml)')
plt.ylabel('Probability score')

# Add a legend
plt.legend([line1, line2, line3, line4], [r'$A-T-$', r'$A+T_{None/MTL}$', r'$A+T_{MOD}+$', r'$A+T_{HIGH}+$'], fontsize=11, loc='upper right')
# plt.savefig('/4curves_sup_impsd_all.PNG', dpi=300, bbox_inches='tight')
plt.show()

# Add table
x_list = [50, 100, 150, 200, 250, 300, 350, 400]
prob_list_y1 = [Med[Med['pTau_217'].between(49, 51)]["y1"].mean(),
                     Med[Med['pTau_217'].between(99, 101)]["y1"].mean(),
                     Med[Med['pTau_217'].between(149, 151)]["y1"].mean(),
                     Med[Med['pTau_217'].between(199, 201)]["y1"].mean(),
                     Med[Med['pTau_217'].between(249, 251)]["y1"].mean(),
                     Med[Med['pTau_217'].between(299, 301)]["y1"].mean(),
                     Med[Med['pTau_217'].between(340, 360)]["y1"].mean(),
                     Med[Med['pTau_217'].between(390, 410)]["y1"].mean()]

prob_list_y2 = [Med[Med['pTau_217'].between(49, 51)]["y2"].mean(),
                     Med[Med['pTau_217'].between(99, 101)]["y2"].mean(),
                     Med[Med['pTau_217'].between(149, 151)]["y2"].mean(),
                     Med[Med['pTau_217'].between(199, 201)]["y2"].mean(),
                     Med[Med['pTau_217'].between(249, 251)]["y2"].mean(),
                     Med[Med['pTau_217'].between(299, 301)]["y2"].mean(),
                     Med[Med['pTau_217'].between(340, 360)]["y2"].mean(),
                     Med[Med['pTau_217'].between(390, 410)]["y2"].mean()]

prob_list_y3 = [Med[Med['pTau_217'].between(49, 51)]["y3"].mean(),
                     Med[Med['pTau_217'].between(99, 101)]["y3"].mean(),
                     Med[Med['pTau_217'].between(149, 151)]["y3"].mean(),
                     Med[Med['pTau_217'].between(199, 201)]["y3"].mean(),
                     Med[Med['pTau_217'].between(249, 251)]["y3"].mean(),
                     Med[Med['pTau_217'].between(299, 301)]["y3"].mean(),
                     Med[Med['pTau_217'].between(340, 360)]["y3"].mean(),
                     Med[Med['pTau_217'].between(390, 410)]["y3"].mean()]


prob_list_y4 = [Med[Med['pTau_217'].between(49, 51)]["y4"].mean(),
                     Med[Med['pTau_217'].between(99, 101)]["y4"].mean(),
                     Med[Med['pTau_217'].between(149, 151)]["y4"].mean(),
                     Med[Med['pTau_217'].between(199, 201)]["y4"].mean(),
                     Med[Med['pTau_217'].between(249, 251)]["y4"].mean(),
                     Med[Med['pTau_217'].between(299, 301)]["y4"].mean(),
                     Med[Med['pTau_217'].between(340, 360)]["y4"].mean(),
                     Med[Med['pTau_217'].between(390, 410)]["y4"].mean()]
table_data_all = pd.DataFrame({
                  'x_list': x_list, 
                  'prob_list_y1': prob_list_y1,
                  'prob_list_y2': prob_list_y2,
                  'prob_list_y3': prob_list_y3,
                  'prob_list_y4': prob_list_y4
                  }).transpose()
table_data_all.iloc[1:5, :] = table_data_all.iloc[1:5, :].round(2)
# table_data_all.to_csv("/prob_curve_values_All.csv")



#%% Figure 3b: Probability curve CI only
Med = []
Med = Data[(Data["Profile_newer"].notna()) & (Data["pTau_217"]<500)].reset_index(drop=True)
Med = Med[Med.simple_Diag.isin(["MCI", "AD"])].reset_index(drop=True)
""" here we remove ptau > 500 fg/ml as logisitic regression is sensitive to outliers """

################################ 0 vs 1-4  ################################
# Probability Score 

# 95% se & sp thresholds ------------------------------------------------------
Med['0vs1_4'] = np.where(Med["Profile_newer"]=="A-T-", "neg", "pos")
A_prob_boot = prob_curve_dist(Med, "pTau_217", "0vs1_4", "Plasma p217+tau (fg/ml)", "Probability score", "sens_spec", 0.95, 0, 460, 50)
plt.close()


################################ 0-2 vs 3-4   ################################
# Probability Score 

# 95% se & sp thresholds ------------------------------------------------------
Med['02vs34']  = np.where((Med["Profile_newer"]=="A-T-") | (Med["Profile_newer"]=="A+T-") | (Med["Profile_newer"]=="A+MTL"), "neg", "pos")

A_prob_boot1 = prob_curve_dist(Med, "pTau_217", "02vs34", "Plasma p217+tau (fg/ml)", "Probability score", "sens_spec", 0.95, 0, 460, 50)
plt.close()


################################ 0-3 vs 4   ################################
# Probability Score 

# 95% se & sp thresholds ------------------------------------------------------
Med['0_3vs4'] = np.where(Med["Profile_newer"]=="A+HIGH", "pos", "neg")

A_prob_boot2 = prob_curve_dist(Med, "pTau_217", "0_3vs4", "Plasma p217+tau (fg/ml)", "Probability score", "sens_spec", 0.95, 0, 460, 50)
plt.close()


########################## All 4 curves in one plot ###########################
# Merge T- and MTL
Med.Profile_newer = Med.Profile_newer.replace('A+T-', "A+T-MTL+")
Med.Profile_newer = Med.Profile_newer.replace('A+MTL', "A+T-MTL+")

# the curves based on bootstrap means:
Med["y1"] = 1 - A_prob_boot.Mean
Med["y2"] = A_prob_boot.Mean - A_prob_boot1.Mean
Med["y3"] = A_prob_boot1.Mean - A_prob_boot2.Mean
Med["y4"] = A_prob_boot2.Mean

# Line plot
sns.set_context("talk", font_scale=0.9)
Med = Med.sort_values("pTau_217")


#### plus Moving average: CI only
# plot    
plt.figure(figsize=(8, 5))
Data_SAC = pd.DataFrame(columns=["pT_anchor", "Prv_AnTn", "Prv_TnegMTL", "Prv_MOD", "Prv_HIGH"], index=range(0, 401))
for i in range(40,440):
    sub=Med[Med["pTau_217"].between(i-40, i+40)]
    Data_SAC["pT_anchor"][i-40] = i
    if len(sub)>1:
        Data_SAC["Prv_AnTn"][i-40] = len(sub[sub["Profile_newer"] == 'A-T-'])/(len(sub))
        Data_SAC["Prv_TnegMTL"][i-40] = len(sub[sub["Profile_newer"] == 'A+T-MTL+'])/(len(sub))
        Data_SAC["Prv_MOD"][i-40] = len(sub[sub["Profile_newer"] == 'A+MOD'])/(len(sub))
        Data_SAC["Prv_HIGH"][i-40] = len(sub[sub["Profile_newer"] == 'A+HIGH'])/(len(sub))
    else:
        continue

for col in Data_SAC.columns:
      Data_SAC[col] = Data_SAC[col].astype(float)          

# from moving average
plt.scatter(Data_SAC['pT_anchor'], Data_SAC['Prv_AnTn'], color='lightblue', alpha=0.4, s=1, marker='.')
plt.scatter(Data_SAC['pT_anchor'], Data_SAC['Prv_TnegMTL'], color='orange', alpha=0.4, s=1, marker='.')
plt.scatter(Data_SAC['pT_anchor'], Data_SAC['Prv_MOD'], color='red', alpha=0.4, s=1, marker='.')
plt.scatter(Data_SAC['pT_anchor'], Data_SAC['Prv_HIGH'], color='darkred', alpha=0.4, s=1, marker='.')

# from regression
line1, = plt.plot(Med['pTau_217'], Med['y1'], color='lightblue')
line2, = plt.plot(Med['pTau_217'], Med['y2'], color='orange')
line3, = plt.plot(Med['pTau_217'], Med['y3'], color='red')
line4, = plt.plot(Med['pTau_217'], Med['y4'], color='darkred')

plt.xlim(0, 460)
plt.ylim(-0.02, 1.02)
plt.xticks(np.arange(50, 500, 50))
plt.xlabel('Plasma p217+tau (fg/ml)')
plt.ylabel('Probability score')

# Add a legend
plt.legend([line1, line2, line3, line4], [r'$A-T-$', r'$A+T_{None/MTL}$', r'$A+T_{MOD}+$', r'$A+T_{HIGH}+$'], fontsize=11, loc='upper right')
# plt.savefig('/Revised Graphics/4curves_sup_impsd_CI.PNG', dpi=300, bbox_inches='tight')
plt.show()

# Add table
x_list = [50, 100, 150, 200, 250, 300, 350, 400]
prob_list_y1 = [Med[Med['pTau_217'].between(49, 51)]["y1"].mean(),
                     Med[Med['pTau_217'].between(99, 101)]["y1"].mean(),
                     Med[Med['pTau_217'].between(149, 151)]["y1"].mean(),
                     Med[Med['pTau_217'].between(199, 201)]["y1"].mean(),
                     Med[Med['pTau_217'].between(249, 251)]["y1"].mean(),
                     Med[Med['pTau_217'].between(299, 301)]["y1"].mean(),
                     Med[Med['pTau_217'].between(340, 360)]["y1"].mean(),
                     Med[Med['pTau_217'].between(390, 410)]["y1"].mean()]

prob_list_y2 = [Med[Med['pTau_217'].between(49, 51)]["y2"].mean(),
                     Med[Med['pTau_217'].between(99, 101)]["y2"].mean(),
                     Med[Med['pTau_217'].between(149, 151)]["y2"].mean(),
                     Med[Med['pTau_217'].between(199, 201)]["y2"].mean(),
                     Med[Med['pTau_217'].between(249, 251)]["y2"].mean(),
                     Med[Med['pTau_217'].between(299, 301)]["y2"].mean(),
                     Med[Med['pTau_217'].between(340, 360)]["y2"].mean(),
                     Med[Med['pTau_217'].between(390, 410)]["y2"].mean()]

prob_list_y3 = [Med[Med['pTau_217'].between(49, 51)]["y3"].mean(),
                     Med[Med['pTau_217'].between(99, 101)]["y3"].mean(),
                     Med[Med['pTau_217'].between(149, 151)]["y3"].mean(),
                     Med[Med['pTau_217'].between(199, 201)]["y3"].mean(),
                     Med[Med['pTau_217'].between(249, 251)]["y3"].mean(),
                     Med[Med['pTau_217'].between(299, 301)]["y3"].mean(),
                     Med[Med['pTau_217'].between(340, 360)]["y3"].mean(),
                     Med[Med['pTau_217'].between(390, 410)]["y3"].mean()]


prob_list_y4 = [Med[Med['pTau_217'].between(49, 51)]["y4"].mean(),
                     Med[Med['pTau_217'].between(99, 101)]["y4"].mean(),
                     Med[Med['pTau_217'].between(149, 151)]["y4"].mean(),
                     Med[Med['pTau_217'].between(199, 201)]["y4"].mean(),
                     Med[Med['pTau_217'].between(249, 251)]["y4"].mean(),
                     Med[Med['pTau_217'].between(299, 301)]["y4"].mean(),
                     Med[Med['pTau_217'].between(340, 360)]["y4"].mean(),
                     Med[Med['pTau_217'].between(390, 410)]["y4"].mean()]
table_data_CI = pd.DataFrame({
                  'x_list': x_list, 
                  'prob_list_y1': prob_list_y1,
                  'prob_list_y2': prob_list_y2,
                  'prob_list_y3': prob_list_y3,
                  'prob_list_y4': prob_list_y4
                  }).transpose()
table_data_CI.iloc[1:5, :] = table_data_CI.iloc[1:5, :].round(2)
# table_data_CI.to_csv("/prob_curve_values_CI.csv")



#%% Figure 2a: ROC analysis All
# 0 vs 1-3, 0-1 vs 2-3 and 0-2 vs 3 classification
Med = []
Med = Data[(Data[["Profile_newer", "pTau_217", "Tau_Age", "Gender", "APOE"]].notna()).all(axis=1)].reset_index(drop=True)

Med = Med.replace('Male', 0)
Med = Med.replace('Female', 1)

Med.APOE = Med.APOE.replace('E3_E3', 0)
Med.APOE = Med.APOE.replace('E3_E2', 0)
Med.APOE = Med.APOE.replace('E2_E3', 0)
Med.APOE = Med.APOE.replace('E2_E2', 0)
Med.APOE = Med.APOE.replace('E4_E4', 1)
Med.APOE = Med.APOE.replace('E4_E2', 1)
Med.APOE = Med.APOE.replace('E2_E4', 1)
Med.APOE = Med.APOE.replace('E4_E3', 1)
Med.APOE = Med.APOE.replace('E3_E4', 1)

# X and y 
X1 = Med[["Tau_Age", "Gender", "APOE"]]
X2 = Med[["pTau_217"]]
X3 = Med[["pTau_217", "Tau_Age"]]
X4 = Med[["pTau_217", "Gender"]]
X5 = Med[["pTau_217", "APOE"]]
X6 = Med[["pTau_217", "Tau_Age", "Gender", "APOE"]]

#### 0 vs 1-4 -----------------------------------------------------------------
y1 = np.where(Med["Profile_newer"]=="A-T-", 0, 1)

""" for logistic regression y should be array
Don't convert to dataframe"""

# y scores  
Log_Reg = []

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X1, y1) 
yscore1 = Log_Reg.predict_proba(X1)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X2, y1) 
yscore2 = Log_Reg.predict_proba(X2)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X3, y1) 
yscore3 = Log_Reg.predict_proba(X3)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X4, y1) 
yscore4 = Log_Reg.predict_proba(X4)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X5, y1) 
yscore5 = Log_Reg.predict_proba(X5)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X6, y1) 
yscore6 = Log_Reg.predict_proba(X6)[:,1]

# AUC statistical differences  -------------------------------------------------
""" 10 to the power of delong is to inverse the log10 of the pvalue"""
""" multiplication by 5 is bonferronie adjustment for multiple comparisons"""

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore1, yscore2)
print("Adjusted p-value for the difference between base_model and pT217: {}".format(p_value*5))
asterisk_yscore2 = get_asterisks(p_value*5)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore3)
print("Adjusted p-value for the difference between pT217 and pT217_age: {}".format(p_value*5))
asterisk_yscore3 = get_hashtag(p_value*5)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore4)
print("Adjusted p-value for the difference between pT217 and pT217_gender: {}".format(p_value*5))
asterisk_yscore4 = get_hashtag(p_value*5)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore5)
print("Adjusted p-value for the difference between pT217 and pT217_APOE: {}".format(p_value*5))
asterisk_yscore5 = get_hashtag(p_value*5)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore6)
print("Adjusted p-value for the difference between pT217 and full_model: {}".format(p_value*5))
asterisk_yscore6 = get_hashtag(p_value*5)

# Plot 
""" for logistic regression y was array.
Here for ROC-plot we need to convert to dataframe"""

yscore1 = pd.DataFrame(yscore1)
yscore1.name = "base_model"

yscore2 = pd.DataFrame(yscore2)
yscore2.name = "pT217"

yscore3 = pd.DataFrame(yscore3)
yscore3.name = "pT217 + age"

yscore4 = pd.DataFrame(yscore4)
yscore4.name = "pT217 + sex"

yscore5 = pd.DataFrame(yscore5)
yscore5.name = "pT217 + APOE"

yscore6 = pd.DataFrame(yscore6)
yscore6.name = "full model"
y1 = pd.DataFrame(y1)


figure = plt.figure()
ROC_plot(yscore1, y1, figure, color = "#e377c2")
ROC_plot(yscore2, y1, figure, color = "#ff7f0e")
ROC_plot(yscore3, y1, figure, color = "#2ca02c")
ROC_plot(yscore4, y1, figure, color = "#d62728")
ROC_plot(yscore5, y1, figure, color = "#9467bd")
ROC_plot(yscore6, y1, figure, color = "#8c564b")
plt.gcf().set_size_inches(6, 4)  # Adjust the figure size as needed

# add asterisk for p value
plt.text(0.19, -0.18, f'{asterisk_yscore2}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=26, color = 'red')
plt.text(0.19, -0.26, f'{asterisk_yscore3}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.34, f'{asterisk_yscore4}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.42, f'{asterisk_yscore5}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.50, f'{asterisk_yscore6}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
# plt.savefig('/ROC_0vs14.PNG', dpi=300, bbox_inches='tight')


#### 0-2 vs 3-4 ---------------------------------------------------------------
y1 = []
y1 = np.where((Med["Profile_newer"]=="A-T-") | (Med["Profile_newer"]=="A+T-") | (Med["Profile_newer"]=="A+MTL"), 0, 1)

""" for logistic regression y should be array
Don't convert to dataframe"""

# y scores  
Log_Reg = []

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X1, y1) 
yscore1 = Log_Reg.predict_proba(X1)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X2, y1) 
yscore2 = Log_Reg.predict_proba(X2)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X3, y1) 
yscore3 = Log_Reg.predict_proba(X3)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X4, y1) 
yscore4 = Log_Reg.predict_proba(X4)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X5, y1) 
yscore5 = Log_Reg.predict_proba(X5)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X6, y1) 
yscore6 = Log_Reg.predict_proba(X6)[:,1]

# AUC statistical differences  -------------------------------------------------
""" 10 to the power of delong is to inverse the log10 of the pvalue"""
""" multiplication by 5 is to adjust for multiple comparisons"""

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore1, yscore2)
print("Adjusted p-value for the difference between base_model and pT217: {}".format(p_value*5))
asterisk_yscore2 = get_asterisks(p_value*5)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore3)
print("Adjusted p-value for the difference between pT217 and pT217_age: {}".format(p_value*5))
asterisk_yscore3 = get_hashtag(p_value*5)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore4)
print("Adjusted p-value for the difference between pT217 and pT217_gender: {}".format(p_value*5))
asterisk_yscore4 = get_hashtag(p_value*5)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore5)
print("Adjusted p-value for the difference between pT217 and pT217_APOE: {}".format(p_value*5))
asterisk_yscore5 = get_hashtag(p_value*5)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore6)
print("Adjusted p-value for the difference between pT217 and full_model: {}".format(p_value*5))
asterisk_yscore6 = get_hashtag(p_value*5)

# Plot 
""" for logistic regression y was array.
Here for ROC-plot we need to convert to dataframe"""

yscore1 = pd.DataFrame(yscore1)
yscore1.name = "base_model"

yscore2 = pd.DataFrame(yscore2)
yscore2.name = "pT217"

yscore3 = pd.DataFrame(yscore3)
yscore3.name = "pT217 + age"

yscore4 = pd.DataFrame(yscore4)
yscore4.name = "pT217 + sex"

yscore5 = pd.DataFrame(yscore5)
yscore5.name = "pT217 + APOE"

yscore6 = pd.DataFrame(yscore6)
yscore6.name = "full model"
y1 = pd.DataFrame(y1)


figure = plt.figure()
ROC_plot(yscore1, y1, figure, color = "#e377c2")
ROC_plot(yscore2, y1, figure, color = "#ff7f0e")
ROC_plot(yscore3, y1, figure, color = "#2ca02c")
ROC_plot(yscore4, y1, figure, color = "#d62728")
ROC_plot(yscore5, y1, figure, color = "#9467bd")
ROC_plot(yscore6, y1, figure, color = "#8c564b")
plt.gcf().set_size_inches(6, 4)  # Adjust the figure size as needed

# add asterisk
plt.text(0.19, -0.18, f'{asterisk_yscore2}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=26, color = 'red')
plt.text(0.19, -0.26, f'{asterisk_yscore3}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.34, f'{asterisk_yscore4}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.42, f'{asterisk_yscore5}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.50, f'{asterisk_yscore6}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
# plt.savefig('/ROC_02vs34.PNG', dpi=300, bbox_inches='tight')


#### 0-3 vs 4 -----------------------------------------------------------------
y1 = []
y1 = np.where(Med["Profile_newer"]=="A+HIGH", 1, 0)

""" for logistic regression y should be array
Don't convert to dataframe"""

# y scores  
Log_Reg = []

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X1, y1) 
yscore1 = Log_Reg.predict_proba(X1)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X2, y1) 
yscore2 = Log_Reg.predict_proba(X2)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X3, y1) 
yscore3 = Log_Reg.predict_proba(X3)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X4, y1) 
yscore4 = Log_Reg.predict_proba(X4)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X5, y1) 
yscore5 = Log_Reg.predict_proba(X5)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X6, y1) 
yscore6 = Log_Reg.predict_proba(X6)[:,1]

# AUC statistical differences  -------------------------------------------------
""" 10 to the power of delong is to inverse the log10 of the pvalue"""
""" multiplication by 5 is to adjust for multiple comparisons"""

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore1, yscore2)
print("Adjusted p-value for the difference between base_model and pT217: {}".format(p_value*5))
asterisk_yscore2 = get_asterisks(p_value*5)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore3)
print("Adjusted p-value for the difference between pT217 and pT217_age: {}".format(p_value*5))
asterisk_yscore3 = get_hashtag(p_value*5)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore4)
print("Adjusted p-value for the difference between pT217 and pT217_gender: {}".format(p_value*5))
asterisk_yscore4 = get_hashtag(p_value*5)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore5)
print("Adjusted p-value for the difference between pT217 and pT217_APOE: {}".format(p_value*5))
asterisk_yscore5 = get_hashtag(p_value*5)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore6)
print("Adjusted p-value for the difference between pT217 and full_model: {}".format(p_value*5))
asterisk_yscore6 = get_hashtag(p_value*5)

# Plot 
""" for logistic regression y was array.
Here for ROC-plot we need to convert to dataframe"""

yscore1 = pd.DataFrame(yscore1)
yscore1.name = "base_model"

yscore2 = pd.DataFrame(yscore2)
yscore2.name = "pT217"

yscore3 = pd.DataFrame(yscore3)
yscore3.name = "pT217 + age"

yscore4 = pd.DataFrame(yscore4)
yscore4.name = "pT217 + sex"

yscore5 = pd.DataFrame(yscore5)
yscore5.name = "pT217 + APOE"

yscore6 = pd.DataFrame(yscore6)
yscore6.name = "full model" 
y1 = pd.DataFrame(y1)


figure = plt.figure()
ROC_plot(yscore1, y1, figure, color = "#e377c2")
ROC_plot(yscore2, y1, figure, color = "#ff7f0e")
ROC_plot(yscore3, y1, figure, color = "#2ca02c")
ROC_plot(yscore4, y1, figure, color = "#d62728")
ROC_plot(yscore5, y1, figure, color = "#9467bd")
ROC_plot(yscore6, y1, figure, color = "#8c564b")
plt.gcf().set_size_inches(6, 4)  # Adjust the figure size as needed

# add asterisk
plt.text(0.19, -0.18, f'{asterisk_yscore2}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=26, color = 'red')
plt.text(0.19, -0.26, f'{asterisk_yscore3}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.34, f'{asterisk_yscore4}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.42, f'{asterisk_yscore5}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.50, f'{asterisk_yscore6}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
# plt.savefig('/ROC_03vs4.PNG', dpi=300, bbox_inches='tight')

# plt.hist(Med[Med["0vs1_4"]==0]["pTau_217"], color = 'green', alpha = 0.3)
# plt.hist(Med[Med["0vs1_4"]==1]["pTau_217"], color = 'grey', alpha = 0.3)
# plt.legend([r'$A-T-$', r'$A+$'], fontsize=11, loc='upper right')
# plt.axvline(Youden_thresh(Med["pTau_217"], Med["0vs1_4"]), linestyle ="dotted", color='grey', alpha=0.9)



#%% Figure 2b: ROC analysis CI only 
# 0 vs 1-3, 0-1 vs 2-3 and 0-2 vs 3 classification
Med = []
Med = Data[(Data[["Profile_newer", "pTau_217", "Tau_Age", "Gender", "APOE"]].notna()).all(axis=1)].reset_index(drop=True)
Med = Med[Med.simple_Diag.isin(["MCI", "AD"])].reset_index(drop=True)

Med = Med.replace('Male', 0)
Med = Med.replace('Female', 1)

Med.APOE = Med.APOE.replace('E3_E3', 0)
Med.APOE = Med.APOE.replace('E3_E2', 0)
Med.APOE = Med.APOE.replace('E2_E3', 0)
Med.APOE = Med.APOE.replace('E2_E2', 0)
Med.APOE = Med.APOE.replace('E4_E4', 1)
Med.APOE = Med.APOE.replace('E4_E2', 1)
Med.APOE = Med.APOE.replace('E2_E4', 1)
Med.APOE = Med.APOE.replace('E4_E3', 1)
Med.APOE = Med.APOE.replace('E3_E4', 1)

# X and y 
X1 = Med[["Tau_Age", "Gender", "APOE"]]
X2 = Med[["pTau_217"]]
X3 = Med[["pTau_217", "Tau_Age"]]
X4 = Med[["pTau_217", "Gender"]]
X5 = Med[["pTau_217", "APOE"]]
X6 = Med[["pTau_217", "Tau_Age", "Gender", "APOE"]]

#### 0 vs 1-4 -----------------------------------------------------------------
y1 = np.where(Med["Profile_newer"]=="A-T-", 0, 1)

""" for logistic regression y should be array
Don't convert to dataframe"""

# y scores  
Log_Reg = []

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X1, y1) 
yscore1 = Log_Reg.predict_proba(X1)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X2, y1) 
yscore2 = Log_Reg.predict_proba(X2)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X3, y1) 
yscore3 = Log_Reg.predict_proba(X3)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X4, y1) 
yscore4 = Log_Reg.predict_proba(X4)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X5, y1) 
yscore5 = Log_Reg.predict_proba(X5)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X6, y1) 
yscore6 = Log_Reg.predict_proba(X6)[:,1]

# AUC statistical differences  -------------------------------------------------
""" 10 to the power of delong is to inverse the log10 of the pvalue"""
""" multiplication by 5 is to adjust for multiple comparisons"""

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore1, yscore2)
print("Adjusted p-value for the difference between base_model and pT217: {}".format(p_value*5))
asterisk_yscore2 = get_asterisks(p_value*5)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore3)
print("Adjusted p-value for the difference between pT217 and pT217_age: {}".format(p_value*5))
asterisk_yscore3 = get_hashtag(p_value*5)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore4)
print("Adjusted p-value for the difference between pT217 and pT217_gender: {}".format(p_value*5))
asterisk_yscore4 = get_hashtag(p_value*5)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore5)
print("Adjusted p-value for the difference between pT217 and pT217_APOE: {}".format(p_value*5))
asterisk_yscore5 = get_hashtag(p_value*5)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore6)
print("Adjusted p-value for the difference between pT217 and full_model: {}".format(p_value*5))
asterisk_yscore6 = get_hashtag(p_value*5)

# Plot 
""" for logistic regression y was array.
Here for ROC-plot we need to convert to dataframe"""

yscore1 = pd.DataFrame(yscore1)
yscore1.name = "base_model"

yscore2 = pd.DataFrame(yscore2)
yscore2.name = "pT217"

yscore3 = pd.DataFrame(yscore3)
yscore3.name = "pT217 + age"

yscore4 = pd.DataFrame(yscore4)
yscore4.name = "pT217 + sex"

yscore5 = pd.DataFrame(yscore5)
yscore5.name = "pT217 + APOE"

yscore6 = pd.DataFrame(yscore6)
yscore6.name = "full model"
y1 = pd.DataFrame(y1)


figure = plt.figure()
ROC_plot(yscore1, y1, figure, color = "#e377c2")
ROC_plot(yscore2, y1, figure, color = "#ff7f0e")
ROC_plot(yscore3, y1, figure, color = "#2ca02c")
ROC_plot(yscore4, y1, figure, color = "#d62728")
ROC_plot(yscore5, y1, figure, color = "#9467bd")
ROC_plot(yscore6, y1, figure, color = "#8c564b")
plt.gcf().set_size_inches(6, 4)  # Adjust the figure size as needed


# add asterisk
plt.text(0.19, -0.18, f'{asterisk_yscore2}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=26, color = 'red')
plt.text(0.19, -0.26, f'{asterisk_yscore3}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.34, f'{asterisk_yscore4}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.42, f'{asterisk_yscore5}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.50, f'{asterisk_yscore6}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
# plt.savefig('/ROC_0vs14_CI.PNG', dpi=300, bbox_inches='tight')


#### 0-2 vs 3-4 ---------------------------------------------------------------
y1 = []
y1 = np.where((Med["Profile_newer"]=="A-T-") | (Med["Profile_newer"]=="A+T-") | (Med["Profile_newer"]=="A+MTL"), 0, 1)

""" for logistic regression y should be array
Don't convert to dataframe"""

# y scores  
Log_Reg = []

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X1, y1) 
yscore1 = Log_Reg.predict_proba(X1)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X2, y1) 
yscore2 = Log_Reg.predict_proba(X2)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X3, y1) 
yscore3 = Log_Reg.predict_proba(X3)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X4, y1) 
yscore4 = Log_Reg.predict_proba(X4)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X5, y1) 
yscore5 = Log_Reg.predict_proba(X5)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X6, y1) 
yscore6 = Log_Reg.predict_proba(X6)[:,1]

# AUC statistical differences  -------------------------------------------------
""" 10 to the power of delong is to inverse the log10 of the pvalue"""
""" multiplication by 5 is to adjust for multiple comparisons"""

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore1, yscore2)
print("Adjusted p-value for the difference between base_model and pT217: {}".format(p_value*5))
asterisk_yscore2 = get_asterisks(p_value*5)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore3)
print("Adjusted p-value for the difference between pT217 and pT217_age: {}".format(p_value*5))
asterisk_yscore3 = get_hashtag(p_value*5)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore4)
print("Adjusted p-value for the difference between pT217 and pT217_gender: {}".format(p_value*5))
asterisk_yscore4 = get_hashtag(p_value*5)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore5)
print("Adjusted p-value for the difference between pT217 and pT217_APOE: {}".format(p_value*5))
asterisk_yscore5 = get_hashtag(p_value*5)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore6)
print("Adjusted p-value for the difference between pT217 and full_model: {}".format(p_value*5))
asterisk_yscore6 = get_hashtag(p_value*5)

# Plot 
""" for logistic regression y was array.
Here for ROC-plot we need to convert to dataframe"""

yscore1 = pd.DataFrame(yscore1)
yscore1.name = "base_model"

yscore2 = pd.DataFrame(yscore2)
yscore2.name = "pT217"

yscore3 = pd.DataFrame(yscore3)
yscore3.name = "pT217 + age"

yscore4 = pd.DataFrame(yscore4)
yscore4.name = "pT217 + sex"

yscore5 = pd.DataFrame(yscore5)
yscore5.name = "pT217 + APOE"

yscore6 = pd.DataFrame(yscore6)
yscore6.name = "full model"

y1 = pd.DataFrame(y1)


figure = plt.figure()
ROC_plot(yscore1, y1, figure, color = "#e377c2")
ROC_plot(yscore2, y1, figure, color = "#ff7f0e")
ROC_plot(yscore3, y1, figure, color = "#2ca02c")
ROC_plot(yscore4, y1, figure, color = "#d62728")
ROC_plot(yscore5, y1, figure, color = "#9467bd")
ROC_plot(yscore6, y1, figure, color = "#8c564b")
plt.gcf().set_size_inches(6, 4)  # Adjust the figure size as needed

# add asterisk
plt.text(0.19, -0.18, f'{asterisk_yscore2}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=26, color = 'red')
plt.text(0.19, -0.26, f'{asterisk_yscore3}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.34, f'{asterisk_yscore4}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.42, f'{asterisk_yscore5}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.50, f'{asterisk_yscore6}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
# plt.savefig('/ROC_02vs34_CI.PNG', dpi=300, bbox_inches='tight')


#### 0-3 vs 4 -----------------------------------------------------------------
y1 = []
y1 = np.where(Med["Profile_newer"]=="A+HIGH", 1, 0)

""" for logistic regression y should be array
Don't convert to dataframe"""

# y scores  
Log_Reg = []

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X1, y1) 
yscore1 = Log_Reg.predict_proba(X1)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X2, y1) 
yscore2 = Log_Reg.predict_proba(X2)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X3, y1) 
yscore3 = Log_Reg.predict_proba(X3)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X4, y1) 
yscore4 = Log_Reg.predict_proba(X4)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X5, y1) 
yscore5 = Log_Reg.predict_proba(X5)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X6, y1) 
yscore6 = Log_Reg.predict_proba(X6)[:,1]

# AUC statistical differences  -------------------------------------------------
""" 10 to the power of delong is to inverse the log10 of the pvalue"""
""" multiplication by 5 is to adjust for multiple comparisons"""

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore1, yscore2)
print("Adjusted p-value for the difference between base_model and pT217: {}".format(p_value*5))
asterisk_yscore2 = get_asterisks(p_value*5)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore3)
print("Adjusted p-value for the difference between pT217 and pT217_age: {}".format(p_value*5))
asterisk_yscore3 = get_hashtag(p_value*5)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore4)
print("Adjusted p-value for the difference between pT217 and pT217_gender: {}".format(p_value*5))
asterisk_yscore4 = get_hashtag(p_value*5)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore5)
print("Adjusted p-value for the difference between pT217 and pT217_APOE: {}".format(p_value*5))
asterisk_yscore5 = get_hashtag(p_value*5)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore6)
print("Adjusted p-value for the difference between pT217 and full_model: {}".format(p_value*5))
asterisk_yscore6 = get_hashtag(p_value*5)

# Plot 
""" for logistic regression y was array.
Here for ROC-plot we need to convert to dataframe"""

yscore1 = pd.DataFrame(yscore1)
yscore1.name = "base_model"

yscore2 = pd.DataFrame(yscore2)
yscore2.name = "pT217"

yscore3 = pd.DataFrame(yscore3)
yscore3.name = "pT217 + age"

yscore4 = pd.DataFrame(yscore4)
yscore4.name = "pT217 + sex"

yscore5 = pd.DataFrame(yscore5)
yscore5.name = "pT217 + APOE"

yscore6 = pd.DataFrame(yscore6)
yscore6.name = "full model"
y1 = pd.DataFrame(y1)


figure = plt.figure()
ROC_plot(yscore1, y1, figure, color = "#e377c2")
ROC_plot(yscore2, y1, figure, color = "#ff7f0e")
ROC_plot(yscore3, y1, figure, color = "#2ca02c")
ROC_plot(yscore4, y1, figure, color = "#d62728")
ROC_plot(yscore5, y1, figure, color = "#9467bd")
ROC_plot(yscore6, y1, figure, color = "#8c564b")
plt.gcf().set_size_inches(6, 4)  # Adjust the figure size as needed


# add asterisk
plt.text(0.19, -0.18, f'{asterisk_yscore2}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=26, color = 'red')
plt.text(0.19, -0.26, f'{asterisk_yscore3}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.34, f'{asterisk_yscore4}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.42, f'{asterisk_yscore5}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.50, f'{asterisk_yscore6}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
# plt.savefig('/ROC_03vs4_CI.PNG', dpi=300, bbox_inches='tight')

# plt.hist(Med[Med["0vs1_4"]==0]["pTau_217"], color = 'green', alpha = 0.3)
# plt.hist(Med[Med["0vs1_4"]==1]["pTau_217"], color = 'grey', alpha = 0.3)
# plt.legend([r'$A-T-$', r'$A+$'], fontsize=11, loc='upper right')
# plt.axvline(Youden_thresh(Med["pTau_217"], Med["0vs1_4"]), linestyle ="dotted", color='grey', alpha=0.9)



#%% Supp Figure 2a: ROC for All (with cognitive status (reviewer's req)): 
""" we repeat Fig 2a and 2b, only difference is, we add MMSE as per reviewer suggestion """

# 0 vs 1-3, 0-1 vs 2-3 and 0-2 vs 3 classification
Med = []
Med = Data[(Data[["Profile_newer", "pTau_217", "Tau_Age", "Gender", "APOE", "MMSE"]].notna()).all(axis=1)].reset_index(drop=True)

Med = Med.replace('Male', 0)
Med = Med.replace('Female', 1)

Med.APOE = Med.APOE.replace('E3_E3', 0)
Med.APOE = Med.APOE.replace('E3_E2', 0)
Med.APOE = Med.APOE.replace('E2_E3', 0)
Med.APOE = Med.APOE.replace('E2_E2', 0)
Med.APOE = Med.APOE.replace('E4_E4', 1)
Med.APOE = Med.APOE.replace('E4_E2', 1)
Med.APOE = Med.APOE.replace('E2_E4', 1)
Med.APOE = Med.APOE.replace('E4_E3', 1)
Med.APOE = Med.APOE.replace('E3_E4', 1)

# X 
X1 = Med[["Tau_Age", "Gender", "APOE", "MMSE"]]
X2 = Med[["pTau_217"]]
X3 = Med[["pTau_217", "Tau_Age"]]
X4 = Med[["pTau_217", "Gender"]]
X5 = Med[["pTau_217", "APOE"]]
X6 = Med[["pTau_217", "MMSE"]]
X7 = Med[["pTau_217", "Tau_Age", "Gender", "APOE", "MMSE"]]

#### 0 vs 1-4 -----------------------------------------------------------------
y1 = np.where(Med["Profile_newer"]=="A-T-", 0, 1)

""" for logistic regression y should be array
Don't convert to dataframe"""

# y scores  
Log_Reg = []

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X1, y1) 
yscore1 = Log_Reg.predict_proba(X1)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X2, y1) 
yscore2 = Log_Reg.predict_proba(X2)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X3, y1) 
yscore3 = Log_Reg.predict_proba(X3)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X4, y1) 
yscore4 = Log_Reg.predict_proba(X4)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X5, y1) 
yscore5 = Log_Reg.predict_proba(X5)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X6, y1) 
yscore6 = Log_Reg.predict_proba(X6)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X7, y1) 
yscore7 = Log_Reg.predict_proba(X7)[:,1]

# AUC statistical differences  -------------------------------------------------
""" 10 to the power of delong is to inverse the log10 of the pvalue"""
""" multiplication by 6 is to adjust for multiple comparisons"""

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore1, yscore2)
print("Adjusted p-value for the difference between base_model and pT217: {}".format(p_value*6))
# Determine asterisks based on p-value
asterisk_yscore2 = get_asterisks(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore3)
print("Adjusted p-value for the difference between pT217 and pT217_age: {}".format(p_value*6))
asterisk_yscore3 = get_hashtag(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore4)
print("Adjusted p-value for the difference between pT217 and pT217_gender: {}".format(p_value*6))
asterisk_yscore4 = get_hashtag(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore5)
print("Adjusted p-value for the difference between pT217 and pT217_APOE: {}".format(p_value*6))
asterisk_yscore5 = get_hashtag(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore6)
print("Adjusted p-value for the difference between pT217 and Dx: {}".format(p_value*6))
asterisk_yscore6 = get_hashtag(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore7)
print("Adjusted p-value for the difference between pT217 and full_model: {}".format(p_value*6))
asterisk_yscore7 = get_hashtag(p_value*6)

# Plot 
""" for logistic regression y was array.
Here for ROC-plot we need to convert to dataframe"""

yscore1 = pd.DataFrame(yscore1)
yscore1.name = "base_model"

yscore2 = pd.DataFrame(yscore2)
yscore2.name = "pT217"

yscore3 = pd.DataFrame(yscore3)
yscore3.name = "pT217 + age"

yscore4 = pd.DataFrame(yscore4)
yscore4.name = "pT217 + sex"

yscore5 = pd.DataFrame(yscore5)
yscore5.name = "pT217 + APOE"

yscore6 = pd.DataFrame(yscore6)
yscore6.name = "pT217 + MMSE"

yscore7 = pd.DataFrame(yscore7)
yscore7.name = "full model"

y1 = pd.DataFrame(y1)

figure = plt.figure()
ROC_plot(yscore1, y1, figure, color = "#e377c2")
ROC_plot(yscore2, y1, figure, color = "#ff7f0e")
ROC_plot(yscore3, y1, figure, color = "#2ca02c")
ROC_plot(yscore4, y1, figure, color = "#d62728")
ROC_plot(yscore5, y1, figure, color = "#9467bd")
ROC_plot(yscore6, y1, figure, color = "#8c564b")
ROC_plot(yscore7, y1, figure, color = "#0ABAB5")
plt.gcf().set_size_inches(6, 4)  # Adjust the figure size as needed

# add asterisk
plt.text(0.19, -0.18, f'{asterisk_yscore2}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=26, color = 'red')
plt.text(0.19, -0.26, f'{asterisk_yscore3}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.34, f'{asterisk_yscore4}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.42, f'{asterisk_yscore5}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.50, f'{asterisk_yscore6}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.58, f'{asterisk_yscore7}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
# plt.savefig('/ROC_0vs14_MMSE.PNG', dpi=300, bbox_inches='tight')


#### 0-2 vs 3-4 ---------------------------------------------------------------
y1 = []
y1 = np.where((Med["Profile_newer"]=="A-T-") | (Med["Profile_newer"]=="A+T-") | (Med["Profile_newer"]=="A+MTL"), 0, 1)

""" for logistic regression y should be array
Don't convert to dataframe"""

# y scores  
Log_Reg = []

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X1, y1) 
yscore1 = Log_Reg.predict_proba(X1)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X2, y1) 
yscore2 = Log_Reg.predict_proba(X2)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X3, y1) 
yscore3 = Log_Reg.predict_proba(X3)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X4, y1) 
yscore4 = Log_Reg.predict_proba(X4)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X5, y1) 
yscore5 = Log_Reg.predict_proba(X5)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X6, y1) 
yscore6 = Log_Reg.predict_proba(X6)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X7, y1) 
yscore7 = Log_Reg.predict_proba(X7)[:,1]

# AUC statistical differences  -------------------------------------------------
""" 10 to the power of delong is to inverse the log10 of the pvalue"""
""" multiplication by 6 is to adjust for multiple comparisons"""

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore1, yscore2)
print("Adjusted p-value for the difference between base_model and pT217: {}".format(p_value*6))
# Determine asterisks based on p-value
asterisk_yscore2 = get_asterisks(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore3)
print("Adjusted p-value for the difference between pT217 and pT217_age: {}".format(p_value*6))
asterisk_yscore3 = get_hashtag(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore4)
print("Adjusted p-value for the difference between pT217 and pT217_gender: {}".format(p_value*6))
asterisk_yscore4 = get_hashtag(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore5)
print("Adjusted p-value for the difference between pT217 and pT217_APOE: {}".format(p_value*6))
asterisk_yscore5 = get_hashtag(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore6)
print("Adjusted p-value for the difference between pT217 and Dx: {}".format(p_value*6))
asterisk_yscore6 = get_hashtag(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore7)
print("Adjusted p-value for the difference between pT217 and full_model: {}".format(p_value*6))
asterisk_yscore7 = get_hashtag(p_value*6)

# Plot 
""" for logistic regression y was array.
Here for ROC-plot we need to convert to dataframe"""

yscore1 = pd.DataFrame(yscore1)
yscore1.name = "base_model"

yscore2 = pd.DataFrame(yscore2)
yscore2.name = "pT217"

yscore3 = pd.DataFrame(yscore3)
yscore3.name = "pT217 + age"

yscore4 = pd.DataFrame(yscore4)
yscore4.name = "pT217 + sex"

yscore5 = pd.DataFrame(yscore5)
yscore5.name = "pT217 + APOE"

yscore6 = pd.DataFrame(yscore6)
yscore6.name = "pT217 + MMSE"

yscore7 = pd.DataFrame(yscore7)
yscore7.name = "full model"

y1 = pd.DataFrame(y1)

figure = plt.figure()
ROC_plot(yscore1, y1, figure, color = "#e377c2")
ROC_plot(yscore2, y1, figure, color = "#ff7f0e")
ROC_plot(yscore3, y1, figure, color = "#2ca02c")
ROC_plot(yscore4, y1, figure, color = "#d62728")
ROC_plot(yscore5, y1, figure, color = "#9467bd")
ROC_plot(yscore6, y1, figure, color = "#8c564b")
ROC_plot(yscore7, y1, figure, color = "#0ABAB5")
plt.gcf().set_size_inches(6, 4)  # Adjust the figure size as needed

# add asterisk
plt.text(0.19, -0.18, f'{asterisk_yscore2}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=26, color = 'red')
plt.text(0.19, -0.26, f'{asterisk_yscore3}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.34, f'{asterisk_yscore4}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.42, f'{asterisk_yscore5}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.50, f'{asterisk_yscore6}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.58, f'{asterisk_yscore7}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
# plt.savefig('/ROC_02vs34_MMSE.PNG', dpi=300, bbox_inches='tight')


#### 0-3 vs 4 -----------------------------------------------------------------
y1 = []
y1 = np.where(Med["Profile_newer"]=="A+HIGH", 1, 0)

""" for logistic regression y should be array
Don't convert to dataframe"""

# y scores  
Log_Reg = []

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X1, y1) 
yscore1 = Log_Reg.predict_proba(X1)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X2, y1) 
yscore2 = Log_Reg.predict_proba(X2)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X3, y1) 
yscore3 = Log_Reg.predict_proba(X3)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X4, y1) 
yscore4 = Log_Reg.predict_proba(X4)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X5, y1) 
yscore5 = Log_Reg.predict_proba(X5)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X6, y1) 
yscore6 = Log_Reg.predict_proba(X6)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X7, y1) 
yscore7 = Log_Reg.predict_proba(X7)[:,1]

# AUC statistical differences  -------------------------------------------------
""" 10 to the power of delong is to inverse the log10 of the pvalue"""
""" multiplication by 6 is to adjust for multiple comparisons"""

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore1, yscore2)
print("Adjusted p-value for the difference between base_model and pT217: {}".format(p_value*6))
# Determine asterisks based on p-value
asterisk_yscore2 = get_asterisks(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore3)
print("Adjusted p-value for the difference between pT217 and pT217_age: {}".format(p_value*6))
asterisk_yscore3 = get_hashtag(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore4)
print("Adjusted p-value for the difference between pT217 and pT217_gender: {}".format(p_value*6))
asterisk_yscore4 = get_hashtag(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore5)
print("Adjusted p-value for the difference between pT217 and pT217_APOE: {}".format(p_value*6))
asterisk_yscore5 = get_hashtag(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore6)
print("Adjusted p-value for the difference between pT217 and Dx: {}".format(p_value*6))
asterisk_yscore6 = get_hashtag(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore7)
print("Adjusted p-value for the difference between pT217 and full_model: {}".format(p_value*6))
asterisk_yscore7 = get_hashtag(p_value*6)

# Plot 
""" for logistic regression y was array.
Here for ROC-plot we need to convert to dataframe"""

yscore1 = pd.DataFrame(yscore1)
yscore1.name = "base_model"

yscore2 = pd.DataFrame(yscore2)
yscore2.name = "pT217"

yscore3 = pd.DataFrame(yscore3)
yscore3.name = "pT217 + age"

yscore4 = pd.DataFrame(yscore4)
yscore4.name = "pT217 + sex"

yscore5 = pd.DataFrame(yscore5)
yscore5.name = "pT217 + APOE"

yscore6 = pd.DataFrame(yscore6)
yscore6.name = "pT217 + MMSE"

yscore7 = pd.DataFrame(yscore7)
yscore7.name = "full model"

y1 = pd.DataFrame(y1)

figure = plt.figure()
ROC_plot(yscore1, y1, figure, color = "#e377c2")
ROC_plot(yscore2, y1, figure, color = "#ff7f0e")
ROC_plot(yscore3, y1, figure, color = "#2ca02c")
ROC_plot(yscore4, y1, figure, color = "#d62728")
ROC_plot(yscore5, y1, figure, color = "#9467bd")
ROC_plot(yscore6, y1, figure, color = "#8c564b")
ROC_plot(yscore7, y1, figure, color = "#0ABAB5")
plt.gcf().set_size_inches(6, 4)  # Adjust the figure size as needed

# add asterisk
plt.text(0.19, -0.18, f'{asterisk_yscore2}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=26, color = 'red')
plt.text(0.19, -0.26, f'{asterisk_yscore3}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.34, f'{asterisk_yscore4}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.42, f'{asterisk_yscore5}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.50, f'{asterisk_yscore6}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.58, f'{asterisk_yscore7}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
# plt.savefig('/ROC_03vs4_MMSE.PNG', dpi=300, bbox_inches='tight')



#%% Supp Figure 2b: ROC for CI only (with cognitive status (reviewer's req)): 
# 0 vs 1-3, 0-1 vs 2-3 and 0-2 vs 3 classification
Med = []
Med = Data[(Data[["Profile_newer", "pTau_217", "Tau_Age", "Gender", "APOE", "MMSE"]].notna()).all(axis=1)].reset_index(drop=True)
Med = Med[Med.simple_Diag.isin(["MCI", "AD"])].reset_index(drop=True)

Med = Med.replace('Male', 0)
Med = Med.replace('Female', 1)

Med.APOE = Med.APOE.replace('E3_E3', 0)
Med.APOE = Med.APOE.replace('E3_E2', 0)
Med.APOE = Med.APOE.replace('E2_E3', 0)
Med.APOE = Med.APOE.replace('E2_E2', 0)
Med.APOE = Med.APOE.replace('E4_E4', 1)
Med.APOE = Med.APOE.replace('E4_E2', 1)
Med.APOE = Med.APOE.replace('E2_E4', 1)
Med.APOE = Med.APOE.replace('E4_E3', 1)
Med.APOE = Med.APOE.replace('E3_E4', 1)

# X 
X1 = Med[["Tau_Age", "Gender", "APOE", "MMSE"]]
X2 = Med[["pTau_217"]]
X3 = Med[["pTau_217", "Tau_Age"]]
X4 = Med[["pTau_217", "Gender"]]
X5 = Med[["pTau_217", "APOE"]]
X6 = Med[["pTau_217", "MMSE"]]
X7 = Med[["pTau_217", "Tau_Age", "Gender", "APOE", "MMSE"]]


#### 0 vs 1-4 -----------------------------------------------------------------
y1 = np.where(Med["Profile_newer"]=="A-T-", 0, 1)

""" for logistic regression y should be array
Don't convert to dataframe"""

# y scores  
Log_Reg = []

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X1, y1) 
yscore1 = Log_Reg.predict_proba(X1)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X2, y1) 
yscore2 = Log_Reg.predict_proba(X2)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X3, y1) 
yscore3 = Log_Reg.predict_proba(X3)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X4, y1) 
yscore4 = Log_Reg.predict_proba(X4)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X5, y1) 
yscore5 = Log_Reg.predict_proba(X5)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X6, y1) 
yscore6 = Log_Reg.predict_proba(X6)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X7, y1) 
yscore7 = Log_Reg.predict_proba(X7)[:,1]

# AUC statistical differences  -------------------------------------------------
""" 10 to the power of delong is to inverse the log10 of the pvalue"""
""" multiplication by 6 is to adjust for multiple comparisons"""

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore1, yscore2)
print("Adjusted p-value for the difference between base_model and pT217: {}".format(p_value*6))
# Determine asterisks based on p-value
asterisk_yscore2 = get_asterisks(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore3)
print("Adjusted p-value for the difference between pT217 and pT217_age: {}".format(p_value*6))
asterisk_yscore3 = get_hashtag(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore4)
print("Adjusted p-value for the difference between pT217 and pT217_gender: {}".format(p_value*6))
asterisk_yscore4 = get_hashtag(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore5)
print("Adjusted p-value for the difference between pT217 and pT217_APOE: {}".format(p_value*6))
asterisk_yscore5 = get_hashtag(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore6)
print("Adjusted p-value for the difference between pT217 and Dx: {}".format(p_value*6))
asterisk_yscore6 = get_hashtag(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore7)
print("Adjusted p-value for the difference between pT217 and full_model: {}".format(p_value*6))
asterisk_yscore7 = get_hashtag(p_value*6)

# Plot 
""" for logistic regression y was array.
Here for ROC-plot we need to convert to dataframe"""

yscore1 = pd.DataFrame(yscore1)
yscore1.name = "base_model"

yscore2 = pd.DataFrame(yscore2)
yscore2.name = "pT217"

yscore3 = pd.DataFrame(yscore3)
yscore3.name = "pT217 + age"

yscore4 = pd.DataFrame(yscore4)
yscore4.name = "pT217 + sex"

yscore5 = pd.DataFrame(yscore5)
yscore5.name = "pT217 + APOE"

yscore6 = pd.DataFrame(yscore6)
yscore6.name = "pT217 + MMSE"

yscore7 = pd.DataFrame(yscore7)
yscore7.name = "full model"

y1 = pd.DataFrame(y1)

figure = plt.figure()
ROC_plot(yscore1, y1, figure, color = "#e377c2")
ROC_plot(yscore2, y1, figure, color = "#ff7f0e")
ROC_plot(yscore3, y1, figure, color = "#2ca02c")
ROC_plot(yscore4, y1, figure, color = "#d62728")
ROC_plot(yscore5, y1, figure, color = "#9467bd")
ROC_plot(yscore6, y1, figure, color = "#8c564b")
ROC_plot(yscore7, y1, figure, color = "#0ABAB5")
plt.gcf().set_size_inches(6, 4)  # Adjust the figure size as needed

# add asterisk
plt.text(0.19, -0.18, f'{asterisk_yscore2}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=26, color = 'red')
plt.text(0.19, -0.26, f'{asterisk_yscore3}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.34, f'{asterisk_yscore4}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.42, f'{asterisk_yscore5}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.50, f'{asterisk_yscore6}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.58, f'{asterisk_yscore7}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
# plt.savefig('/ROC_0vs14_MMSE_CI.PNG', dpi=300, bbox_inches='tight')


#### 0-2 vs 3-4 ---------------------------------------------------------------
y1 = []
y1 = np.where((Med["Profile_newer"]=="A-T-") | (Med["Profile_newer"]=="A+T-") | (Med["Profile_newer"]=="A+MTL"), 0, 1)

""" for logistic regression y should be array
Don't convert to dataframe"""

# y scores  
Log_Reg = []

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X1, y1) 
yscore1 = Log_Reg.predict_proba(X1)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X2, y1) 
yscore2 = Log_Reg.predict_proba(X2)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X3, y1) 
yscore3 = Log_Reg.predict_proba(X3)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X4, y1) 
yscore4 = Log_Reg.predict_proba(X4)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X5, y1) 
yscore5 = Log_Reg.predict_proba(X5)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X6, y1) 
yscore6 = Log_Reg.predict_proba(X6)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X7, y1) 
yscore7 = Log_Reg.predict_proba(X7)[:,1]

# AUC statistical differences  -------------------------------------------------
""" 10 to the power of delong is to inverse the log10 of the pvalue"""
""" multiplication by 6 is to adjust for multiple comparisons"""

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore1, yscore2)
print("Adjusted p-value for the difference between base_model and pT217: {}".format(p_value*6))
# Determine asterisks based on p-value
asterisk_yscore2 = get_asterisks(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore3)
print("Adjusted p-value for the difference between pT217 and pT217_age: {}".format(p_value*6))
asterisk_yscore3 = get_hashtag(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore4)
print("Adjusted p-value for the difference between pT217 and pT217_gender: {}".format(p_value*6))
asterisk_yscore4 = get_hashtag(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore5)
print("Adjusted p-value for the difference between pT217 and pT217_APOE: {}".format(p_value*6))
asterisk_yscore5 = get_hashtag(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore6)
print("Adjusted p-value for the difference between pT217 and Dx: {}".format(p_value*6))
asterisk_yscore6 = get_hashtag(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore7)
print("Adjusted p-value for the difference between pT217 and full_model: {}".format(p_value*6))
asterisk_yscore7 = get_hashtag(p_value*6)

# Plot 
""" for logistic regression y was array.
Here for ROC-plot we need to convert to dataframe"""

yscore1 = pd.DataFrame(yscore1)
yscore1.name = "base_model"

yscore2 = pd.DataFrame(yscore2)
yscore2.name = "pT217"

yscore3 = pd.DataFrame(yscore3)
yscore3.name = "pT217 + age"

yscore4 = pd.DataFrame(yscore4)
yscore4.name = "pT217 + sex"

yscore5 = pd.DataFrame(yscore5)
yscore5.name = "pT217 + APOE"

yscore6 = pd.DataFrame(yscore6)
yscore6.name = "pT217 + MMSE"

yscore7 = pd.DataFrame(yscore7)
yscore7.name = "full model"

y1 = pd.DataFrame(y1)

figure = plt.figure()
ROC_plot(yscore1, y1, figure, color = "#e377c2")
ROC_plot(yscore2, y1, figure, color = "#ff7f0e")
ROC_plot(yscore3, y1, figure, color = "#2ca02c")
ROC_plot(yscore4, y1, figure, color = "#d62728")
ROC_plot(yscore5, y1, figure, color = "#9467bd")
ROC_plot(yscore6, y1, figure, color = "#8c564b")
ROC_plot(yscore7, y1, figure, color = "#0ABAB5")
plt.gcf().set_size_inches(6, 4)  # Adjust the figure size as needed

# add asterisk
plt.text(0.19, -0.18, f'{asterisk_yscore2}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=26, color = 'red')
plt.text(0.19, -0.26, f'{asterisk_yscore3}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.34, f'{asterisk_yscore4}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.42, f'{asterisk_yscore5}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.50, f'{asterisk_yscore6}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.58, f'{asterisk_yscore7}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
# plt.savefig('/ROC_02vs34_MMSE_CI.PNG', dpi=300, bbox_inches='tight')


#### 0-3 vs 4 -----------------------------------------------------------------
y1 = []
y1 = np.where(Med["Profile_newer"]=="A+HIGH", 1, 0)

""" for logistic regression y should be array
Don't convert to dataframe"""

# y scores  
Log_Reg = []

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X1, y1) 
yscore1 = Log_Reg.predict_proba(X1)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X2, y1) 
yscore2 = Log_Reg.predict_proba(X2)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X3, y1) 
yscore3 = Log_Reg.predict_proba(X3)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X4, y1) 
yscore4 = Log_Reg.predict_proba(X4)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X5, y1) 
yscore5 = Log_Reg.predict_proba(X5)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X6, y1) 
yscore6 = Log_Reg.predict_proba(X6)[:,1]

Log_Reg = LogisticRegression(random_state=0, solver = 'lbfgs').fit(X7, y1) 
yscore7 = Log_Reg.predict_proba(X7)[:,1]

# AUC statistical differences  -------------------------------------------------
""" 10 to the power of delong is to inverse the log10 of the pvalue"""
""" multiplication by 6 is to adjust for multiple comparisons"""

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore1, yscore2)
print("Adjusted p-value for the difference between base_model and pT217: {}".format(p_value*6))
# Determine asterisks based on p-value
asterisk_yscore2 = get_asterisks(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore3)
print("Adjusted p-value for the difference between pT217 and pT217_age: {}".format(p_value*6))
asterisk_yscore3 = get_hashtag(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore4)
print("Adjusted p-value for the difference between pT217 and pT217_gender: {}".format(p_value*6))
asterisk_yscore4 = get_hashtag(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore5)
print("Adjusted p-value for the difference between pT217 and pT217_APOE: {}".format(p_value*6))
asterisk_yscore5 = get_hashtag(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore6)
print("Adjusted p-value for the difference between pT217 and Dx: {}".format(p_value*6))
asterisk_yscore6 = get_hashtag(p_value*6)

p_value = 10 ** compare_auc_delong_xu.delong_roc_test(y1, yscore2, yscore7)
print("Adjusted p-value for the difference between pT217 and full_model: {}".format(p_value*6))
asterisk_yscore7 = get_hashtag(p_value*6)

# Plot 
""" for logistic regression y was array.
Here for ROC-plot we need to convert to dataframe"""

yscore1 = pd.DataFrame(yscore1)
yscore1.name = "base_model"

yscore2 = pd.DataFrame(yscore2)
yscore2.name = "pT217"

yscore3 = pd.DataFrame(yscore3)
yscore3.name = "pT217 + age"

yscore4 = pd.DataFrame(yscore4)
yscore4.name = "pT217 + sex"

yscore5 = pd.DataFrame(yscore5)
yscore5.name = "pT217 + APOE"

yscore6 = pd.DataFrame(yscore6)
yscore6.name = "pT217 + MMSE"

yscore7 = pd.DataFrame(yscore7)
yscore7.name = "full model"

y1 = pd.DataFrame(y1)

figure = plt.figure()
ROC_plot(yscore1, y1, figure, color = "#e377c2")
ROC_plot(yscore2, y1, figure, color = "#ff7f0e")
ROC_plot(yscore3, y1, figure, color = "#2ca02c")
ROC_plot(yscore4, y1, figure, color = "#d62728")
ROC_plot(yscore5, y1, figure, color = "#9467bd")
ROC_plot(yscore6, y1, figure, color = "#8c564b")
ROC_plot(yscore7, y1, figure, color = "#0ABAB5")
plt.gcf().set_size_inches(6, 4)  # Adjust the figure size as needed

# add asterisk
plt.text(0.19, -0.18, f'{asterisk_yscore2}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=26, color = 'red')
plt.text(0.19, -0.26, f'{asterisk_yscore3}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.34, f'{asterisk_yscore4}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.42, f'{asterisk_yscore5}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.50, f'{asterisk_yscore6}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
plt.text(0.19, -0.58, f'{asterisk_yscore7}', ha='right', va='center', transform=plt.gcf().transFigure, fontsize=21, color = 'red')
# plt.savefig('/ROC_03vs4_MMSE_CI.PNG', dpi=300, bbox_inches='tight')



#%% [NOT USED IN PAPER]: ad-hoc Figures for Supp Table 6: 0 vs 1, 1 vs 2, 2 vs 3 for All
Med = []
Med = Data[(Data[["Profile_newer", "pTau_217", "Tau_Age", "Gender", "Composite_memory", "MMSE"]].notna()).all(axis=1) & (Data.Centiloid>=Estab_thresh_A)].reset_index(drop=True)

Med = Med.replace('Male', 0)
Med = Med.replace('Female', 1)


# 0 vs 1 -----------------------------------------------------------
sub=[]
# sub = Med[(Med.Profile_newer=="A+T-") | (Med.Profile_newer=="A+MTL")]
# sub = Med[(Med.Profile_newer=="A+MTL") | (Med.Profile_newer=="A+MOD")]
sub = Med[(Med.Profile_newer=="A+MOD") | (Med.Profile_newer=="A+HIGH")]


# classify 
X1 = sub[["pTau_217"]]
X2 = sub[["pTau_217", "Tau_Age"]]
X3 = sub[["pTau_217", "Gender"]]
# X4 = sub[["pTau_217", "Composite_memory"]]
X4 = sub[["pTau_217", "MMSE"]]
# X5 = sub[["pTau_217", "simple_Diag"]]
# X6 = sub[["pTau_217", "Tau_Age", "Gender", "Composite_memory"]]
X6 = sub[["pTau_217", "Tau_Age", "Gender", "MMSE"]]

# y1 = np.where(sub["Profile_newer"]=="A+T-", 0, 1)
# y1 = np.where(sub["Profile_newer"]=="A+MTL", 0, 1)
y1 = np.where(sub["Profile_newer"]=="A+MOD", 0, 1)

""" for logistic regression y should be array
Don't convert to dataframe"""

# y scores
Log_Reg = []
Log_Reg = LogisticRegression(random_state=0).fit(X1, y1) 
yscore1 = pd.DataFrame(Log_Reg.predict_proba(X1)[:,1])
yscore1.name = "pT217"

Log_Reg = []
Log_Reg = LogisticRegression(random_state=0).fit(X2, y1) 
yscore2 = pd.DataFrame(Log_Reg.predict_proba(X2)[:,1])
yscore2.name = "pT217 + Age"

Log_Reg = []
Log_Reg = LogisticRegression(random_state=0).fit(X3, y1) 
yscore3 = pd.DataFrame(Log_Reg.predict_proba(X3)[:,1])
yscore3.name = "pT217 + Gender"

Log_Reg = []
Log_Reg = LogisticRegression(random_state=0).fit(X4, y1) 
yscore4 = pd.DataFrame(Log_Reg.predict_proba(X4)[:,1])
# yscore4.name = "pT217 + Memory"
yscore4.name = "pT217 + MMSE"

# Log_Reg = []
# Log_Reg = LogisticRegression(random_state=0).fit(X5, y1) 
# yscore5 = pd.DataFrame(Log_Reg.predict_proba(X5)[:,1])
# # yscore4.name = "pT217 + Comp. Memory"
# yscore5.name = "pT217 + Dx"

Log_Reg = []
Log_Reg = LogisticRegression(random_state=0).fit(X6, y1) 
yscore6 = pd.DataFrame(Log_Reg.predict_proba(X6)[:,1])
yscore6.name = "full model"

num_colors = 6
cmap = plt.get_cmap('plasma') 
colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]


y1 = pd.DataFrame(y1)
# y1.name = "0 vs 1-3"
# y1.name = "0-1 vs 2-3"
# y1.name = "0-2 vs 3"
""" for logistic regression y was array.
Here for ROC-plot we need to convert to dataframe"""

figure = plt.figure()
ROC_plot(yscore1, y1, figure, colors[0])
ROC_plot(yscore2, y1, figure, colors[1])
ROC_plot(yscore3, y1, figure, colors[2])
ROC_plot(yscore4, y1, figure, colors[3])
# ROC_plot(yscore5, y1, figure, colors[4])
ROC_plot(yscore6, y1, figure, colors[5])

# plt.savefig('/ /Figures/ROC_0vs1.PNG', dpi=300, bbox_inches='tight')
# plt.savefig('/ /Figures/ROC_1vs2.PNG', dpi=300, bbox_inches='tight')
# plt.savefig('/ /Figures/ROC_2vs3.PNG', dpi=300, bbox_inches='tight')


#%% Supp Table 6: 0 vs 1, 1 vs 2, 2 vs 3 for All
Med = []
Med = Data[(Data[["Profile_newer"]].notna()).all(axis=1) & (Data.Centiloid>=Estab_thresh_A)].reset_index(drop=True)

sub1 = Med[(Med.Profile_newer=="A+T-") | (Med.Profile_newer=="A+MTL")].reset_index(drop=True)
sub2 = Med[(Med.Profile_newer=="A+MTL") | (Med.Profile_newer=="A+MOD")].reset_index(drop=True)
sub3 = Med[(Med.Profile_newer=="A+MOD") | (Med.Profile_newer=="A+HIGH")].reset_index(drop=True)

# Binarize the profiles
sub1['OnevsOther'] = np.where(sub1['Profile_newer'] == 'A+T-', 0, 1)
sub2['OnevsOther'] = np.where(sub2["Profile_newer"]=="A+MTL", 0, 1)
sub3['OnevsOther'] = np.where(sub3["Profile_newer"]=="A+MOD", 0, 1)


Comp1 = pd.DataFrame(ci_bootstraps(sub1['pTau_217'], sub1['OnevsOther'], alpha=0.95, n_bootstraps=1000, rng_seed=42)).T
Comp1.rename(columns={0:"AUC", 1:'Youden threshold', 2:"Sensitivity", 3:"Specificity", 4:"PPV", 5:"NPV"}, inplace=True)


Comp2 = pd.DataFrame(ci_bootstraps(sub2['pTau_217'], sub2['OnevsOther'], alpha=0.95, n_bootstraps=1000, rng_seed=42)).T
Comp2.rename(columns={0:"AUC", 1:'Youden threshold', 2:"Sensitivity", 3:"Specificity", 4:"PPV", 5:"NPV"}, inplace=True)


Comp3 = pd.DataFrame(ci_bootstraps(sub3['pTau_217'], sub3['OnevsOther'], alpha=0.95, n_bootstraps=1000, rng_seed=42)).T
Comp3.rename(columns={0:"AUC", 1:'Youden threshold', 2:"Sensitivity", 3:"Specificity", 4:"PPV", 5:"NPV"}, inplace=True)


Comp = pd.concat([Comp1, Comp2, Comp3])
# Comp.to_csv("/SuppT6_All.csv")




