import os
import sys
import imp
import pandas as pd
import numpy as np
from joblib import load, dump
from src import utils as ut

from sklearn.impute import SimpleImputer
from sklearn.decomposition import IncrementalPCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler


def get_email_based_features(dftrain, dftest):
    """
    Create email based features for feature set FS1 
    """
    dfin_dict = {'train': dftrain, 'test': dftest}
    dfout_dict = {}
    
    for dataset in dfin_dict:
        
        dfin = dfin_dict[dataset]
    
        dfout = pd.DataFrame(index=dfin.index)

        dfout['P_emaildomain_company'] = dfin['P_emaildomain'].str.split('.') \
                            .apply(lambda x: x[0] if x is not None else "NA" )

        dfout['R_emaildomain_company'] = dfin['R_emaildomain'].str.split('.') \
                            .apply(lambda x: x[0] if x is not None else "NA" )

        dfout['P_emaildomain_tld'] = dfin['P_emaildomain'].str.split('.') \
                            .apply(lambda x: x[-1] if x is not None else "NA" )

        dfout['R_emaildomain_tld'] = dfin['R_emaildomain'].str.split('.') \
                            .apply(lambda x: x[-1] if x is not None else "NA" )

        dfout['P_emailco-R_emailco'] = dfout['P_emaildomain_company'].astype(str) \
                            + '-' + dfout['R_emaildomain_company'].astype(str)

        dfout['P_emailtld-R_emailtld'] = dfout['P_emaildomain_tld'].astype(str) \
                            + '-' + dfout['R_emaildomain_tld'].astype(str)
        
        dfout_dict[dataset] = dfout
        
    # Frequency encode columns
    col_list = ['P_emaildomain_company', 'R_emaildomain_company', 'P_emailco-R_emailco' ]
    dfout_dict['train'], dfout_dict['test'] =  ut.frequency_encode(dfout_dict['train'], 
                                                                   dfout_dict['test'], 
                                                                   col_list)
    
    return dfout_dict['train'], dfout_dict['test']



def get_device_based_features(dftrain, dftest):
    """
    Create device based features for feature set FS1 
    """
    dfin_dict = {'train': dftrain, 'test': dftest}
    dfout_dict = {}
    ksme_dict = {}
    
    for dataset in dfin_dict:
        
        dfin = dfin_dict[dataset]
    
        dfout = pd.DataFrame(index=dfin.index)


        # Device Build
        dfout['Build'] = dfin['DeviceInfo'].fillna('NA').str.split(r'Build\/') \
                                        .apply(lambda x: x[-1] if len(x) > 1 else "NA" ) \
                                        .str.strip()

        # Device Brand if available
        dfout['StatedBrand'] =  dfin['DeviceInfo'].str.extract(r'^([aA-zZ]+)\b').loc[:,0] \
                                        .str.lower().str.strip().fillna('NA')


        # Create interaction feature and mean encode
        dfout['DeviceFeaturesInt'] = ut.create_category_interactions(dfin, ['DeviceType','DeviceInfo'])

        dfout_dict[dataset] = dfout
        
        
    # Frequency encode columns
    col_list = ['DeviceFeaturesInt', 'StatedBrand']
    dfout_dict['train'], dfout_dict['test'] =  ut.frequency_encode(dfout_dict['train'], 
                                                                   dfout_dict['test'], 
                                                                   col_list)
        
        
    return dfout_dict['train'], dfout_dict['test']


def get_dist_based_features(dftrain, dftest):
    """
    Get dist variable based features
    """
    
    dfin_dict = {'train': dftrain, 'test': dftest}
    dfout_dict = {}
    
    dftemp =  pd.concat([dftrain.loc[:,['dist1', 'dist2']], dftest.loc[:,['dist1', 'dist2']]], axis=0)
    
    qt_dist1 = QuantileTransformer()
    qt_dist1.fit(dftemp['dist1'].values.reshape(-1,1))
    
    qt_dist2 = QuantileTransformer()
    qt_dist2.fit(dftemp['dist2'].values.reshape(-1,1))
    
    for dataset in dfin_dict:
        
        dfin = dfin_dict[dataset]
    
        dfout = pd.DataFrame(index=dfin.index)
        
        dfout['dist1_by_dist2'] = (dfin['dist1'].fillna(-99999)/(1.0 + dfin['dist2'].fillna(99999)))
        
       
        dfout['quantile_dist1'] = pd.Series(qt_dist1.transform(dfin['dist1'].values.reshape(-1,1)) \
                                            .reshape(-1)).fillna(-9999)
        
        dfout['quantile_dist2'] = pd.Series(qt_dist2.transform(dfin['dist2'].values.reshape(-1,1)) \
                                            .reshape(-1)).fillna(-9999)
        
        
        dfout['addr1_grouped_quantile_dist1'] = dfin.fillna({'addr1':-9999}).groupby('addr1') \
                                                    ['dist1'].transform(ut.pandas_group_quantile_transform) \
                                                            .fillna(-9999)
        
        dfout['addr1_grouped_quantile_dist2'] = dfin.fillna({'addr1':-9999}).groupby('addr1') \
                                                    ['dist2'].transform(ut.pandas_group_quantile_transform) \
                                                            .fillna(-9999)
        
       
        dfout_dict[dataset] = dfout
        
    return dfout_dict['train'], dfout_dict['test']
    
    
def get_misc_interaction_features(dftrain, dftest):
    """
    Get miscellaneous interaction based features
    """
    
    dfin_dict = {'train': dftrain, 'test': dftest}
    dfout_dict = {}
    
    for dataset in dfin_dict:
        
        dfin = dfin_dict[dataset]
    
        dfout = pd.DataFrame(index=dfin.index)
        
        dfout['addr2-addr1'] = ut.create_category_interactions(dfin, ['addr2','addr1'])
        
        dfout['card4-card6'] = ut.create_category_interactions(dfin, ['card4','card6'])
        
        dfout['product-card4-card6'] = ut.create_category_interactions(dfin, 
                                                    ['ProductCD','card4','card6'])
        
        dfout['R_emaildomain-P_emaildomain'] = ut.create_category_interactions(dfin, 
                                                    ['R_emaildomain','P_emaildomain'])
        
        dfout['addr2-R_emaildomain-P_emaildomain'] = ut.create_category_interactions(dfin,
                                                       ['addr2','R_emaildomain','P_emaildomain'])
        
        
        dfout_dict[dataset] = dfout
        
    # Frequency encode columns
    col_list = ['addr2-R_emaildomain-P_emaildomain', 'product-card4-card6', 'addr2-addr1']
    dfout_dict['train'], dfout_dict['test'] =  ut.frequency_encode(dfout_dict['train'], 
                                                                   dfout_dict['test'], 
                                                                   col_list)
        
    return dfout_dict['train'], dfout_dict['test']



def get_proxyid1_based_features(dftrain, dftest):
    "proxyid1 based features"
    
    dfin_dict = {'train': dftrain, 'test': dftest}
    dfout_dict = {}
    ksme_dict = {}
    
    for dataset in dfin_dict:
        
        dfin = dfin_dict[dataset]
    
        dfout = pd.DataFrame(index=dfin.index)
        
        id_key_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']
        
        na_filler = dict(zip(id_key_cols,["NA"]*len(id_key_cols)))
        
        # group object by id_key_cols
        grouped_by_id = dfin.fillna( na_filler ) \
                        .groupby(id_key_cols)
        
        # Feature 1: unique P_emaildomain - R_emaildomain ratio
        P_email_unique = grouped_by_id['P_emaildomain']  \
                        .apply(lambda x: x.unique().shape[0]) \
                        .rename('P_email_unique')
        
        R_email_unique = grouped_by_id['R_emaildomain']  \
                        .apply(lambda x: x.unique().shape[0]) \
                        .rename('R_email_unique')
        
        
        P_R_email_ratio = (P_email_unique/R_email_unique).rename('P_R_email_ratio')
        
        
        # Feature 2: unique P_emaildomain - R_emaildomain ratio is fractional (bool)
        P_R_email_ratio_isfrac = ((P_R_email_ratio %1 ) > 0).rename('P_R_email_ratio_isfrac')
        
        # Feature 3: transactions per card group
        tx_per_card_group = grouped_by_id.size().rename('tx_per_card_group')
        
        # create a single data frame 
        dfgrouped = pd.concat([P_email_unique, R_email_unique,
                               P_R_email_ratio,
                               P_R_email_ratio_isfrac,
                               tx_per_card_group], axis=1)
        
        new_features_list = ['P_email_unique', 'R_email_unique', 'P_R_email_ratio',
                             'P_R_email_ratio_isfrac', 'tx_per_card_group']
        
        dftemp = pd.merge(dfin.fillna(na_filler), 
                          dfgrouped, left_on = id_key_cols, right_index=True) \
                          .loc[:,new_features_list]
        
        dfout = pd.concat([dfout, dftemp], axis=1)
        
        dfout['card_group_int'] = ut.create_category_interactions(dfin, id_key_cols)

        dfout_dict[dataset] = dfout
        
    # Frequency encode columns
    col_list = ['card_group_int']
    dfout_dict['train'], dfout_dict['test'] =  ut.frequency_encode(dfout_dict['train'], 
                                                                   dfout_dict['test'], 
                                                                   col_list)
        
    
    return dfout_dict['train'], dfout_dict['test']
    
    
def get_proxyid2_based_features(dftrain, dftest):
    "proxyid2 based features"
    
    dfin_dict = {'train': dftrain, 'test': dftest}
    dfout_dict = {}
    
    for dataset in dfin_dict:
        
        dfin = dfin_dict[dataset]
    
        dfout = pd.DataFrame(index=dfin.index)
        
        id_key_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 
                       'addr2','addr1', 'P_emaildomain', 'R_emaildomain']
        
        # group object by id_key_cols
        grouped_by_id = dfin.fillna( dict(zip(id_key_cols,["NA"]*len(id_key_cols))) ) \
                        .groupby(id_key_cols)
        
        
        # group internal index - group-size index.
        proxyid2_gii_tx_ratio = grouped_by_id['dist1'] \
                                        .transform(lambda x: np.arange(0,x.shape[0])/x.shape[0]) \
                                        .rename('proxyid2_gii_tx_ratio')
        
        # dist1 group avg
        dist1_avg_proxyid2 = grouped_by_id['dist1'] \
                                        .transform('mean').fillna(-9999) \
                                        .rename('dist1_avg_proxyid2')
        
        # dist2 group avg
        dist2_avg_proxyid2 = grouped_by_id['dist2'] \
                                        .transform('mean').fillna(-9999) \
                                        .rename('dist2_avg_proxyid2')

        
        # group size
        proxyid2_group_size = grouped_by_id['TransactionID'].transform('count') \
                                        .rename('proxyid2_group_size')
        
        # new features
        dftemp = pd.concat([proxyid2_gii_tx_ratio,
                           dist1_avg_proxyid2,
                           dist2_avg_proxyid2,
                           proxyid2_group_size], axis=1)
        
        
        dfout = pd.concat([dfout, dftemp], axis=1)
        
        
        dfout_dict[dataset] = dfout
        
    
    return dfout_dict['train'], dfout_dict['test']


def get_time_based_features(dftrain, dftest):
    """
    Create time based features for feature set FS1 
    """
    dfin_dict = {'train': dftrain, 'test': dftest}
    dfout_dict = {}
    
    for dataset in dfin_dict:
        
        dfin = dfin_dict[dataset]
    
        dfout = pd.DataFrame(index=dfin.index)

        dfout['DayNumber'] = dfin['TransactionDT']//86400
        
        dfout['DatasetDayNumber'] = dfout['DayNumber']  - dfout['DayNumber'].min()

        dfout['TimeOfDay'] = ((dfin['TransactionDT']/86400)%1)*24

        dfout['HourOfDay'] = ((dfin['TransactionDT']/86400)%1)*24//1
        
        dfout_dict[dataset] = dfout      
    
    return dfout_dict['train'], dfout_dict['test']


def get_na_based_features(dftrain, dftest):
    """
    Create NA based features for various feature sets.
    """
    dfin_dict = {'train': dftrain, 'test': dftest}
    dfout_dict = {}
    
    feature_sets = {'Vesta_subset': dftrain.columns[221:333].tolist(),
                    'id_subset': ['id_01', 'id_02', 'id_05', 'id_06', 'id_11', 'id_12', 'id_13', 'id_15',
                                   'id_17', 'id_19', 'id_20', 'id_28', 'id_29', 'id_31', 'id_35', 'id_36',
                                   'id_37', 'id_38'],
                    'M_subset': ['M1', 'M2', 'M3', 'M7', 'M8', 'M9']}
    
    
    for dataset in dfin_dict:
        
        dfin = dfin_dict[dataset]
    
        dfout = pd.DataFrame(index=dfin.index)
        
        # iterate through feature sets
        for fset in feature_sets:
            features = feature_sets[fset]
            n_features = len(features)
            dfout['percent_na_' + fset] = dfin.loc[:,features].isnull().sum(axis=1)/n_features
        
        dfout_dict[dataset] = dfout
        
    
    return dfout_dict['train'], dfout_dict['test']



def get_vesta_features(dftrain, dftest):
    """
    Create new features based on vesta features
    """
    dfin_dict = {'train': dftrain, 'test': dftest}
    dfout_dict = {}
    qt_dict = {}
    
    # define vesta_features
    vesta_features = dftrain.columns[dftrain.columns.str.contains(r"^V\d")]
    
    # PCA pipeline
    n_comp = 100
    pipe_pca = Pipeline([('impute_mean', SimpleImputer(strategy='mean')),
                         ('zscaler', StandardScaler()),
                         ('pca', IncrementalPCA(n_components=n_comp, batch_size=10000))])
    
    for dataset in dfin_dict:
        
        dfin = dfin_dict[dataset]
    
        dfout = pd.DataFrame(index=dfin.index)

        Xv = dfin.loc[:,vesta_features]
        
        if (dataset == 'train'):
            Xv_pca = pd.DataFrame(pipe_pca.fit_transform(Xv), index=Xv.index,
                                  columns=[ 'V_pca' + str(comp) for comp in range(n_comp)])
        else:
            Xv_pca = pd.DataFrame(pipe_pca.transform(Xv), index=Xv.index,
                                  columns=[ 'V_pca' + str(comp) for comp in range(n_comp)])
     
        # For columns with significantly different feature medians in Fraud vs Not Fraud
        # quantile-transform the columns
        v_sub_cols = ['V13','V36','V40', 'V50', 'V51', 'V52', 'V53', 'V54', 'V76',
                      'V140','V167','V168','V171','V172','V202','V203','V204', 'V274','V275',
                      'V307','V204', 'V203', 'V218', 'V102']                        
        for col in v_sub_cols:
            if (dataset == 'train'):
                qt = QuantileTransformer()
                dfout['qt_' + col] = qt.fit_transform(dfin[col].values.reshape(-1,1)).reshape(-1)
                dfout['qt_' + col].fillna(-999, inplace=True)
                qt_dict[col] = qt
            else:
                qt = qt_dict[col]
                dfout['qt_' + col] = qt.transform(dfin[col].values.reshape(-1,1)).reshape(-1)
                dfout['qt_' + col].fillna(-999, inplace=True)
                                  
        dfout = pd.concat([dfout, Xv_pca], axis=1)
        dfout_dict[dataset] = dfout

    return dfout_dict['train'], dfout_dict['test']
    
    

def get_preprocessed_features(dftrain, dftest, dftrain_fs1, dftest_fs1):
    """
    Preprocess rest of the input features for a Tree based model.
    """
    # Create copies
    train = dftrain.copy()
    test = dftest.copy()
    train_fs1 = dftrain_fs1.copy()
    test_fs1 = dftest_fs1.copy()
    
    # separate target variable 
    target = train['isFraud']
    train.drop(columns=['isFraud'], inplace = True)    
        
    # drop vesta_features
    vesta_features = train.columns[train.columns.str.contains(r"^V\d")]    
    train = train.drop(columns=vesta_features)
    test  = test.drop(columns=vesta_features)  
    
    # drop single valued and highly null columns
    train, test =  ut.drop_single_valued_and_highly_null_cols(train, test, single_value_thresh=1.0)

    # fillna with -9999 etc. for numeric and "NA" for categorical
    train, test =  ut.fill_na_general(train, test)

    # Label encode categorical.  
    train, test = ut.labelencode_categorical(train, test)
    train_fs1, test_fs1 =  ut.labelencode_categorical(train_fs1, test_fs1) 
    
    train = pd.concat([train, train_fs1], axis=1)
    test = pd.concat([test, test_fs1], axis=1)
         
    return train, test, target



def get_counts_features(dftrain, dftest):
    """
    
    """
    dfin_dict = {'train': dftrain, 'test': dftest}
    dfout_dict = {}
    
    for dataset in dfin_dict:
        
        dfin = dfin_dict[dataset]
    
        dfout = pd.DataFrame(index=dfin.index)

        
            
        
        dfout_dict[dataset] = dfout
        
    # Frequency encode columns
    col_list = [ ]
    dfout_dict['train'], dfout_dict['test'] =  ut.frequency_encode(dfout_dict['train'], 
                                                                   dfout_dict['test'], 
                                                                   col_list)
    
    return dfout_dict['train'], dfout_dict['test']


def get_match_features(dftrain, dftest):
    """
    
    """
    dfin_dict = {'train': dftrain, 'test': dftest}
    dfout_dict = {}
     
    card_cols = ['card3', 'card4']
    match_features = ['M1', 'M2', 'M3', 'M4' ,'M5', 'M6', 'M7', 'M8', 'M9']
    subcols =  card_cols + match_features
    
    for dataset in dfin_dict:
        
        dfin = dfin_dict[dataset]
    
        dfout = dfin.loc[:,card_cols + match_features]
        
        dfout.fillna('NA', inplace=True)
        dfout['M1-3'] = ut.create_category_interactions(dfout, ['M1', 'M2', 'M3'])
        dfout['M5-6'] = ut.create_category_interactions(dfout, ['M5', 'M6'])
        dfout['M7-9'] = ut.create_category_interactions(dfout, ['M7', 'M8', 'M9'])
        dfout['card34_M4-6'] = ut.create_category_interactions(dfout, ['card3', 'card4'] + ['M4','M5','M6'])

        dfout.drop(columns= card_cols + match_features, inplace=True)
        dfout_dict[dataset] = dfout
    
    # Frequency encode columns
    col_list = ['M1-3','M5-6','M7-9', 'card34_M4-6']
    dfout_dict['train'], dfout_dict['test'] =  ut.frequency_encode(dfout_dict['train'], 
                                                                   dfout_dict['test'], 
                                                                   col_list)
    
    return dfout_dict['train'], dfout_dict['test']


def get_counts_features(dftrain, dftest):
    """
    Get counts based features.
    """
    dfin_dict = {'train': dftrain, 'test': dftest}
    dfout_dict = {}
     
        
    card_cols = dftrain.columns[dftrain.columns.str.contains('card')].tolist()
    counts_cols = dftrain.columns[dftrain.columns.str.contains(r'C\d+$')].tolist()
    
    for dataset in dfin_dict:
        
        dfin = dfin_dict[dataset]
    
        dfout = dfin.loc[:, card_cols + counts_cols]
        
        na_filler = {'card3':-999, 'card4':'NA', 'card6':'NA'}
        groupobj = dfout.fillna(na_filler).groupby(['card3','card4','card6'])
        
        for col in counts_cols:
            dfout[col+'_card346median'] = groupobj[col].transform(np.median).fillna(-99)

            dfout[col+'_sub_card346median'] = (dfout[col] - groupobj[col].transform(np.median)).fillna(-99)

            dfout[col+'_card346_5quantile'] = groupobj[col].transform(np.quantile, q=0.05).fillna(-99)

            dfout[col+'_card346_95quantile'] = groupobj[col].transform(np.quantile, q=0.95).fillna(-99)
            
        dfout['card346groupsize'] = groupobj['C1'].transform('count')/dfout.shape[0]

        dfout.drop(columns= card_cols + counts_cols, inplace=True)
        dfout_dict[dataset] = dfout
    
    return dfout_dict['train'], dfout_dict['test']
    

 
    

    