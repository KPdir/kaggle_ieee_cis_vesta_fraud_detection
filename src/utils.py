import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

def create_category_interactions(df, feature_list):
    """
    Create interaction terms between categorical features.
    """
    n_features = len(feature_list)
    for idx in range(n_features):
        if (idx == 0):
            dummy = df[feature_list[idx]].fillna('NA').astype(str)
        else:
            dummy = dummy + '-'  + df[feature_list[idx]].fillna('NA').astype(str)
            
    return dummy


def pandas_group_quantile_transform(x):
    "Used inside the transform function after a pandas groupby operation"
    qt = QuantileTransformer()
    return qt.fit_transform(x.values.reshape(-1,1)).reshape(-1)


def drop_single_valued_and_highly_null_cols(dftrain, dftest,
                                            null_thresh = 0.95,
                                            single_value_thresh = 0.95):
    
    # single valued cols
    single_value_cols_train = [col for col in dftrain.columns if dftrain[col].nunique() <= 1]
    single_value_cols_test = [col for col in dftest.columns if dftest[col].nunique() <= 1]

    
    # highly null cols
    highly_null_cols_train = [col for col in dftrain.columns if dftrain[col].isnull().sum() \
                                                                  / dftrain.shape[0] >= null_thresh]
    highly_null_cols_test = [col for col in dftest.columns if dftest[col].isnull().sum() \
                                                                 / dftest.shape[0] >= null_thresh]

    highly_single_valued_cols_train = [col for col in dftrain.columns \
                                       if dftrain[col].value_counts(dropna=False,
                                                                  normalize=True).values[0] >= single_value_thresh]
    highly_single_valued_cols_test = [col for col in dftest.columns \
                                      if dftest[col].value_counts(dropna=False,
                                                                normalize=True).values[0] >= single_value_thresh]
    
    cols_to_drop = list(set(highly_null_cols_train +
                        highly_null_cols_test + 
                        highly_single_valued_cols_train + 
                        highly_single_valued_cols_test + 
                        single_value_cols_train + 
                        single_value_cols_test))
    
    print("Dropping %d highly null or single valued columns ..." %len(cols_to_drop) )
    
    dftrain.drop(columns=cols_to_drop, axis=1, inplace=True)
    dftest.drop(columns=cols_to_drop, axis=1, inplace=True)
    
    return dftrain, dftest




def fill_na_general(dftrain, dftest):
    """
    fills columns in `dftrain` and `dftest` with
    "NA" for categorical (dtype=='O') and 
    -9999 or  (1 - column minimum rounded to next order of magnitude)
    """
    for col in dftrain.columns:
        
        col_dtype = dftrain[col].dtype
        
        if (col_dtype == 'O'):
            na_value = "NA"
        elif (col_dtype == 'float') or (col_dtype == 'int'):
            colmin = min( dftrain[col].min(), dftest[col].min())
            if (colmin < 0):
                na_value = 1 - 10**round(np.log10(abs(colmin)) + 1)
            else:
                na_value = -9999
            
            # change dtype with float or int appropriately
            if (col_dtype == 'float'):
                na_value = float(na_value) 
            elif(col_dtype == 'int'):
                na_value = int(na_value) 

        dftrain[col] = dftrain[col].fillna(na_value)
        dftest[col] = dftest[col].fillna(na_value)
        
    return dftrain, dftest



def labelencode_categorical(dftrain, dftest):
    "labelencode categorical features in `dftrain` and `dftest`"
    labenc_classes_dict = {}
    for col in dftrain.columns:
        if (dftrain[col].dtype == 'O'):
            labenc = LabelEncoder()
            labenc.fit( pd.concat([dftrain[col], dftest[col] ], axis=0) )
            labenc_classes_dict[col] = labenc.classes_
            dftrain[col] = labenc.transform(dftrain[col])
            dftest[col] = labenc.transform(dftest[col])  
    
    return dftrain, dftest
