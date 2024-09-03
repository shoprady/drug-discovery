import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from utils import gen_smiles2graph


#===============================================================================
# graph execution error 및 중복, inf 제거
#===============================================================================

binding = pd.read_csv('./open/BindingDB_IC50.csv')
competition = pd.read_csv('./open/train.csv')

print("BindingDB shape:", binding.shape)
print("Competition shape:", competition.shape)

# Graph 생성 -> 에러 뜨는 경우 제거
binding.loc[:, 'Graph'] = binding['Smiles'].apply(gen_smiles2graph)
binding = binding[binding['Graph'].apply(lambda x: x[0] is not None or x[1] is not None)]

# Graph 열은 csv에서 string으로 저장되므로 제거
binding.drop(columns='Graph', inplace=True)

# 중복 제거
binding.drop_duplicates(inplace=True)

# pIC50에서 inf 값 제거
binding = binding.replace([np.inf, -np.inf], np.nan)
binding = binding.dropna(subset=['pIC50'])

# reset index 후 csv로 저장
binding.reset_index(drop=True, inplace=True)
binding.to_csv('./dataset/BindingDB_IC50_dropped.csv', index=False)


#===============================================================================
# SMILES&Uniprot 중복 존재할 경우 대회 train 것만 남기기
#===============================================================================

# 대회 데이터에 Uniprot 열 추가
competition['Uniprot'] = 'Q9NWZ3'

# 특정 열만 골라내기
competition_selected = competition[['Smiles', 'Uniprot', 'pIC50', 'IC50_nM']].copy()
binding_selected = binding[['Smiles', 'Uniprot', 'pIC50', 'IC50_nM']].copy()

# 중복 체크
data = pd.concat([competition_selected, binding_selected])
data.drop_duplicates(subset=['Smiles', 'Uniprot'], inplace=True)
data.reset_index(drop=True, inplace=True)

data['IC50_nM'] = data['IC50_nM'].astype('float64')
print("Final shape:", data.shape)


#===============================================================================
# train test split 하고 저장
#===============================================================================

train, test = train_test_split(data, test_size=113, random_state=42)
train.to_csv('./dataset/new_train.csv', index=False)
test.to_csv('./dataset/new_test.csv', index=False)