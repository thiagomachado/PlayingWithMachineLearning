# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 17:21:07 2020

@author: Thiago Machado
"""
import pandas as pd
import math

from sklearn.neighbors     import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, LabelEncoder


class DataProcessing():
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.labelencoder = LabelEncoder()
        
    def exclude_columns(self, columns):
        self.data = self.data.drop(columns, axis=1)
        
    def change_value(self, column_name, value, new_value):
        for idx, column_value in enumerate(self.data[column_name]):
            if column_value == value :                
                self.data[column_name][idx] = new_value
                
    def change_value_nan(self, column_name, new_value):
        for idx, column_value in enumerate(self.data[column_name]):
            if math.isnan(column_value):                
                self.data[column_name][idx] = new_value
                
    def organize_categories(self, column_name, num_categories):
        self.data[column_name] = self.labelencoder.fit_transform(pd.qcut(self.data[column_name],duplicates='drop', q=num_categories, precision=0))
    
    def process_data(self):
        columns_to_drop = ['id_solicitante', 'grau_instrucao', 'meses_no_trabalho', 
            'qtde_contas_bancarias_especiais', 'possui_telefone_celular',
            'estado_onde_nasceu', 'estado_onde_nasceu', 'estado_onde_trabalha',
            'codigo_area_telefone_residencial', 'codigo_area_telefone_trabalho',
            'grau_instrucao_companheiro', 'profissao_companheiro']
        
        self.exclude_columns(columns_to_drop)
        self.change_value('sexo', ' ', 'N')        
        self.change_value_nan('ocupacao', 2.0)
        self.change_value_nan('profissao', 9.0)
        self.change_value_nan('tipo_residencia', 1.0)
        self.change_value_nan('meses_na_residencia', 1.0)
                 
        binarizer = LabelBinarizer()
        for label in ['tipo_endereco','possui_telefone_residencial', 'vinculo_formal_com_empresa', 'possui_telefone_trabalho']:
            self.data[label] = binarizer.fit_transform(self.data[label])
        self.data = pd.get_dummies(self.data,columns=['forma_envio_solicitacao', 'estado_onde_reside', 'sexo'])
        self.orgnize_categories('local_onde_reside', 40)
        self.orgnize_categories('renda_mensal_regular', 60)
        self.orgnize_categories('meses_na_residencia', 5)
        self.orgnize_categories('renda_extra', 60)
        self.orgnize_categories('idade', 5)
        self.orgnize_categories('valor_patrimonio_pessoal', 5)
        self.orgnize_categories('local_onde_trabalha', 40)
        
    
        

traning = 'datasets/training_dataset.csv'
test = 'datasets/test_dataset.csv'


dataProcessing = DataProcessing(traning)
dataProcessing.process_data()

testDataProcessing = DataProcessing(test)
testDataProcessing.process_data()
#print(dataProcessing.data.T)

#print(dataProcessing.data.dtypes)


variaveis_categoricas = [
    x for x in dataProcessing.data.columns
    ]

print(variaveis_categoricas)

print ( '\nVerificar a cardinalidade de cada variável categórica:')
print ( 'obs: cardinalidade = qtde de valores distintos que a variável pode assumir\n')

for v in variaveis_categoricas:          
    print ('\n%15s:'%v , "%4d categorias" % len(dataProcessing.data[v].unique()))        
    print (dataProcessing.data[v].unique(),'\n')    

 

# print('ocupacao:')
# print(dataProcessing.data['ocupacao'].value_counts())
# print('tipo_residencia:')
# print(dataProcessing.data['tipo_residencia'].value_counts())
# print('meses_na_residencia:')
# print(dataProcessing.data['meses_na_residencia'].value_counts())

# nan_count = 0
# for sexo in dataProcessing.data['sexo']:
#     if sexo == ' ' :
#         nan_count += 1
# print(nan_count)

#print (dataProcessing.data['grau_instrucao_companheiro'])

#shuffle
dataProcessing.data = dataProcessing.data.sample(frac=1,random_state=12345)




#------------------------------------------------------------------------------
# Criar os arrays X e Y separando atributos e alvo
#------------------------------------------------------------------------------

x = dataProcessing.data.loc[:,dataProcessing.data.columns!='inadimplente'].values
y = dataProcessing.data.loc[:,dataProcessing.data.columns=='inadimplente'].values
q=20000
x_treino = x[:q,:]
y_treino = y[:q].ravel()


x_teste = testDataProcessing.data


#-------------------------------------------------------------------------------
# Ajustar a escala dos atributos nos conjuntos de treino e de teste
#-------------------------------------------------------------------------------

ajustador_de_escala = MinMaxScaler()
ajustador_de_escala.fit(x_treino)

x_treino = ajustador_de_escala.transform(x_treino)



classifier = KNeighborsClassifier(n_neighbors=5, weights='uniform', p= 1)
classifier = classifier.fit(x_treino,y_treino)
y_resposta_treino = classifier.predict(x_treino)
y_resposta_teste = classifier.predict(x_teste)


pd.DataFrame(y_resposta_teste).to_csv("file.csv")
print ("\nDESEMPENHO DENTRO DA AMOSTRA DE TREINO\n")

total   = len(y)
acertos = sum(y_resposta_treino==y_treino)
erros   = sum(y_resposta_treino!=y_treino)

print ("Total de amostras: " , total)
print ("Respostas corretas:" , acertos)
print ("Respostas erradas: " , erros)

acuracia = acertos / total

print (f"Acurácia = {100*acuracia}")


# for k in range(1,50,2):

#     classificador = KNeighborsClassifier(
#         n_neighbors = k,
#         weights     = 'uniform',
#         p           = 1
#         )
#     classificador = classificador.fit(x_treino,y_treino)

#     y_resposta_treino = classificador.predict(x_treino)
#     #y_resposta_teste  = classificador.predict(x_teste)
    
#     acuracia_treino = sum(y_resposta_treino==y_treino)/len(y_treino)
#     # acuracia_teste  = sum(y_resposta_teste ==y_teste) /len(y_teste)
    
#     print(
#         "%3d"%k,
#         "%6.1f" % (100*acuracia_treino)
#         )

