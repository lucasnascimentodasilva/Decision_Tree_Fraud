from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import matplotlib.pyplot as plt


#################### DECISION TREE ######################
base_risco_fraude = pd.read_csv(r'.\risco_fraude.csv')
#atributos previsores
x_risco_fraude = base_risco_fraude.iloc[:,0:4].values

#classe 
y_risco_fraude = base_risco_fraude.iloc[:,4].values

print(x_risco_fraude)
print(y_risco_fraude)

#transformando as strings em números
label_encoder_estabelecimento = LabelEncoder()
label_encoder_valor = LabelEncoder()
label_encoder_regiao = LabelEncoder()
label_encoder_limite = LabelEncoder()

x_risco_fraude[:,0] = label_encoder_estabelecimento.fit_transform(x_risco_fraude[:,0])
x_risco_fraude[:,1] = label_encoder_valor.fit_transform(x_risco_fraude[:,1])
x_risco_fraude[:,2] = label_encoder_regiao.fit_transform(x_risco_fraude[:,2])
x_risco_fraude[:,3] = label_encoder_limite.fit_transform(x_risco_fraude[:,3])

print(x_risco_fraude)

#salvando atributos transformados para utilizacoes futuras
with open('risco_fraude.pkl', 'wb') as f:
    pickle.dump([x_risco_fraude, y_risco_fraude], f)

#lendo atributos guardados acima
with open('risco_fraude.pkl', 'rb') as f:
    X_risco_fraude, y_risco_fraude = pickle.load(f)

# Decision Tree | Criterios e Visualização
arvore_risco_fraude = DecisionTreeClassifier(criterion='entropy')
arvore_risco_fraude.fit(X_risco_fraude, y_risco_fraude)
arvore_risco_fraude.feature_importances_
previsores = ['estabelecimento', 'valor', 'regiao', 'limite']
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
tree.plot_tree(arvore_risco_fraude, feature_names=previsores, class_names = arvore_risco_fraude.classes_, filled = True);

#Submetendo novos casos a árvore
previsoes = arvore_risco_fraude.predict([[2,0,2,2],[1,1,0,0],[0,0,1,1],[1,1,1,1]])
previsoes