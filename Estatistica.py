#Trabalho realizado por Matheus Santos, Matheus Barros e Matheus Andreossi.
import numpy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

dados = pd.read_excel('dados_biohealer.xlsx')
X = dados.iloc[:,1].values
y = dados.iloc[:,2].values

#Dados gerais

print("Ficha dos dados do tempo dos jogadores")
print("Média: ", numpy.mean(X))#Acredito que numpy pode ser uma biblioteca ou uma classe, mas enfim ela tem uma função responsável por calcular a média entre os números.
print("Variância: ", numpy.var(X))#Variancia
print("Desvio Padrão: ", numpy.std(X, ddof = 1))#Desvio Padrão
print("Mediana: ", numpy.median(X))#Mediana

print("Ficha dos dados da quantidades de inimigos recuperados")
print("Média: ", numpy.mean(y))#Acredito que numpy pode ser uma biblioteca ou uma classe, mas enfim ela tem uma função responsável por calcular a média entre os números.
print("Variância: ", numpy.var(y))#Variancia
print("Desvio Padrão: ", numpy.std(y, ddof = 1))#Desvio Padrão
print("Mediana: ", numpy.median(y))#Mediana


#Calculo do boxplot e demonstração gráfica:

plt.boxplot(X)
plt.title("Boxplot do tempo dos jogadores")#Titulo da tabela
plt.show()

plt.boxplot(y)
plt.title("Boxplot dos inimigos recuperados dos jogadores")#Titulo da tabela
plt.show()

#Calculo do histograma e demonstração gráfica:
plt.title("Histograma do tempo dos jogadores")#Titulo da tabela
plt.xlabel("Tempo")#Nomeia o eixo x dos dados
plt.ylabel("Frêquencia absoluta")#Nomeia o eixo y dos dados
plt.hist(X, 5, rwidth=0.5, edgecolor="black")#edgecolor é a propriedade que atribui borda para o gráfico.
plt.show()
plt.title("Histograma dos inimigos recuperados")#Titulo da tabela
plt.xlabel("Inimigos recuperados")#Nomeia o eixo x dos dados
plt.ylabel("Frêquencia absoluta")#Nomeia o eixo y dos dados
plt.hist(y, 5, rwidth=0.5, edgecolor="black")#edgecolor é a propriedade que atribui borda para o gráfico.
plt.show()

#Regressão linear e coeficientes
X = X.reshape(-1,1)
regressor = LinearRegression()
regressor.fit(X,y)
print("O termo A0 dessa equação é ", regressor.coef_)
print("O termo A1 dessa equação é ", regressor.intercept_)
plt.scatter(X,y)
plt.plot(X, regressor.predict(X), color = "red")
plt.xlabel("Tempo")
plt.ylabel("Inimigos recuperados")
plt.title("Regressão linear entre o tempo do jogador com inimigos recuperados")
plt.show()

