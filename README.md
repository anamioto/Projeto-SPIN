# Projeto-SPIN: Sistema de Previsão de Inadimplência

## 📋 Visão Geral

O Projeto SPIN (Sistema de Predição de Inadimplência) é uma solução completa de Machine Learning operacionalizada em ambiente de produção, voltada à predição da probabilidade de inadimplência de um serviço de assinaturas.
O projeto foi desenvolvido utilizando Python, MLflow, Azure Machine Learning e GitHub Actions, seguindo as diretrizes de MLOps, observabilidade e boas práticas de engenharia de software para ciência de dados. 

> Esta é uma **PROPOSTA DE DOCUMENTAÇÃO**, como parte do curso da Alura de **MLOps: implantação de modelos com automação e monitoramento**.
>> Fique a vontade para clonar, fazer as modificações necessárias e aplicar as **boas práticas em seus projetos**.
>>
>> **Observação:** Durante o curso foram realizadas diversas simplificações para facilitar as explicações, como unir os códigos em uma única pasta. Deixo aqui um incentivo para que teste utilizar da forma mais adequada, lembrando de ajustar os caminhos!

## 🎯 Objetivos

- **Objetivo Principal**: Desenvolver um modelo preditivo para identificar clientes propensos a se tornarem inadimplentes.
- **Objetivos Específicos**:
  - Analisar padrões comportamentais dos clientes.
  - Implementar e comparar diferentes algoritmos de ML, utilizando experimentos.
  - Fornecer insights acionáveis para retenção de clientes e melhoria em seus pagamentos.
  - Criar visualizações interpretáveis dos resultados.
  - Escalar a solução para toda organização.
  - Aplicar as práticas de MLOPs no fluxo do projeto.

## 🏗️ Arquitetura do Projeto
> Neste repositório temos uma pasta extra dentro de data/: dados-desafio
>> **dados-desafio** contêm os dados a serem utilizados no desafio/atividade prática durante o curso. Será informado na plataforma o momento de utiliza-lo.
>>


*- Apresentar toda a arquitetura e design do projeto.*  
```
projeto-CHURN/
|   ├── data/                                    # Datasets e arquivos de dados
│   │   ├── dados-desafio/                       # Dados da atividade prática
│   │   ├── base_cliente_inadimplencia.csv       # Dados utilizados para treinar e registrar os modelos
│   │   └── base_cliente_inadimplencia_2.csv     # Dados simulando em produção
│   ├── jobs/                                    # Arquivos do tipo yaml com as configurações dos jobs
│   │   ├── agendamento-scoring-pipe.yaml        # Agendamento para rodar o pipeline
│   │   ├── pipeline.yml                         # Job automatizado para rodar o pipeline (estrutura)
│   │   └── scoring_job.yaml                     # Job para rodar as predições do modelo produtivo (estrutura)
│   ├── src/                                     # Código fonte
│   │   ├── model_registry.py                    # Script de treinamento/teste dos Modelos de ML em experimentos 
│   │   ├── pre_processamento.py                 # Script de processamento de dados
│   │   └── scoring_model_final.py               # Script de aplicação do modelo campeão em produção  
│   ├── tests/                                   # Testes unitários
│   │   ├── test_model.py                        # Teste e validação das funções de aplicação do modelo
|   |   └── test_pre_processamento.py            # Teste e validação das funções de pré-processamento  
│   ├── requirements.txt                         # Dependências
│   └── README.md                                # Documentação principal
```

## 🔧 Tecnologias Utilizadas

### Linguagens e Frameworks
- **Python 3.8+**: Linguagem principal
- **Pandas**: Manipulação de dados
- **NumPy**: Computação numérica
- **Scikit-learn e XGBoost**: Algoritmos de Machine Learning
- ...

### Ferramentas de Desenvolvimento
- **Jupyter Notebook**: Desenvolvimento interativo
- **Git**: Controle de versão
- **Pip**: Gerenciamento de dependências
- ...

## 📊 Metodologia

### 1. Análise Exploratória
- Estatísticas descritivas dos dados
- Identificação de padrões e outliers
- Análise de correlações entre variáveis
- ...

### 2. Pré-processamento
- Tratamento de valores ausentes
- Codificação de variáveis categóricas
- Normalização/padronização de features numéricas
- Divisão dos dados (train/validation/test)
- ...

### 3. Feature Engineering
- Criação de novas variáveis derivadas
- Seleção de features relevantes
- ...

### 4. Modelagem
Algoritmos implementados:
- **XGBClassifier**: Modelo baseline (modelo campeão)
- **RandomForestClassifier**: Modelo utilizado para comparação nos experimentos
- **Modelo 3**: Modelo de x
- ...

### 5. Avaliação
Métricas utilizadas:
- **Accuracy**: Precisão geral
- **Precision**: Precisão por classe
- **Recall**: Sensibilidade
- **F1-Score**: Média harmônica
- **Confusion Matrix**: Matriz de confusão
- ...

## 🚀 Como Executar

### Pré-requisitos
```bash
# Python 3.8 ou superior
python --version

# Azure ML configurado de acordo com o *Preparando ambiente*

# Git para clonar o repositório
git --version
```

### Instalação
*- Breve explicação de como instalar/rodar o seu projeto.*
  
```bash
# 1. Clone o repositório
git clone https://github.com/anamioto/projeto-SPIN.git
cd projeto-SPIN/

# 2. Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# 3. Instale as dependências
pip install -r requirements.txt
```

### Execução
*- Explicação da ordem do seu pipeline e como rodar de forma individual cada etapa.*

```bash
# 1. Execute os notebooks na ordem (exemplo no caso de notebooks)
jupyter notebook notebooks/01_exploratory_analysis.ipynb
jupyter notebook notebooks/02_data_preprocessing.ipynb
jupyter notebook notebooks/03_feature_engineering.ipynb
jupyter notebook notebooks/04_model_training.ipynb

# 2. Ou execute scripts individuais
python src/model_registry.py
python src/pre_processamento.py
python src/scoring_model_final.py

```

## 📈 Resultados Principais

### Performance dos Modelos
*- Apresentar um comparativo dos resultados dos modelos.*

| Modelo | Accuracy | Precision | Recall | F1-Score | 
|--------|----------|-----------|--------|----------|
| XGBClassifier | 0.89 | 0.75 | 0.50 | 0.81 | 
| Modelo 2 | 0.00 | 0.00 | 0.00 | 0.00 | 
| Modelo 3 | 0.00 | 0.00 | 0.00 | 0.00 | 
...

### Features Mais Importantes
*- Destacar as features mais importantes, exemplo:*
  
1. **Plano_Contratado** 
2. **Serviço_Adicional** 
3. **Data_Vencimento_Fatura** 
4. **Valor_em_Aberto** 
5. **Status_Pagamento**
6. ...

### Insights de Negócio
*- Adicionar achados e tomadas de decisões feitas.*

## 📁 Estrutura dos Dados

### Dataset Principal
*- Apresentar uma breve descrição do dataset utilizado.*
- **Registros**: 10,000 clientes
- **Features**: 16 variáveis
- **Target**: Status_Pagemnto (0: Inadimplente, 1: Em dia)
- **Taxa de Inadimplencia**: 27.96%

### Principais Variáveis
*- Apresentar quais são as variáveis utilizadas.*
- **SocioDemográficas**: Cidade, Estado, Data_Nascimento, Telefone
- **Serviços**: Servico_Adicional
- **Contratuais**: Plano_Contratado, Data_Vencimento_Fatura, Data_Contratação
- **Financeiras**: Valor_Fatura_Mensal, Valor_em_Aberto, Status_Pagamento
- ...

## 🔄 Pipeline de ML
*- Descrever como rodar o pipeline criado.*

```python
# Exemplo simplificado do pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

## 📚 Dependências
*- Destacar as dependências de bibliotecas e suas versões minimas necessarias para rodar o projeto.*

```txt
pandas>=1.3
numpy>=1.21.6
scikit-learn>=1.0
xgboost>=1.6
mlflow>=2.3
azure-ai-ml>=1.11
azure-identity>=1.14
pytest>=7.0
```

## 🤝 Contribuindo
*- Um passo-a-passo para incentivar que novas ideias ou melhorias possam ser feitas no seu projeto.*

1. **Fork** o projeto
2. Crie uma **branch** para sua feature (`git checkout -b feature/nova-feature`)
3. **Commit** suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. **Push** para a branch (`git push origin feature/nova-feature`)
5. Abra um **Pull Request**

## 📝 Próximos Passos
*- Apresentar quais são as ideias de melhoria e próximos passos a serem desenvolvidos.*

### Melhorias Técnicas
- [ ] Implementar validação cruzada estratificada
- [ ] Otimização de hiperparâmetros
- [ ] Implementar SHAP para interpretabilidade

### Análises Adicionais
- [ ] Análise de cohort dos clientes
- [ ] Segmentação de clientes (clustering)
- [ ] A/B testing para estratégias de pagamento

## 👥 Autor
*- Descreva brevemente quem é você e sua formação.*

**Ana Clara Mioto**
- GitHub: [@anamioto](https://github.com/anamioto)
- Formação: Bacharel em Informática Biomédica, Mestre em Bioengenharia
- Especialização: Data Science e Machine Learning

## 📄 Licença

*Adicionar licença ao projeto caso haja.*

## 📞 Contato

Para dúvidas, sugestões ou colaborações:
- **Issues**: Abra uma issue no GitHub
- **Email**: [incluir email]
- **LinkedIn**: [incluir perfil]
- **Instagram**: @ana_mioto

---

## 🔍 Glossário
*- Explicação/significado de termos de négocio e técnicos para entendimendo do seu projeto.*

- **Inadimplencia**: Taxa de não pagamento ou atraso no saldo devedor dos clientes
- **Feature Engineering**: Processo de criação e seleção de variáveis
- **Pipeline**: Sequência automatizada de processamento
- **Cross-validation**: Técnica de validação de modelos
- **Ensemble**: Combinação de múltiplos modelos
- **Cluster de Computação**: Conecta dois ou mais computadores em uma rede para que trabalhem de forma conjunta.
- ...

---

*Documentação atualizada em: Agosto 2025*

