# Galaxy Zoo — Comparação de Arquiteturas CNN

Projeto da disciplina de **Inteligência Artificial II** para treinamento, avaliação e comparação de arquiteturas de Redes Neurais Convolucionais aplicadas à classificação de galáxias.

O projeto utiliza o dataset **`mrJordi0/galaxy-zoo-dataset`**, disponibilizado pelo Hugging Face, e compara três arquiteturas:

* **ResNet18**
* **GoogLeNet**
* **MobileNetV3 Small**

Os três modelos são **treinados do zero**, utilizando inicialização aleatória dos pesos.

O objetivo é realizar os experimentos sob condições controladas e coletar métricas que possam ser utilizadas na elaboração do artigo científico.

---

# Tecnologias

* Python
* PyTorch
* Torchvision
* Hugging Face Datasets
* NumPy
* Pillow
* Scikit-learn
* Matplotlib

---

# Estrutura do projeto

```text
IA2/
│
├── models/
│   ├── resnet.py
│   ├── googlenet.py
│   └── mobilenet.py
│
├── src/
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   ├── inspect_dataset.py
│   ├── visualize_dataset.py
│   └── compare_results.py
│
├── results/
│   ├── resnet/
│   ├── googlenet/
│   └── mobilenet/
│
├── config.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

A pasta `results/` **não precisa ser criada manualmente**. O código cria os diretórios necessários automaticamente durante a execução.

---

# Instalação

## 1. Clonar o repositório

```bash
git clone <URL_DO_REPOSITORIO>

cd IA2
```

## 2. Criar o ambiente virtual

```bash
python3 -m venv .venv
```

## 3. Ativar o ambiente virtual

### Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

## 4. Instalar as dependências

```bash
pip install -r requirements.txt
```

---

# GPU NVIDIA

Antes de iniciar os experimentos, é recomendado verificar se o PyTorch está reconhecendo a GPU:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA disponível:', torch.cuda.is_available()); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Em uma máquina com CUDA configurada corretamente, o resultado deverá indicar:

```text
CUDA disponível: True
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
```

Caso `torch.cuda.is_available()` retorne `False`, o treinamento será executado na CPU.

---

# Dataset

O projeto utiliza:

```text
mrJordi0/galaxy-zoo-dataset
```

O dataset é carregado automaticamente através da biblioteca Hugging Face Datasets:

```python
load_dataset(DATASET_NAME)
```

Não é necessário baixar manualmente os arquivos nem adicioná-los ao repositório.

Na primeira execução, os arquivos do dataset serão baixados para o cache local do Hugging Face. Execuções posteriores utilizarão os arquivos armazenados em cache.

## Divisão dos dados

O dataset utilizado já possui três divisões:

```text
Train:      99.808 imagens
Validation: 24.952 imagens
Test:       31.191 imagens
```

O código utiliza:

* `train` para treinamento;
* `validation` para acompanhamento durante o treinamento e early stopping;
* `test` exclusivamente para avaliação final.

---

# Classes

O dataset possui oito classes:

| ID | Classe                  |
| -: | ----------------------- |
|  0 | Round Elliptical        |
|  1 | In-between Elliptical   |
|  2 | Cigar-shaped Elliptical |
|  3 | Edge-on Spiral          |
|  4 | Barred Spiral           |
|  5 | Unbarred Spiral         |
|  6 | Irregular               |
|  7 | Merger                  |

Existe um desbalanceamento entre as classes, especialmente nas classes `Irregular` e `Merger`.

Por isso, o treinamento utiliza **pesos de classe** na função de perda para reduzir o impacto do desbalanceamento.

---

# Inspecionando o dataset

Antes de iniciar os experimentos, é recomendado verificar a estrutura e a distribuição do dataset:

```bash
python -m src.inspect_dataset
```

O script `inspect_dataset.py` realiza uma inspeção básica do dataset e apresenta informações como:

* quantidade de exemplos em cada split;
* features disponíveis;
* tamanho e formato das imagens;
* distribuição das classes;
* percentual de cada classe.

Esse script serve principalmente para **validar o dataset antes do treinamento**, evitando iniciar experimentos longos com dados carregados ou estruturados incorretamente.

---

# Visualizando o dataset

Também é possível gerar algumas visualizações das imagens antes do treinamento:

```bash
python -m src.visualize_dataset
```

O script `visualize_dataset.py` gera visualizações para facilitar a análise inicial dos dados.

São produzidos:

* exemplos de imagens de cada uma das oito classes;
* gráfico com a distribuição das classes no conjunto de treinamento.

Os arquivos são salvos em:

```text
results/
├── dataset_samples.png
└── class_distribution.png
```

Essas visualizações permitem verificar visualmente as classes e identificar o desbalanceamento do dataset antes da execução dos modelos.

---

# Configuração

Os principais parâmetros do experimento estão centralizados em:

```text
config.py
```

A configuração utilizada nos experimentos é baseada em:

```python
IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4

PRETRAINED = False

FREEZE_BACKBONE = False

USE_CLASS_WEIGHTS = True

EARLY_STOPPING_PATIENCE = 3

SEED = 42
```

Os três modelos utilizam a mesma configuração experimental, permitindo uma comparação mais justa entre as arquiteturas.

---

# Estratégia de treinamento

Os três modelos são **treinados do zero**.

Isso significa que não são utilizados pesos previamente aprendidos em datasets como ImageNet.

A configuração utilizada é:

```python
PRETRAINED = False
```

Consequentemente, cada arquitetura começa o treinamento com seus pesos inicializados aleatoriamente e aprende exclusivamente a partir do dataset Galaxy Zoo.

A escolha por treinamento do zero mantém o experimento simples e permite comparar diretamente o comportamento das três arquiteturas sob as mesmas condições.

---

# Modelos

## ResNet18

Implementada utilizando a arquitetura disponibilizada pelo Torchvision:

```text
ResNet18
```

A camada final é substituída para produzir oito classes.

O modelo é inicializado sem pesos pré-treinados.

---

## GoogLeNet

Implementada utilizando:

```text
GoogLeNet
```

O modelo possui dois classificadores auxiliares (`aux1` e `aux2`), além do classificador principal.

Todos os classificadores são adaptados para produzir oito classes.

Durante o treinamento, a função de perda considera:

```text
Loss =
    Loss principal
    + 0.3 × Loss auxiliar 1
    + 0.3 × Loss auxiliar 2
```

Durante a validação e a avaliação final, somente a saída principal é utilizada.

O modelo também é treinado do zero, sem pesos pré-treinados.

---

## MobileNetV3 Small

Implementada utilizando:

```text
MobileNetV3 Small
```

A camada classificadora final é substituída para produzir oito classes.

A MobileNetV3 Small foi escolhida por possuir uma arquitetura consideravelmente mais leve, permitindo comparar não apenas o desempenho de classificação, mas também o custo computacional e a quantidade de parâmetros.

O modelo é treinado do zero.

---

# Treinamento

## ResNet18

```bash
python main.py --model resnet
```

## GoogLeNet

```bash
python main.py --model googlenet
```

## MobileNetV3 Small

```bash
python main.py --model mobilenet
```

## Treinar os três modelos

Para executar o experimento completo:

```bash
python main.py --model all
```

Os modelos serão treinados sequencialmente:

```text
ResNet18
    ↓
GoogLeNet
    ↓
MobileNetV3 Small
```

Cada modelo possui seu próprio diretório de resultados.

---

# O que acontece durante a execução?

Ao executar:

```bash
python main.py --model all
```

o programa:

1. carrega o dataset;
2. prepara os DataLoaders;
3. cria cada arquitetura;
4. inicializa os modelos sem pesos pré-treinados;
5. configura a função de perda com pesos de classe;
6. treina o modelo;
7. calcula as métricas de treinamento;
8. avalia no conjunto de validação;
9. aplica early stopping quando necessário;
10. salva o melhor checkpoint;
11. avalia o melhor modelo no conjunto de teste;
12. calcula as métricas finais;
13. gera gráficos;
14. salva os resultados em JSON;
15. ao final, gera uma comparação entre os três modelos.

---

# Métricas coletadas

O projeto coleta métricas durante o treinamento e durante a avaliação final.

## Durante o treinamento

São registradas:

* Training Loss;
* Validation Loss;
* Training Accuracy;
* Validation Accuracy;
* tempo por época;
* tempo total de treinamento;
* melhor Validation Accuracy;
* número de épocas executadas.

## Avaliação final

São calculadas:

* Accuracy;
* Precision Weighted;
* Recall Weighted;
* F1-score Weighted;
* Precision Macro;
* Recall Macro;
* F1-score Macro;
* Classification Report;
* Matriz de confusão.

O **F1 Macro** é especialmente importante neste projeto devido ao desbalanceamento das classes, pois atribui o mesmo peso a cada classe.

---

# Métricas de custo computacional

Também são registrados:

* número total de parâmetros;
* número de parâmetros treináveis;
* tempo total de treinamento;
* tempo de avaliação;
* dispositivo utilizado;
* GPU utilizada;
* memória máxima utilizada pela GPU.

Essas informações permitem comparar não apenas qual modelo possui melhor desempenho, mas também qual apresenta melhor relação entre **desempenho e custo computacional**.

---

# Resultados

Os resultados são automaticamente armazenados em:

```text
results/
```

Para cada arquitetura:

```text
results/
├── resnet/
│   ├── best_model.pth
│   ├── metrics.json
│   ├── training_history.json
│   ├── confusion_matrix.png
│   ├── loss_curve.png
│   └── accuracy_curve.png
│
├── googlenet/
│   ├── best_model.pth
│   ├── metrics.json
│   ├── training_history.json
│   ├── confusion_matrix.png
│   ├── loss_curve.png
│   └── accuracy_curve.png
│
└── mobilenet/
    ├── best_model.pth
    ├── metrics.json
    ├── training_history.json
    ├── confusion_matrix.png
    ├── loss_curve.png
    └── accuracy_curve.png
```

Quando todos os modelos forem executados, também será criado:

```text
results/comparison.json
```

Esse arquivo contém os resultados consolidados dos três modelos.

---

# Comparação dos resultados

Depois que os três modelos forem treinados, pode-se executar:

```bash
python -m src.compare_results
```

Esse script apresenta uma comparação dos principais resultados obtidos pelos modelos.

As comparações incluem:

| Modelo      | Accuracy | F1 Macro | F1 Weighted | Parâmetros | Tempo |
| ----------- | -------: | -------: | ----------: | ---------: | ----: |
| ResNet18    |        - |        - |           - |          - |     - |
| GoogLeNet   |        - |        - |           - |          - |     - |
| MobileNetV3 |        - |        - |           - |          - |     - |

Os valores serão preenchidos após a execução dos experimentos.

---

# Arquivos importantes para o artigo

## `metrics.json`

Contém as métricas finais do modelo, informações de hardware, quantidade de parâmetros e tempos de execução.

## `training_history.json`

Contém os dados obtidos durante cada época do treinamento.

Pode ser utilizado para analisar:

* convergência;
* overfitting;
* estabilidade do treinamento;
* evolução da loss;
* evolução da acurácia.

## `confusion_matrix.png`

Permite analisar quais classes são mais confundidas pelo modelo.

## `loss_curve.png`

Mostra a evolução da loss de treinamento e validação.

## `accuracy_curve.png`

Mostra a evolução da accuracy de treinamento e validação.

## `comparison.json`

Consolida os resultados dos três modelos e facilita a criação das tabelas comparativas do artigo.

---

# Fluxo do experimento

```text
                  Galaxy Zoo Dataset
                         │
                         ▼
                  Inspeção dos dados
                         │
                         ▼
                  Visualização dos dados
                         │
                         ▼
                  Pré-processamento
                         │
                         ▼
                     DataLoader
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          ResNet18    GoogLeNet   MobileNetV3
             │           │           │
             └───────────┼───────────┘
                         ▼
              Inicialização aleatória
                         │
                         ▼
                    Treinamento
                         │
                         ▼
                  Melhor checkpoint
                         │
                         ▼
                    Teste final
                         │
                         ▼
                      Métricas
                         │
                         ▼
                Matrizes e gráficos
                         │
                         ▼
                  Comparação final
                         │
                         ▼
                  Artigo científico
```

---

# Reprodutibilidade

O projeto utiliza uma seed fixa:

```python
SEED = 42
```

A seed é aplicada às principais bibliotecas utilizadas no treinamento.

Além disso, os três modelos utilizam a mesma configuração experimental, permitindo uma comparação mais justa entre as arquiteturas.

---

# Comandos essenciais

### Verificar dataset

```bash
python -m src.inspect_dataset
```

### Visualizar dataset

```bash
python -m src.visualize_dataset
```

### Treinar ResNet18

```bash
python main.py --model resnet
```

### Treinar GoogLeNet

```bash
python main.py --model googlenet
```

### Treinar MobileNetV3 Small

```bash
python main.py --model mobilenet
```

### Treinar todos

```bash
python main.py --model all
```

### Comparar resultados

```bash
python -m src.compare_results
```

---

# Objetivo do projeto

O objetivo não é apenas identificar qual arquitetura apresenta a maior acurácia.

A análise pretende comparar as arquiteturas considerando:

* desempenho de classificação;
* F1 Macro;
* F1 Weighted;
* desempenho por classe;
* matriz de confusão;
* comportamento durante o treinamento;
* quantidade de parâmetros;
* tempo de treinamento;
* tempo de avaliação;
* utilização de memória da GPU.

Dessa forma, os resultados poderão ser utilizados para discutir os **trade-offs entre desempenho e custo computacional** das arquiteturas avaliadas no artigo científico de Inteligência Artificial II.

```

### Uma observação importante

Eu **não colocaria `FREEZE_BACKBONE = False` no README**, apesar de ele existir no `config.py`. Como vocês decidiram treinar do zero, esse parâmetro deixou de ter relevância para o experimento e pode até gerar confusão sobre transfer learning.

Também corrigi a descrição do GoogLeNet: **as saídas auxiliares são usadas no treinamento**, com peso `0.3`, enquanto validação/teste usam apenas a saída principal.

E agora o fluxo fica bem coerente: **inspecionar → visualizar → treinar → avaliar → comparar → usar os resultados no artigo**.
```
