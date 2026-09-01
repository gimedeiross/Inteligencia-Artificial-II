# Galaxy Zoo — Comparação de Arquiteturas

Projeto da disciplina de **Inteligência Artificial II** para treinamento e comparação de arquiteturas de Redes Neurais Convolucionais aplicadas à classificação de imagens de galáxias.

As arquiteturas avaliadas são:

* **ResNet18**
* **GoogLeNet**
* **MobileNetV3 Small**

O projeto utiliza **PyTorch**, **Torchvision** e **Hugging Face Datasets**, com transferência de aprendizado a partir de pesos pré-treinados no ImageNet.

---

## Tecnologias

* Python
* PyTorch
* Torchvision
* Hugging Face Datasets
* NumPy
* Scikit-learn
* Matplotlib
* Pillow

---

## Estrutura do projeto

```text
galaxy-zoo/
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
│   ├── inspect_dataset.py
│   └── utils.py
│
├── results/
│   ├── resnet/
│   ├── googlenet/
│   └── mobilenet/
│
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

A pasta `results/` **não precisa ser criada manualmente**. O código cria automaticamente os diretórios necessários durante a execução.

---

# Instalação

## 1. Clonar o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd galaxy-zoo
```

## 2. Criar o ambiente virtual

Linux:

```bash
python3 -m venv .venv
```

Ativar:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

## 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

---

# GPU NVIDIA

Antes de iniciar os experimentos, é recomendado verificar se o PyTorch está reconhecendo a GPU:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA disponível:', torch.cuda.is_available()); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Em uma máquina com GPU NVIDIA corretamente configurada, deve aparecer algo semelhante a:

```text
PyTorch: 2.x.x+cuXXX
CUDA disponível: True
CUDA: X.X
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
```

Se `CUDA disponível` retornar `False`, o treinamento será executado na CPU.

---

# Dataset

O projeto utiliza o dataset:

```text
mrJordi0/galaxy-zoo-dataset
```

O dataset é carregado automaticamente através da biblioteca **Hugging Face Datasets**.

Não é necessário baixar manualmente os arquivos nem adicioná-los ao repositório.

Na primeira execução, os arquivos serão baixados e armazenados no cache local do Hugging Face.

O dataset atualmente possui:

```text
Train:       99.808 imagens
Validation:  24.952 imagens
Test:        31.191 imagens
```

Total:

```text
155.951 imagens
```

As imagens possuem resolução original de aproximadamente `424 × 424` pixels e são redimensionadas para `224 × 224` durante o pré-processamento.

---

# Classes

O problema possui 8 classes:

| Código | Classe                  |
| -----: | ----------------------- |
|      0 | Round Elliptical        |
|      1 | In-between Elliptical   |
|      2 | Cigar-shaped Elliptical |
|      3 | Edge-on Spiral          |
|      4 | Barred Spiral           |
|      5 | Unbarred Spiral         |
|      6 | Irregular               |
|      7 | Merger                  |

A distribuição das classes é desbalanceada. Por isso, além da **Accuracy**, o projeto utiliza métricas como **F1 Macro**, que atribui o mesmo peso a todas as classes.

---

# Inspecionando o dataset

Para verificar a estrutura, quantidade de imagens e distribuição das classes:

```bash
python -m src.inspect_dataset
```

Esse comando apresenta:

* divisões `train`, `validation` e `test`;
* quantidade de imagens;
* features disponíveis;
* resolução e formato das imagens;
* distribuição das classes.

---

# Configuração do experimento

Os principais parâmetros estão centralizados em:

```text
config.py
```

Exemplo:

```python
NUM_CLASSES = 8

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4

PRETRAINED = True

EARLY_STOPPING_PATIENCE = 3

SEED = 42
```

Isso permite que os três modelos sejam treinados sob as **mesmas condições experimentais**.

Não é utilizado learning rate scheduler no projeto.

---

# Modelos

## ResNet18

Implementada em:

```text
models/resnet.py
```

Utiliza pesos pré-treinados e substitui a camada final para realizar a classificação das 8 classes.

## GoogLeNet

Implementada em:

```text
models/googlenet.py
```

Utiliza pesos pré-treinados e possui as camadas auxiliares (`aux1` e `aux2`) configuradas para o problema de 8 classes.

## MobileNetV3 Small

Implementada em:

```text
models/mobilenet.py
```

Utiliza a variante **MobileNetV3 Small**, também com pesos pré-treinados.

A escolha permite comparar uma arquitetura mais leve e eficiente computacionalmente com arquiteturas mais tradicionais.

---

# Treinamento

## Treinar somente a ResNet18

```bash
python main.py --model resnet
```

## Treinar somente a GoogLeNet

```bash
python main.py --model googlenet
```

## Treinar somente a MobileNetV3 Small

```bash
python main.py --model mobilenet
```

## Treinar as três arquiteturas

Para executar o experimento completo:

```bash
python main.py --model all
```

O código executará:

```text
ResNet18
   ↓
GoogLeNet
   ↓
MobileNetV3 Small
```

Cada modelo será:

1. criado;
2. treinado;
3. avaliado no conjunto de teste;
4. terá suas métricas salvas;
5. terá seus gráficos gerados;
6. terá seus pesos salvos;
7. será incluído na comparação final.

---

# Resultados

A pasta `results/` é criada automaticamente.

Após executar:

```bash
python main.py --model all
```

a estrutura será semelhante a:

```text
results/
│
├── resnet/
│   ├── best_model.pth
│   ├── training_history.json
│   ├── metrics.json
│   ├── confusion_matrix.png
│   ├── loss_curve.png
│   └── accuracy_curve.png
│
├── googlenet/
│   ├── best_model.pth
│   ├── training_history.json
│   ├── metrics.json
│   ├── confusion_matrix.png
│   ├── loss_curve.png
│   └── accuracy_curve.png
│
├── mobilenet/
│   ├── best_model.pth
│   ├── training_history.json
│   ├── metrics.json
│   ├── confusion_matrix.png
│   ├── loss_curve.png
│   └── accuracy_curve.png
│
└── comparison.json
```

---

# Métricas

Durante o treinamento são registradas:

* Training Loss
* Validation Loss
* Training Accuracy
* Validation Accuracy
* Tempo de cada época
* Tempo total de treinamento
* Melhor Validation Accuracy
* Número de épocas executadas

Durante a avaliação no conjunto de teste são calculadas:

* Accuracy
* Precision Weighted
* Recall Weighted
* F1-score Weighted
* Precision Macro
* Recall Macro
* F1-score Macro
* Classification Report
* Matriz de confusão

Além disso, são registrados:

* número total de parâmetros;
* número de parâmetros treináveis;
* dispositivo utilizado;
* GPU utilizada;
* memória máxima de GPU utilizada;
* batch size;
* tamanho das imagens;
* learning rate;
* weight decay;
* número de workers;
* seed;
* utilização de pesos pré-treinados.

---

# Arquivos de resultados

## `training_history.json`

Contém o histórico do treinamento de cada época.

Pode ser utilizado para analisar a evolução de:

```text
Train Loss
Validation Loss
Train Accuracy
Validation Accuracy
```

e para estudar possíveis sinais de overfitting.

---

## `metrics.json`

Contém as métricas finais do modelo no conjunto de teste, além de informações sobre:

* parâmetros;
* tempo de treinamento;
* tempo de avaliação;
* hardware;
* configuração do experimento.

Esse é um dos principais arquivos para a elaboração das tabelas do artigo.

---

## `confusion_matrix.png`

Apresenta a matriz de confusão do modelo no conjunto de teste.

Ela permite identificar quais classes são mais confundidas entre si.

---

## `loss_curve.png`

Apresenta:

```text
Training Loss
Validation Loss
```

ao longo das épocas.

---

## `accuracy_curve.png`

Apresenta:

```text
Training Accuracy
Validation Accuracy
```

ao longo das épocas.

---

## `comparison.json`

Quando o comando:

```bash
python main.py --model all
```

é utilizado, os resultados das três arquiteturas são reunidos automaticamente em:

```text
results/comparison.json
```

Esse arquivo facilita a comparação direta entre os modelos.

---

# Comparação dos modelos

Os principais indicadores para comparação serão:

| Modelo            | Accuracy | F1 Macro | F1 Weighted | Parâmetros | Tempo |
| ----------------- | -------: | -------: | ----------: | ---------: | ----: |
| ResNet18          |        — |        — |           — |          — |     — |
| GoogLeNet         |        — |        — |           — |          — |     — |
| MobileNetV3 Small |        — |        — |           — |          — |     — |

Os valores serão preenchidos após a execução dos experimentos.

A análise também poderá considerar:

* Precision;
* Recall;
* desempenho individual por classe;
* matriz de confusão;
* evolução de Loss;
* evolução de Accuracy;
* número de parâmetros;
* tempo de treinamento;
* utilização de memória da GPU.

---

# Reprodutibilidade

O projeto utiliza uma seed fixa:

```python
SEED = 42
```

A seed é aplicada ao Python, NumPy e PyTorch.

Além disso, são configurados os parâmetros de determinismo do CUDA/cuDNN quando uma GPU está disponível.

Isso permite tornar os experimentos mais reprodutíveis.

---

# Fluxo do experimento

```text
             Galaxy Zoo Dataset
                     │
                     ▼
             Carregamento
                     │
                     ▼
            Pré-processamento
                     │
                     ▼
               DataLoaders
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       ResNet18   GoogLeNet  MobileNetV3
          │          │          │
          └──────────┼──────────┘
                     ▼
                Treinamento
                     │
                     ▼
              Melhor modelo
                     │
                     ▼
                Avaliação
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Métricas   Gráficos   Confusão
          │
          ▼
       comparison.json
          │
          ▼
     Comparação para o artigo
```

---

# Experimento para o artigo

O objetivo é manter as condições experimentais iguais para as três arquiteturas, permitindo uma comparação mais justa.

Os modelos serão treinados utilizando:

* mesmo dataset;
* mesmas divisões `train`, `validation` e `test`;
* mesmo tamanho de entrada;
* mesmo batch size;
* mesmo número máximo de épocas;
* mesmo learning rate;
* mesmo weight decay;
* mesma seed;
* pesos pré-treinados;
* mesmo critério de avaliação.

Dessa forma, as diferenças observadas nos resultados poderão ser relacionadas principalmente às características das arquiteturas avaliadas.

---

# Artigo científico

Os resultados produzidos pelo projeto serão utilizados na elaboração do artigo científico da disciplina de **Inteligência Artificial II**.

Os arquivos presentes em `results/` servirão como fonte para:

* tabelas comparativas;
* gráficos;
* análise de desempenho;
* análise por classe;
* discussão dos resultados;
* conclusões sobre as arquiteturas.