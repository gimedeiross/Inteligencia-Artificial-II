# Galaxy Zoo — Comparação de Arquiteturas CNN

Projeto da disciplina de **Inteligência Artificial II** para treinamento, avaliação e comparação de arquiteturas de Redes Neurais Convolucionais aplicadas à classificação de galáxias.

O projeto utiliza o dataset **`mrJordi0/galaxy-zoo-dataset`**, disponibilizado pelo Hugging Face, e compara três arquiteturas:

* **ResNet18**
* **GoogLeNet**
* **MobileNetV3 Small**

Os três modelos utilizam **transfer learning**: partem de pesos pré-treinados na ImageNet e, por padrão, treinam apenas a cabeça de classificação (backbone congelado).

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

PRETRAINED = True

FREEZE_BACKBONE = True

USE_CLASS_WEIGHTS = True

EARLY_STOPPING_PATIENCE = 3

SEED = 42
```

Os três modelos utilizam a mesma configuração experimental, permitindo uma comparação mais justa entre as arquiteturas.

---

# Estratégia de treinamento — Transfer Learning

Os três modelos utilizam **transfer learning** a partir de pesos pré-treinados na ImageNet.

Dois parâmetros em `config.py` controlam o comportamento:

```python
PRETRAINED = True

FREEZE_BACKBONE = True
```

* **`PRETRAINED = True`** — cada arquitetura é inicializada com os pesos pré-treinados na ImageNet (`ResNet18_Weights.DEFAULT`, `GoogLeNet_Weights.DEFAULT`, `MobileNet_V3_Small_Weights.DEFAULT`), em vez de pesos aleatórios.
* **`FREEZE_BACKBONE = True`** — o extrator de características (backbone) é congelado (`requires_grad = False`), e **apenas a cabeça de classificação** — recriada para as 8 classes do Galaxy Zoo — é treinada. Essa técnica é conhecida como **feature extraction**.

Se `FREEZE_BACKBONE = False`, o backbone continua inicializado com pesos da ImageNet, mas toda a rede é treinada (**fine-tuning**), ajustando também as camadas convolucionais pré-treinadas ao domínio do Galaxy Zoo.

O otimizador (`AdamW`) recebe apenas os parâmetros com `requires_grad = True`, então, com o backbone congelado, o gradiente é calculado e atualizado exclusivamente na cabeça de classificação — reduzindo o custo computacional do treinamento.

A quantidade de parâmetros treináveis (em relação ao total) é impressa no console a cada modelo treinado, e também fica registrada em `metrics.json` (`parameters.total`), permitindo comparar o "tamanho efetivo" do treinamento entre os três modelos.

---

# Modelos

## ResNet18

Implementada utilizando a arquitetura disponibilizada pelo Torchvision:

```text
ResNet18
```

Inicializada com pesos pré-treinados na ImageNet (`ResNet18_Weights.DEFAULT`).

A camada final (`fc`) é substituída para produzir oito classes; essa nova camada é sempre treinável, independentemente de `FREEZE_BACKBONE`.

---

## GoogLeNet

Implementada utilizando:

```text
GoogLeNet
```

Inicializada com pesos pré-treinados na ImageNet (`GoogLeNet_Weights.DEFAULT`).

O modelo possui dois classificadores auxiliares (`aux1` e `aux2`), além do classificador principal.

Todos os classificadores (`fc`, `aux1.fc2`, `aux2.fc2`) são adaptados para produzir oito classes e permanecem sempre treináveis, mesmo com o restante do backbone congelado.

Durante o **treinamento**, a função de perda considera:

```text
Loss =
    Loss principal
    + 0.3 × Loss auxiliar 1
    + 0.3 × Loss auxiliar 2
```

Durante a **validação e a avaliação final**, somente a saída principal é utilizada.

---

## MobileNetV3 Small

Implementada utilizando:

```text
MobileNetV3 Small
```

Inicializada com pesos pré-treinados na ImageNet (`MobileNet_V3_Small_Weights.DEFAULT`).

Quando o backbone é congelado, apenas `model.features` (o extrator convolucional) tem os pesos congelados — todo o `model.classifier` (não só a última camada) permanece treinável, já que a cabeça do MobileNetV3 é composta por múltiplas camadas (`Linear → Hardswish → Dropout → Linear`) que costumam se beneficiar de treinar juntas.

A MobileNetV3 Small foi escolhida por possuir uma arquitetura consideravelmente mais leve, permitindo comparar não apenas o desempenho de classificação, mas também o custo computacional e a quantidade de parâmetros.

---

# Treinamento

O `main.py` aceita a flag `--model` para escolher o que treinar:

## Treinar um único modelo (útil para testar rapidamente se o pipeline está funcionando)

```bash
python main.py --model resnet
```

```bash
python main.py --model googlenet
```

```bash
python main.py --model mobilenet
```

## Treinar os três modelos sequencialmente

```bash
python main.py --model all
```

`--model all` é o valor **padrão** — rodar `python main.py` sem nenhuma flag também treina os três modelos, na ordem definida em `config.MODELS`:

```text
ResNet18
    ↓
GoogLeNet
    ↓
MobileNetV3 Small
```

Cada modelo possui seu próprio diretório de resultados (`results/<modelo>/`), independentemente de ter sido treinado sozinho ou em conjunto com os outros.

---

# O que acontece durante a execução?

Ao executar (por exemplo):

```bash
python main.py --model all
```

o programa:

1. carrega o dataset;
2. prepara os DataLoaders;
3. cria cada arquitetura selecionada, carregando pesos pré-treinados na ImageNet;
4. congela o backbone (se `FREEZE_BACKBONE = True`), mantendo treinável apenas a cabeça de classificação;
5. configura a função de perda com pesos de classe;
6. treina o modelo, calculando accuracy e F1 macro por época em treino e validação;
7. avalia no conjunto de validação a cada época e aplica early stopping quando necessário;
8. salva o checkpoint sempre que o F1 macro de validação melhora;
9. ao final do treino, recarrega o melhor checkpoint salvo;
10. avalia o melhor modelo no conjunto de teste;
11. calcula as métricas finais;
12. gera gráficos (loss, accuracy, F1 macro e matriz de confusão);
13. salva os resultados em JSON;
14. ao final de todos os modelos selecionados, salva um resumo consolidado (`training_summary.json`).

---

# Métricas coletadas

O projeto coleta métricas durante o treinamento e durante a avaliação final.

## Durante o treinamento

São registradas por época:

* Training Loss / Validation Loss;
* Training Accuracy / Validation Accuracy;
* Training F1 Macro / Validation F1 Macro;
* tempo por época;
* tempo total de treinamento;
* melhor Validation F1 Macro e a Validation Accuracy correspondente;
* número de épocas executadas;
* memória máxima utilizada pela GPU (`max_gpu_memory_gb`).

O **F1 Macro de validação** — e não a accuracy — é o critério usado para decidir qual checkpoint salvar e para o early stopping, por ser mais robusto ao desbalanceamento entre classes.

## Avaliação final

São calculadas, no conjunto de teste:

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
* número de parâmetros treináveis (relevante especialmente com `FREEZE_BACKBONE = True`, quando esse número é bem menor que o total);
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
│   ├── history.json
│   ├── confusion_matrix.png
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   └── f1_macro_curve.png
│
├── googlenet/
│   ├── best_model.pth
│   ├── metrics.json
│   ├── history.json
│   ├── confusion_matrix.png
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   └── f1_macro_curve.png
│
└── mobilenet/
    ├── best_model.pth
    ├── metrics.json
    ├── history.json
    ├── confusion_matrix.png
    ├── loss_curve.png
    ├── accuracy_curve.png
    └── f1_macro_curve.png
```

Ao final de uma execução do `main.py`, também é criado:

```text
results/training_summary.json
```

Esse arquivo consolida os resultados brutos dos modelos treinados **naquela execução** (um único modelo, se `--model resnet/googlenet/mobilenet` foi usado, ou os três, se `--model all`).

> **Atenção:** `results/comparison.json` — a tabela comparativa formatada, usada para o artigo — **não** é gerado automaticamente pelo `main.py`. Ele é produzido separadamente por `compare_results.py` (veja a seção seguinte), que também exige que os três modelos já tenham sido treinados e seus `metrics.json` estejam presentes.

---

# Comparação dos resultados

Depois que os três modelos forem treinados (`python main.py --model all`, ou os três `--model <modelo>` individualmente), execute:

```bash
python -m src.compare_results
```

Esse script lê `results/<modelo>/metrics.json` de cada arquitetura, monta a tabela comparativa e imprime no console:

| Modelo      | Accuracy | F1 Macro | F1 Weighted | Parâmetros | Tempo |
| ----------- | -------: | -------: | ----------: | ---------: | ----: |
| ResNet18    |        - |        - |           - |          - |     - |
| GoogLeNet   |        - |        - |           - |          - |     - |
| MobileNetV3 |        - |        - |           - |          - |     - |

Os valores serão preenchidos após a execução dos experimentos.

Além da tabela, o script gera e salva:

```text
results/
├── comparison.json
└── model_comparison.png
```

---

# Arquivos importantes para o artigo

## `metrics.json`

Contém as métricas finais do modelo, organizadas em três blocos:

```json
{
  "model": "resnet",
  "metrics": { "accuracy": "...", "f1_macro": "...", "..." : "..." },
  "parameters": { "total": "...", "trainable": "..." },
  "training": {
    "training_time_seconds": "...",
    "evaluation_time_seconds": "...",
    "epochs_completed": "...",
    "best_validation_accuracy": "...",
    "best_validation_f1_macro": "...",
    "max_gpu_memory_gb": "...",
    "pretrained": true,
    "freeze_backbone": true,
    "class_weights": true
  }
}
```

Esse schema é o mesmo lido por `compare_results.py` para montar a tabela comparativa.

## `history.json`

Contém os dados obtidos durante cada época do treinamento (loss, accuracy e F1 macro de treino e validação).

Pode ser utilizado para analisar:

* convergência;
* overfitting;
* estabilidade do treinamento;
* evolução da loss, da accuracy e do F1 macro.

## `confusion_matrix.png`

Permite analisar quais classes são mais confundidas pelo modelo.

## `loss_curve.png` / `accuracy_curve.png` / `f1_macro_curve.png`

Mostram a evolução, por época, da loss, da accuracy e do F1 macro de treino e validação, respectivamente.

## `training_summary.json`

Gerado pelo `main.py` ao final de cada execução. Consolida em uma lista os resultados brutos (mesmo schema de `metrics.json`) de todos os modelos treinados naquela execução específica.

## `comparison.json`

Gerado pelo `compare_results.py` (execução separada, depois de treinar os três modelos). Consolida os resultados dos três modelos em uma tabela achatada e facilita a criação das tabelas comparativas do artigo.

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
        Pesos pré-treinados (ImageNet)
                         │
                         ▼
          Freeze do backbone (opcional)
                         │
                         ▼
                    Treinamento
                         │
                         ▼
                  Melhor checkpoint
                    (F1 Macro)
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

A seed é aplicada às principais bibliotecas utilizadas no treinamento (`random`, `numpy`, `torch`, `torch.cuda`), além de `cudnn.deterministic = True` e `cudnn.benchmark = False`.

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

### Treinar apenas um modelo (teste rápido do pipeline)

```bash
python main.py --model resnet
```

```bash
python main.py --model googlenet
```

```bash
python main.py --model mobilenet
```

### Treinar todos (padrão)

```bash
python main.py
```

ou, de forma explícita:

```bash
python main.py --model all
```

### Comparar resultados (depois de treinar os três modelos)

```bash
python -m src.compare_results
```

---

# Técnicas de IA utilizadas — base teórica

Esta seção resume as técnicas empregadas no pipeline e **por que** cada uma foi escolhida neste experimento especificamente, servindo como rascunho para a seção de metodologia do artigo.

## Transfer Learning

* **Pesos pré-treinados na ImageNet** para os três backbones. Evita treinar do zero com um dataset de tamanho moderado (~156 mil imagens no total) e acelera a convergência, já que filtros de baixo/médio nível (bordas, texturas, formas) aprendidos na ImageNet são reaproveitáveis para imagens de galáxias.
* **Freeze de backbone configurável** (`FREEZE_BACKBONE`). Por padrão, congelamos o backbone e treinamos só a cabeça de classificação (*feature extraction*) — reduz custo computacional e risco de overfitting em relação a ajustar toda a rede. A opção de fine-tuning completo fica disponível para quem quiser comparar o trade-off.

## Balanceamento de classes

* **Pesos de classe na função de perda** (`nn.CrossEntropyLoss(weight=...)`), calculados a partir da frequência de cada classe no split de treino. Necessário porque classes como `Irregular` e `Merger` são bem mais raras no Galaxy Zoo Dataset que `Round Elliptical` — sem isso, o modelo tenderia a ignorar as classes minoritárias.

## Data Augmentation

* `RandomHorizontalFlip` e `RandomRotation(10)` — galáxias não têm uma orientação "correta"; a orientação da imagem é um artefato da captura, não uma característica da classe.
* `ColorJitter(brightness, contrast)` — simula variações de exposição entre observações astronômicas.
* Aplicado **só no split de treino**; validação e teste usam transformação sem augmentation, para medir desempenho em condições realistas.

## Normalização

* `Normalize` com média/desvio-padrão da ImageNet — obrigatório para transfer learning: a distribuição de entrada precisa bater com a que os pesos pré-treinados "esperam".

## Regularização

* **Weight decay** (AdamW) — penaliza pesos grandes, reduz overfitting.
* **Early stopping** (`EARLY_STOPPING_PATIENCE`) — interrompe o treino quando o F1 macro de validação para de melhorar, evitando treino desnecessário e overfitting tardio.
* **Batch Normalization** e **Dropout**, herdados das arquiteturas do Torchvision — não implementados por nós, mas ativos e relevantes para a estabilidade do treino.

## Loss auxiliar (GoogLeNet)

* Os classificadores auxiliares (`aux1`, `aux2`) contribuem com peso `0.3` cada na loss de treino. Técnica original da arquitetura Inception/GoogLeNet para injetar gradiente em camadas intermediárias e mitigar vanishing gradient em redes profundas. Usados só no treino; validação e teste usam exclusivamente a saída principal, para uma avaliação justa e comparável às outras arquiteturas.

## Otimização

* **AdamW** em vez de Adam — desacopla o weight decay do gradiente adaptativo, mais correto teoricamente que L2 embutido no Adam clássico.
* O otimizador recebe apenas parâmetros com `requires_grad=True`, coerente com o freeze de backbone — evita desperdiçar memória/computação com parâmetros congelados.

## Critério de seleção de modelo

* **F1 Macro de validação** (não accuracy) como critério para salvar o melhor checkpoint e para early stopping. Accuracy pode mascarar desempenho ruim em classes minoritárias; F1 macro pondera todas as classes igualmente, mais alinhado ao objetivo de comparar as arquiteturas de forma justa num dataset desbalanceado.

## Avaliação

* Métricas macro e weighted (precision, recall, F1), `classification_report` por classe e matriz de confusão — permitem diagnosticar *onde* cada modelo erra, não só *quanto*.
* Split em train/validation/test, com o conjunto de teste usado **exclusivamente** na avaliação final, nunca durante o treino ou tuning.

## Reprodutibilidade

* Seed fixa aplicada a todas as bibliotecas relevantes (`random`, `numpy`, `torch`, `torch.cuda`) e `cudnn` em modo determinístico — garante que os três modelos sejam comparados sob exatamente as mesmas condições experimentais.

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
* quantidade de parâmetros totais e treináveis (relevante com backbone congelado);
* tempo de treinamento;
* tempo de avaliação;
* utilização de memória da GPU.

Dessa forma, os resultados poderão ser utilizados para discutir os **trade-offs entre desempenho e custo computacional** das arquiteturas avaliadas no artigo científico de Inteligência Artificial II.