# Projeto Final M38 — Streamlit VI e PyCaret

Projeto de credit scoring para cartão de crédito usando a base `credit_scoring.ftr`.

## Objetivo

Construir uma aplicação Streamlit capaz de receber uma base em CSV/FTR/Feather/Parquet, aplicar o mesmo pipeline de pré-processamento usado no treino e escorar os clientes com probabilidade de inadimplência (`score_mau`).

## Arquivos do projeto

- `Mod38Projeto_Resolvido.ipynb`: notebook com análise, treino, avaliação e preparação dos arquivos finais.
- `credit_scoring_pipeline.py`: funções e classes do pipeline de pré-processamento e escoragem.
- `train_model.py`: script para treinar e salvar o modelo.
- `model_final.pkl`: modelo final treinado.
- `app_projeto_final_m38.py`: aplicação Streamlit.
- `requirements_projeto_final_m38.txt`: dependências para rodar o app.
- `requirements_pycaret_m38.txt`: dependências opcionais para a parte de PyCaret/LightGBM.

## Como treinar o modelo

Coloque `credit_scoring.ftr` na pasta do projeto e rode:

```bash
pip install -r requirements_projeto_final_m38.txt
python train_model.py --data credit_scoring.ftr --model model_final.pkl --sample-size 150000
```

Para treinar com a base completa:

```bash
python train_model.py --data credit_scoring.ftr --model model_final.pkl --sample-size 0
```

## Como rodar o Streamlit

```bash
streamlit run app_projeto_final_m38.py
```

Depois, suba um arquivo `.csv`, `.ftr`, `.feather` ou `.parquet` com as colunas esperadas pelo modelo.

## Métricas obtidas na versão treinada com amostra de 150 mil linhas

| Base | Acurácia | AUC | Gini | KS |
|---|---:|---:|---:|---:|
| Desenvolvimento | 0.6820 | 0.7699 | 0.5397 | 0.3976 |
| OOT | 0.3675 | 0.7372 | 0.4744 | 0.3453 |

A acurácia deve ser interpretada com cuidado porque o problema é desbalanceado e a taxa de inadimplência muda bastante entre desenvolvimento e OOT. Para credit scoring, KS, AUC e Gini são mais informativos.

## Vídeo de funcionamento

Grave a tela com o app rodando e adicione o link aqui:

**Link do vídeo:** COLE_AQUI_O_LINK_DO_VIDEO

Sugestão de roteiro do vídeo:

1. Abrir o terminal e executar `streamlit run app_projeto_final_m38.py`.
2. Mostrar o navegador abrindo o app.
3. Fazer upload de uma base CSV/FTR.
4. Mostrar a prévia da base.
5. Mostrar a escoragem, as métricas e o botão de download.
6. Baixar a base escorada.

## Link para entrega

Suba este repositório no GitHub e envie ao tutor o link do repositório.
