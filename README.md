# Classificação de distúrbios de qualidade de energia com aprendizado de máquina

Código que reproduz todos os números, tabelas e figuras do artigo
**"Aprendizado de máquina aplicado à detecção e classificação de distúrbios de qualidade de energia elétrica, incluindo harmônicos"**.

O script gera sinais sintéticos de tensão (60 Hz), adiciona ruído gaussiano branco, extrai seis atributos
(RMS por ciclo e THD) e compara uma árvore de decisão, uma SVM (RBF) e uma MLP.

## Requisitos

- Python 3.13 (testado; versões 3.10 ou superiores devem funcionar)
- Bibliotecas:

```
pip install scikit-learn numpy pandas scipy matplotlib
```

Atenção: o pacote se chama `scikit-learn`, e não `sklearn`.

## Como executar

```
python simulacao_qualidade_energia.py
```

Tempo de execução: cerca de 10 minutos em um computador comum.
Os resultados são gravados na pasta `out/`, criada ao lado do script.

Também é possível rodar só uma parte, passando os nomes das etapas:

```
python simulacao_qualidade_energia.py main thd
```

| Etapa | O que faz |
|---|---|
| `main` | Experimento principal: 3 classificadores, 6 níveis de SNR, 10 repetições, IC de 95 %, métricas por classe, testes pareados e matrizes de confusão |
| `thd` | Estudo da THD: eventos de ciclos inteiros e fracionários, e desvio de frequência |
| `robustez` | Ruído de teste diferente do de treino |
| `robustez_bcd` | Eventos fracionários, distúrbios combinados e intensidades |
| `ablacao` | Subconjuntos de atributos (SVM, 20 e 10 dB) |
| `extra` | Figura da distribuição da THD por classe |
| `imp` | Importância dos atributos na árvore (Gini e permutação) |

## Arquivos gerados em `out/`

- `results.json`: todos os números usados no artigo
- `tabela_acuracia.csv`, `tabela_metricas_por_classe.csv`, `testes_pareados.csv`
- `tabela_ablacao_atributos_SVM.csv`, `tabela_importancia_atributos_arvore.csv`
- `resultados_brutos.csv`, `metricas_por_classe_brutas.csv`
- Figuras `fig1` a `fig6` (PNG)

## Reprodutibilidade

- Todas as sementes aleatórias são fixas. Executar o script de novo dá exatamente os mesmos números.
- Os parâmetros de geração dos sinais (amplitudes, durações, frequência, amostras por ciclo) e os hiperparâmetros
  dos classificadores estão no início do arquivo e na função `models`.
- Valores de referência (acurácia média de 10 repetições): SVM 99,0 % a 20 dB, 87,9 % a 10 dB e 76,3 % a 5 dB.

## Limitações

Os sinais são sintéticos e o ruído é gaussiano branco. Os resultados não comprovam desempenho em instalações reais.

## Autor

Matheus Stilhano. Orientador: Prof. André Ferrus Filho. Universidade São Judas Tadeu.

