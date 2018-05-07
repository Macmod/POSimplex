# TP1 - Simplex

O objetivo deste trabalho é implementar o Simplex primal/dual, abrindo caminho para trabalhos futuros da disciplina de Pesquisa Operacional e desenvolvendo um entendimento maior a respeito dos métodos aprendidos. O arquivo [spec.pdf](spec.pdf) detalha a especificação do trabalho.

A biblioteca `fractions` é utilizada para suporte a frações. A saída é dada em frações e a entrada pode conter expressões a serem interpertadas pela `numpy`.

A única dependência é a `numpy`.

# Instalação

As dependências podem ser obtidas com:
```bash
$ pipenv install
```

E o ambiente ativado com:
```bash
$ pipenv shell
```

# Uso

O programa pode ser chamado como especificado:
```bash
$ ./simplex.py entrada.txt
```

Ou passando a entrada via `stdin`:
```bash
$ ./simplex.py < entrada.txt
```

É possível ainda incluir a biblioteca `simplex` em outros programas a fim de utilizar a classe `StdLP`:
```bash
from simplex import StdLP
```
