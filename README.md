# POSimplex

O objetivo da primeira parte do trabalho é implementar o Simplex primal/dual, abrindo caminho para trabalhos futuros da disciplina de Pesquisa Operacional e desenvolvendo um entendimento maior a respeito dos métodos aprendidos. O objetivo da segunda parte é implementar os métodos de Planos de Corte e Branch & Bound para resolução de programações inteiras. O arquivo [spec-tp1.pdf](spec-tp1.pdf) detalha a especificação da primeira parte e o arquivo [spec-tp2.pdf](spec-tp2.pdf) detalha a especificação da segunda parte.

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

O programa `simplex.py` resolve programações lineares e pode ser chamado como segue (TP1):
```bash
$ ./simplex.py entrada.txt
```

Ou passando a entrada via `stdin`:
```bash
$ ./simplex.py < entrada.txt
```

As duas maneiras também são válidas para o programa `integer_programming.py`, que resolve programações inteiras (TP2):
```bash
$ ./integer_programming.py entrada.txt
$ ./integer_programming.py < entrada.txt
```

É possível ainda incluir a biblioteca `simplex` em outros programas a fim de utilizar a classe `StdLP`:
```bash
from simplex import StdLP
```
