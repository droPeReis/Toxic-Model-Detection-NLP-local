ToChiquinho é um sistema de detecção de toxicidade para textos em português brasileiro baseado no conjunto de dados OLID-BR. Ele oferece várias funcionalidades, como política de interrupção precoce, tratamento de conjuntos de dados desequilibrados e otimização de hiperparâmetros por meio de técnicas bayesianas.

##  Instalação das Ferramentas Necessárias

- Docker: Instale o Docker em sua máquina. Você pode encontrar instruções de instalação específicas para o seu sistema operacional no site oficial do Docker.

- Docker Compose: Após instalar o Docker, adicione o Docker Compose. Geralmente, ele é instalado junto com o Docker em alguns sistemas, mas se não estiver, você pode encontrar instruções para instalação no site oficial do Docker.

- Python 3.10: Instale o Python 3.10 em sua máquina. Dependendo do seu sistema operacional, existem diferentes maneiras de instalar o Python. Você pode usar gerenciadores de pacotes como pip para isso.


## Instalação de Dependências

- Após a instalação do Python 3.10, navegue até o diretório raiz do projeto ToChiquinho no seu terminal ou prompt de comando.

- Execute os seguintes comandos para instalar as dependências necessárias:
```bash
pip install -r requirements-dev.txt
pip install -r requirements-docs.txt
pip install -r requirements.txt
```
- Instale os ganchos pré-commit para garantir que os padrões de código sejam mantidos:
```bash
  pre-commit install
```
** Condiguração de variáveis de ambiente
- Se durante a execução dos ganchos pré-commit você receber um erro ou for necessário adicionar possíveis variáveis a um arquivo de linha de base, execute o seguinte comando:
 ````bash
detect-secrets scan --baseline .secrets.baseline
````
## Execução de Testes
- Para garantir que tudo esteja configurado corretamente, execute os testes do projeto. No diretório raiz do projeto, execute:
  ````bash
  pytest
  ````
