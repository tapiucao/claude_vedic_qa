# Vedic Knowledge AI System

Um sistema de conhecimento abrangente alimentado por IA para bhakti Gaudiya Math e escrituras védicas, projetado para entender, recuperar e fornecer respostas sobre terminologia sânscrita, versos e conceitos filosóficos da literatura védica.

## Índice

1.  [Visão Geral do Sistema](#visao-geral-do-sistema)
2.  [Arquitetura do Sistema](#arquitetura-do-sistema)
3.  [Estrutura de Diretórios](#estrutura-de-diretorios)
4.  [Instalação e Configuração](#instalacao-e-configuracao)
5.  [Guia de Uso](#guia-de-uso)
    * [Interface de Linha de Comando](#interface-de-linha-de-comando)
    * [Modo Interativo](#modo-interativo)
    * [Servidor de API](#servidor-de-api)
6.  [Adicionando Conteúdo](#adicionando-conteudo)
7.  [Web Scraping](#web-scraping)
8.  [Recursos de Exportação](#recursos-de-exportacao)
9.  [Personalização](#personalizacao)
10. [Solução de Problemas](#solucao-de-problemas)
11. [Recursos Avançados](#recursos-avancados)
12. [Implantação na Nuvem](#implantacao-na-nuvem)
13. [Funcionalidades de Capítulo](#funcionalidades-de-capitulo)

## Visão Geral do Sistema

O Vedic Knowledge AI é um sistema especializado para trabalhar com escrituras védicas e textos Gaudiya Math. O sistema oferece:

* **Q&A em Linguagem Natural**: Faça perguntas sobre conceitos védicos e receba respostas detalhadas e com fontes.
* **Explicações de Termos em Sânscrito**: Obtenha explicações abrangentes de termos em sânscrito com etimologias.
* **Interpretações de Versos**: Explore os significados e interpretações de versos das escrituras védicas.
* **Web Scraping Inteligente**: Colete e integre eticamente conhecimento de sites confiáveis.
* **Capacidades de Exportação**: Gere dicionários, relatórios e logs de Q&A para estudo posterior.
* **Múltiplas Interfaces**: Opções de acesso por linha de comando, interativo e API.

## Arquitetura do Sistema

O sistema é construído em torno destes componentes principais:

* **Processamento de Documentos**: Lida com a ingestão de PDFs e estruturação de texto. Inclui processamento específico para sânscrito.
* **Base de Conhecimento**: Repositório vetorial (ChromaDB) para recuperação eficiente de conteúdo relevante, utilizando modelos de embedding configuráveis.
* **Sistema de Q&A**: Combina recuperação de informações (RAG) com modelos de linguagem generativa (Gemini) para respostas precisas, incluindo gerenciamento de citações.
* **Web Scraper**: Coleta dados de fontes confiáveis de forma ética, com gerenciamento de cache e capacidade de lidar com sites estáticos e dinâmicos (baseados em JavaScript). Inclui um agendador para atualizações periódicas.
* **Sistema de Exportação**: Gera exportações estruturadas em múltiplos formatos (JSON, Markdown, CSV).
* **Utilitários**: Inclui logging, sincronização com armazenamento em nuvem (AWS, GCP, Azure) e outras ferramentas de suporte.

## Estrutura de Diretórios

vedic-knowledge-ai/
├── .env                        # Variáveis de ambiente
├── requirements.txt            # Dependências
├── README.md                   # Documentação do projeto
├── app.py                      # Arquivo principal da aplicação (CLI e lógica central)
├── api.py                      # Interface da API (FastAPI)
├── docker-compose.yml          # Para conteinerização com ChromaDB
├── Dockerfile                  # Para conteinerização da aplicação
├── data/
│   ├── books/                  # Armazenamento de PDFs
│   │   └── [seus_pdfs_aqui]    # Coloque os PDFs aqui
│   ├── db_new/                 # Armazenamento do banco de dados vetorial (ChromaDB)
│   ├── exports/                # Exportações geradas
│   │   ├── qa_logs/            # Registros de perguntas e respostas
│   │   ├── reports/            # Relatórios do sistema
│   │   └── summaries/          # Resumos de texto e dicionários
│   ├── web_cache/              # Conteúdo web em cache
│   │   └── metadata.json       # Metadados do cache
│   └── temp/                   # Processamento temporário
└── src/
├── config.py               # Configurações (paths, chaves de API, parâmetros de LLM)
├── document_processor/     # Processamento de PDFs e texto
│   ├── init.py
│   ├── pdf_loader.py       # Carregamento de PDFs
│   ├── text_splitter.py    # Divisão de texto em chunks
│   └── sanskrit_processor.py # Manipulação de sânscrito
├── knowledge_base/         # Banco de dados vetorial
│   ├── init.py
│   ├── vector_store.py     # Operações do banco de dados vetorial (ChromaDB)
│   ├── embeddings.py       # Modelos de embedding (HuggingFace)
│   └── prompt_templates.py # Prompts especializados para o LLM
├── qa_system/              # Funcionalidade de Q&A
│   ├── init.py
│   ├── gemini_interface.py # Integração com LLM Gemini
│   ├── retriever.py        # Recuperação de documentos (RAG local e híbrido)
│   └── citation.py         # Atribuição de fontes
├── web_scraper/            # Web scraping
│   ├── init.py
│   ├── scraper.py          # Scraper básico (requests)
│   ├── dynamic_scraper.py  # Manipulador de JavaScript (Selenium)
│   ├── scheduler.py        # Agendamento de scraping
│   ├── cache_manager.py    # Cache de conteúdo web
│   └── ethics.py           # Regras éticas de scraping
├── utils/                  # Utilitários
│   ├── init.py
│   ├── exporter.py         # Funcionalidade de exportação
│   ├── logger.py           # Configuração de logging
│   └── cloud_sync.py       # Sincronização com armazenamento em nuvem
└── test_gemini_api.py      # Script de teste para a API Gemini


## Instalação e Configuração

### Pré-requisitos

* Python 3.9+
* Pelo menos 8GB de RAM recomendado
* Chave de API do Google Gemini (Generative Language API)
* Docker e Docker Compose (para usar o ChromaDB como um serviço separado)

### Passo 1: Clonar ou Criar a Estrutura de Diretórios

Crie a estrutura de pastas do projeto como mostrado acima ou clone o repositório:

```bash
git clone [https://github.com/seuusuario/vedic-knowledge-ai.git](https://github.com/seuusuario/vedic-knowledge-ai.git)
cd vedic-knowledge-ai
Passo 2: Instalar Dependências
Bash

pip install -r requirements.txt
Pode ser necessário instalar o chromedriver manualmente ou garantir que ele esteja no seu PATH se o webdriver-manager não conseguir configurá-lo automaticamente para o DynamicScraper.

Passo 3: Configurar Variáveis de Ambiente
Crie um arquivo .env na raiz do projeto e adicione sua configuração:

Code snippet

# API Keys
GEMINI_API_KEY='SUA_CHAVE_API_GEMINI_AQUI'

# Directories (geralmente não precisam ser alterados se usar a estrutura padrão)
PDF_DIR=./data/books
DB_DIR=./data/db_new
EXPORT_DIR=./data/exports
WEB_CACHE_DIR=./data/web_cache
TEMP_DIR=./data/temp

# LLM Configuration
MODEL_NAME="gemini-1.5-flash" # ou "gemini-1.5-pro", etc.
TEMPERATURE=0.2
MAX_TOKENS=2048

# Vector Database Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5 # Para busca local no VectorStore
EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2" # Mantenha consistência com o DB existente

# Web Scraping Configuration
SCRAPING_INTERVAL=86400 # Diariamente
REQUEST_DELAY=5 # Segundos
TRUSTED_WEBSITES="[https://www.purebhakti.com](https://www.purebhakti.com),[https://vedabase.io/en/](https://vedabase.io/en/)" # Sites confiáveis para scraping

# Logging Configuration
LOG_LEVEL="INFO"
# LOG_FILE=./data/vedic_knowledge_ai.log # O logger.py geralmente lida com isso

# Cloud Storage (opcional, descomente e preencha se for usar)
# AWS_ACCESS_KEY_ID=
# AWS_SECRET_ACCESS_KEY=
# AWS_REGION=us-east-1
# S3_BUCKET=
# GCP_PROJECT_ID=
# GCP_BUCKET_NAME=
# AZURE_STORAGE_CONNECTION_STRING=
# AZURE_CONTAINER_NAME=

# ChromaDB (se estiver usando docker-compose.yml para o ChromaDB)
CHROMA_HOST=chroma # Nome do serviço no docker-compose
CHROMA_PORT=8001
Passo 4: Configurar o ChromaDB (Banco de Dados Vetorial)
Recomenda-se usar o Docker Compose para executar o ChromaDB como um serviço separado para persistência e gerenciamento mais fácil.

Bash

docker-compose up -d chroma
Isso iniciará o serviço ChromaDB e o manterá em execução em segundo plano. A aplicação (vedic-knowledge-ai) se conectará a ele conforme configurado no docker-compose.yml e no .env.

Se preferir executar o ChromaDB localmente (persistência no sistema de arquivos, sem Docker para o DB), certifique-se de que CHROMA_HOST e CHROMA_PORT não estejam definidos no .env para que ele use o DB_DIR.

Passo 5: Inicializar o Sistema (Opcional, mas bom para criar pastas)
Execute o comando de inicialização para configurar a estrutura de diretórios, se ainda não existir:

Bash

python app.py init
Passo 6: Adicionar Seus Documentos PDF
Coloque seus PDFs védicos e Gaudiya Math no diretório data/books. Você pode organizá-los em subdiretórios, se desejar.

Passo 7: Carregar Documentos no Sistema
Bash

python app.py load
Este comando processará os PDFs, dividirá o texto em chunks e os adicionará ao banco de dados vetorial ChromaDB.

Guia de Uso
Interface de Linha de Comando
O sistema fornece uma interface de linha de comando abrangente (app.py):

Comandos Básicos
Bash

# Obter ajuda
python app.py -h

# Obter informações sobre o banco de dados
python app.py info
Consultando Conhecimento
Bash

# Fazer uma pergunta (usará a estratégia RAG híbrida por padrão)
python app.py answer "Qual o significado de dharma no Bhagavad Gita?"

# Fazer uma pergunta e exportar o resultado
python app.py answer "Qual o significado de dharma no Bhagavad Gita?" --export

# Explicar um termo em sânscrito
python app.py explain-term "atma"

# Explicar um verso
python app.py explain-verse "karmanye vadhikaraste ma phaleshu kadachana"

# Explicar com referência
python app.py explain-verse "karmanye vadhikaraste ma phaleshu kadachana" --reference "Bhagavad Gita 2.47"

# Consultar um termo em sânscrito no Vedabase.io e adicionar à base de conhecimento
python app.py lookup-term "ahimsa"
python app.py lookup-term "ahimsa" --export # Também exporta os dados do termo
Web Scraping
Bash

# Fazer scraping de um site (o sistema decidirá entre scraper estático/dinâmico com base na URL)
python app.py scrape "[https://www.vedabase.com/en/bg/2/47](https://www.vedabase.com/en/bg/2/47)"

# Forçar a atualização do conteúdo em cache
python app.py scrape "[https://www.vedabase.com/en/bg/2/47](https://www.vedabase.com/en/bg/2/47)" --bypass-cache

# Controlar o agendador de scraping
python app.py scheduler start
python app.py scheduler start --immediate # Executa imediatamente e depois agenda
python app.py scheduler stop
python app.py scheduler status
Gerenciamento de Cache
Bash

# Mostrar estatísticas do cache
python app.py cache stats

# Limpar entradas de cache expiradas
python app.py cache clear

# Limpar todas as entradas de cache
python app.py cache clear --all
Funções de Exportação
Bash

# Exportar dicionário de termos em sânscrito (compilado da base de conhecimento)
python app.py export terms

# Gerar relatório do sistema
python app.py export report
Funcionalidades de Capítulo (Novidade)
Bash

# Listar todos os textos e capítulos disponíveis na base de conhecimento
python app.py chapters

# Listar capítulos para um texto específico (ex: bhagavad-gita)
python app.py chapters --text "bhagavad-gita" # O text_id pode ser o nome do arquivo ou título.

# Obter conteúdo de um capítulo específico (preview)
python app.py chapter --text_id "bhagavad-gita" --chapter "2"

# Exportar conteúdo completo de um capítulo específico
python app.py chapter --text_id "bhagavad-gita" --chapter "2" --export
Modo Interativo
O modo interativo fornece uma interface fácil de usar para explorar o sistema:

Bash

python app.py interactive
Comandos disponíveis no modo interativo:

ask <pergunta>              - Fazer uma pergunta
term <termo sânscrito>      - Explicar um termo em sânscrito
verse <texto do verso>      - Explicar um verso
lookup <termo>             - Consultar um termo sânscrito no Vedabase
scrape <url>                - Fazer scraping de um site
dynamic <url>               - Fazer scraping de um site JS-heavy (geralmente automático)
cache stats                 - Mostrar estatísticas do cache
cache clear                 - Limpar entradas de cache expiradas
export terms                - Exportar dicionário de termos em sânscrito
export report               - Gerar relatório do sistema
chapters                    - Listar todos os textos e capítulos disponíveis
chapters <text_id>          - Listar capítulos para um texto específico
chapter <text_id> <cap>     - Mostrar preview do conteúdo de um capítulo específico
export chapter <text_id> <cap> - Exportar conteúdo completo de um capítulo
info                        - Mostrar informações do banco de dados
exit                        - Sair do modo interativo
Servidor de API
Para acesso programático ou construção de interfaces web, use o servidor de API (api.py):

Bash

# Primeiro, certifique-se que o serviço ChromaDB está rodando (se usar docker-compose)
# docker-compose up -d chroma

# Depois, inicie o servidor da API da aplicação
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
Isso inicia um servidor FastAPI em http://localhost:8000 com documentação interativa (Swagger UI) em http://localhost:8000/docs.

Endpoints da API incluem:

POST /answer - Responder a uma pergunta.
POST /explain/term - Explicar um termo em sânscrito.
POST /explain/verse - Explicar um verso.
POST /scrape - Fazer scraping de um site.
GET /database/info - Obter informações do banco de dados.
GET /system/health - Verificar a saúde do sistema.
GET /cache/stats - Obter estatísticas do cache web.
POST /cache/clear - Limpar o cache web.
POST /chapters/documents - Obter documentos de um capítulo específico.
POST /chapters/summary - Obter um resumo de um capítulo específico.
POST /chapters/answer - Responder a uma pergunta baseada em um capítulo específico.
GET /chapters - Listar capítulos disponíveis.
Adicionando Conteúdo
Adicionando PDFs
Coloque os arquivos PDF no diretório data/books.
Organize em subdiretórios, se desejar.
Execute python app.py load para processar e indexar os livros.
Boas práticas para PDFs:

Use PDFs de alta qualidade com camadas de texto adequadas (não apenas imagens digitalizadas).
Para textos digitalizados, execute OCR antes de adicionar ao sistema.
Padronize as convenções de nomenclatura para facilitar a referência.
Adicionando Fontes Web
Para adicionar novas fontes para scraping:

Edite src/config.py para adicionar sites confiáveis a TRUSTED_WEBSITES.
Ajuste SITE_SPECIFIC_SCRAPING_CONFIG em src/config.py se os novos sites precisarem de seletores CSS específicos para extração de conteúdo, título ou metadados.
Execute python app.py scheduler start --immediate para iniciar o scraping.
Alternativamente, faça scraping de sites individuais:

Bash

python app.py scrape "[https://www.exemplo.com/conteudo-vedico](https://www.exemplo.com/conteudo-vedico)"
Web Scraping
O sistema inclui um sistema de web scraping sofisticado com estes recursos:

Scraping Ético
Todo o web scraping segue diretrizes éticas rigorosas:

Respeita as diretivas do robots.txt.
Implementa limitação de taxa para evitar sobrecarregar os sites.
Verifica avisos de direitos autorais e termos de serviço.
Evita áreas sensíveis (páginas de login, seções de administração, etc.).
Sistema de Cache
O sistema de cache da web otimiza o desempenho e reduz a carga da rede:

Armazena conteúdo web previamente buscado por domínio.
Expira automaticamente as entradas de cache após um período configurável.
Rastreia acertos e erros de cache para monitoramento de desempenho.
Organiza o conteúdo em cache por domínio para fácil gerenciamento.
Para visualizar as estatísticas do cache:

Bash

python app.py cache stats
Manipulação de Sites Dinâmicos
Para sites pesados em JavaScript que renderizam conteúdo dinamicamente, o sistema utiliza o DynamicVedicScraper (baseado em Selenium e ChromeDriver). A seleção entre scraper estático e dinâmico para a busca de resultados ou conteúdo de artigos é feita com base nas configurações sites_requiring_dynamic_search e sites_requiring_dynamic_article_fetch no VedicRetriever.

Scraping Agendado
Para manter sua base de conhecimento atualizada:

Bash

# Iniciar scraping agendado
python app.py scheduler start

# Executar imediatamente e depois agendar
python app.py scheduler start --immediate

# Verificar status
python app.py scheduler status

# Parar agendamento
python app.py scheduler stop
Recursos de Exportação
O sistema pode exportar vários tipos de dados:

Logs de Q&A
Registros de perguntas feitas e respostas fornecidas:

Bash

python app.py answer "O que é dharma?" --export
Exporta para data/exports/qa_logs/ em formatos JSON e Markdown.

Dicionário de Termos em Sânscrito
Gere um dicionário abrangente de termos em sânscrito:

Bash

python app.py export terms
Cria um arquivo Markdown pesquisável em data/exports/summaries/ com:

Escrita Devanagari (se disponível)
Transliteração
Definição
Etimologia
Exemplos
Termos relacionados
Relatórios do Sistema
Gere relatórios sobre o status do seu sistema:

Bash

python app.py export report
Cria relatórios detalhados em data/exports/reports/ incluindo:

Estatísticas do banco de dados
Desempenho do cache
Status do scraper
Configuração do sistema
Personalização
Configuração do Banco de Dados Vetorial
Parâmetros chave no .env (ou src/config.py):

Code snippet

# Vector Database
CHUNK_SIZE=1000         # Tamanho dos chunks de texto
CHUNK_OVERLAP=200       # Sobreposição entre chunks
TOP_K_RESULTS=5         # Número de resultados a serem recuperados
Para textos sânscritos e védicos:

Considere tamanhos de chunk maiores (1000-2000) para preservar o contexto para conceitos filosóficos.
Use maior sobreposição (20-30%) para manter o contexto para versos e comentários.
Ajuste TOP_K_RESULTS com base na complexidade da consulta (3-5 para perguntas específicas, 5-10 para investigações filosóficas).
Configuração do LLM
Code snippet

# LLM Configuration
MODEL_NAME="gemini-1.5-flash" # Qual modelo Gemini usar
TEMPERATURE=0.2          # Criatividade vs. determinismo (menor é mais determinístico)
MAX_TOKENS=2048          # Comprimento máximo da resposta
Templates de Prompt
Personalize templates de prompt em src/knowledge_base/prompt_templates.py para diferentes tipos de consulta. O sistema já inclui templates para:

Conhecimento védico geral
Definições de termos em sânscrito
Explicações de versos
Comparações de conceitos
Informações históricas/biográficas
Explicações de rituais/práticas
Modelos de Embedding
Altere o modelo de embedding em src/config.py ou via variável de ambiente EMBEDDING_MODEL:

Python

# src/config.py
# Modelo padrão
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Modelos alternativos a considerar:
# - "sentence-transformers/multi-qa-mpnet-base-dot-v1" (melhor para Q&A)
# - "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" (pode lidar melhor com sânscrito)
Importante: Se você alterar o modelo de embedding após já ter carregado documentos, precisará recarregar todos os documentos para que os embeddings sejam recalculados com o novo modelo.

Solução de Problemas
Problemas Comuns
Problemas de Processamento de PDF
Sintoma: PDFs falham ao carregar ou têm texto ausente.
Soluções:
Certifique-se de que os PDFs sejam baseados em texto, não apenas imagens digitalizadas.
Execute OCR em PDFs digitalizados primeiro.
Verifique as permissões do PDF (alguns PDFs são bloqueados).
Erros do Modelo de Embedding
Sintoma: "Erro ao inicializar embeddings".
Soluções:
Verifique a conexão com a internet (o modelo pode precisar ser baixado).
Verifique se você tem RAM suficiente (pelo menos 8GB recomendado).
Tente um modelo de embedding menor em src/config.py.
Erros da API LLM (Gemini)
Sintoma: "Erro ao gerar resposta", "Permissão negada", "Modelo não encontrado".
Soluções:
Verifique sua GEMINI_API_KEY no .env.
Verifique se a API "Generative Language" está habilitada no seu projeto Google Cloud.
Certifique-se de que o MODEL_NAME configurado está disponível para sua chave de API.
Verifique sua conexão com a internet.
Problemas de Web Scraping
Sintoma: "Falha ao fazer scraping da URL".
Soluções:
Verifique se o site permite scraping (robots.txt).
O sistema tenta usar o scraper dinâmico para sites que o exigem; certifique-se de que o ChromeDriver esteja acessível.
Aumente o REQUEST_DELAY para sites com limitação de taxa.
Logs
Logs são armazenados em:

vedic_knowledge_ai.log (ou conforme configurado em src/utils/logger.py e src/config.py).
Para verificar os logs:

Bash

tail -f vedic_knowledge_ai.log # ou o nome do arquivo de log configurado
Recursos Avançados
Detecção e Processamento de Sânscrito
O sistema inclui processamento especializado para sânscrito:

Detecção automática de conteúdo sânscrito (Devanagari e IAST).
Manipulação de texto em Devanagari e transliterado.
Extração e definição de termos em sânscrito (usando IndoWordNet se disponível).
Construção de dicionário de terminologia sânscrita.
Sincronização com a Nuvem
Para backup e implantação em vários dispositivos, use a sincronização com a nuvem (src/utils/cloud_sync.py). Suporta AWS S3, Google Cloud Storage e Azure Blob Storage.
Configure no .env:

Code snippet

# AWS
# AWS_ACCESS_KEY_ID=sua_chave_de_acesso_aws
# AWS_SECRET_ACCESS_KEY=sua_chave_secreta_aws
# AWS_REGION=us-east-1
# S3_BUCKET=seu_nome_de_bucket

# GCP
# GCP_PROJECT_ID=seu_id_de_projeto_gcp
# GCP_BUCKET_NAME=seu_nome_de_bucket_gcp

# Azure
# AZURE_STORAGE_CONNECTION_STRING=sua_string_de_conexao_azure
# AZURE_CONTAINER_NAME=seu_container_azure
Dicionário de Termos em Sânscrito Personalizado
Crie um dicionário semente de termos em sânscrito usando DataExporter.export_sanskrit_terms com seus próprios dados.

Implantação na Nuvem
Implantação com Docker
Construa e execute com Docker:

Bash

# Construir a imagem Docker para a aplicação
docker-compose build vedic-knowledge-ai

# Iniciar a aplicação e o ChromaDB
docker-compose up -d
O docker-compose.yml fornecido configura a aplicação vedic-knowledge-ai e um serviço chroma para o ChromaDB, com persistência de dados para o ChromaDB usando um volume Docker.

Implantação Serverless (Ex: AWS Lambda)
Adapte api.py para criar um manipulador Lambda.
Use Lambda Layers para dependências.
Configure S3 para armazenamento de PDF.
Use API Gateway para acesso à API.
Configuração Multi-Servidor
Para grandes implantações:

Use armazenamento em nuvem compartilhado para dados.
Implante servidores de API atrás de um balanceador de carga.
Use servidores separados para web scraping e processamento de documentos.
Funcionalidades de Capítulo
O sistema Vedic Knowledge AI agora inclui funcionalidades aprimoradas para trabalhar com capítulos específicos dentro dos textos:

Listar Capítulos
Você pode listar todos os capítulos disponíveis na base de conhecimento ou filtrar por um texto específico.

CLI:

Bash

# Listar todos os capítulos de todos os textos
python app.py chapters

# Listar capítulos de um texto específico (e.g., "bhagavad-gita")
python app.py chapters --text "bhagavad-gita"
Modo Interativo:

> chapters
> chapters bhagavad-gita
API:

GET /chapters
Obter Documentos por Capítulo
Recupere todos os documentos (chunks) pertencentes a um capítulo específico de um livro.

CLI:

Bash

# Obter preview do conteúdo do capítulo 2 do "bhagavad-gita"
python app.py chapter --text_id "bhagavad-gita" --chapter "2"

# Exportar o conteúdo completo do capítulo
python app.py chapter --text_id "bhagavad-gita" --chapter "2" --export
Modo Interativo:

> chapter bhagavad-gita 2
> export chapter bhagavad-gita 2
API:

POST /chapters/documents
JSON

{
  "book": "Bhagavad Gita As It Is", // Opcional, pode ser o título do livro
  "chapter": 2,
  "limit": 50 // Opcional
}
Resumo do Capítulo
Gere um resumo para um capítulo específico usando o LLM.

API:

POST /chapters/summary
JSON

{
  "book": "Srimad Bhagavatam", // Opcional
  "chapter": 3
}
(Nota: A funcionalidade de resumo de capítulo via CLI/interativo precisaria ser adicionada a app.py se desejado, utilizando o método ai.retriever.get_chapter_summary.)

Perguntas Baseadas em Capítulos
Faça perguntas e restrinja a busca de respostas a um capítulo específico.

API:

POST /chapters/answer
JSON

{
  "question": "Descreva Arjuna no campo de batalha.",
  "book": "Bhagavad Gita As It Is", // Opcional
  "chapter": 1
}
(Nota: A funcionalidade de perguntas baseadas em capítulos via CLI/interativo pode ser alcançada usando o comando answer e especificando filtros, embora a API ofereça um endpoint dedicado.)

Consulta de Termos em Sânscrito (Vedabase.io)
O sistema Vedic Knowledge AI inclui uma funcionalidade para consulta direta de termos em sânscrito no Vedabase.io. Isso permite:

Buscar termos sânscritos no Vedabase.io.
Extrair a escrita Devanagari, significado e ocorrências.
Adicionar as informações à sua base de conhecimento.
Opcionalmente, exportar os dados para o seu dicionário de termos em sânscrito.
Uso pela Linha de Comando
Bash

# Consulta básica de termo
python app.py lookup-term "ahimsa"

# Consulta com dados atualizados (ignora o cache)
python app.py lookup-term "ahimsa" --bypass-cache

# Consulta e exporta para o dicionário de termos em sânscrito
python app.py lookup-term "ahimsa" --export
Uso no Modo Interativo
No modo interativo, use o comando lookup:

> lookup ahimsa
Este recurso melhora significativamente a capacidade do sistema de trabalhar com terminologia sânscrita, fornecendo definições autorizadas e referências escriturísticas diretamente do Vedabase.