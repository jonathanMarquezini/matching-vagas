# Matching Inteligente de Vagas

> Plataforma de apoio à identificação de aderência entre colaboradores e oportunidades internas.

O **Matching Inteligente de Vagas** é uma aplicação desenvolvida para apoiar a análise e identificação de oportunidades internas compatíveis com o perfil profissional dos colaboradores.

A solução utiliza informações como **descrição profissional, ROL, taxa, localização e contexto da vaga** para calcular um nível de aderência entre o colaborador selecionado e as oportunidades disponíveis.

---

## Visão geral

O processo tradicional de identificação de oportunidades pode exigir a análise manual de diversas vagas e perfis profissionais.

Este projeto busca simplificar esse processo por meio de uma abordagem baseada em **Processamento de Linguagem Natural (NLP)**, regras de negócio e **similaridade textual**, permitindo encontrar oportunidades potencialmente compatíveis de forma mais rápida e estruturada.

### Principais critérios considerados

* **Descrição profissional**
* **ROL / nível profissional**
* **Taxa**
* **Localização do colaborador**
* **Localização da vaga**
* **Perfil solicitado pela vaga**
* **Conhecimentos funcionais**
* **Conhecimentos técnicos**
* **Projeto**
* **Contexto da oportunidade**

---

## Como funciona

O fluxo da aplicação é dividido em algumas etapas:

```text
Base de Colaboradores
        │
        ▼
Seleção do colaborador
        │
        ▼
Análise do perfil
        │
        ├── Descrição profissional
        ├── ROL
        ├── Taxa
        └── Localização
        │
        ▼
Comparação com as vagas
        │
        ├── Compatibilidade de ROL
        ├── Compatibilidade de Taxa
        ├── Compatibilidade de Localização
        └── Similaridade textual
        │
        ▼
Ranking de oportunidades
        │
        ▼
Vagas com maior aderência
```

O usuário seleciona um colaborador e a aplicação apresenta as vagas com maior compatibilidade.

---

## Matching

A lógica de matching combina diferentes critérios para gerar uma pontuação de aderência.

### 1. ROL

O ROL do colaborador é comparado ao ROL solicitado pela oportunidade, considerando o tipo profissional e o nível.

A lógica busca evitar que um colaborador seja apresentado para oportunidades incompatíveis com sua senioridade.

### 2. Taxa

A taxa do colaborador é comparada com a **taxa máxima desejável** definida para a vaga.

Vagas que ultrapassam esse limite são consideradas menos aderentes ou desconsideradas conforme as regras do matching.

### 3. Localização

A localização do colaborador é comparada com a localização associada à oportunidade.

Esse critério permite identificar se existe compatibilidade geográfica entre o profissional e a vaga, contribuindo para a priorização das oportunidades mais adequadas.

### 4. Similaridade textual

A descrição profissional do colaborador é comparada com as informações disponíveis na vaga.

Para isso, o projeto utiliza **TF-IDF** e **similaridade por cosseno**, permitindo identificar termos e contextos semelhantes entre o perfil do colaborador e a oportunidade.

### 5. Contexto da vaga

Também são consideradas informações complementares da oportunidade, como:

* Perfil profissional
* Perfil solicitado
* Conhecimentos funcionais
* Conhecimentos técnicos
* Projeto
* Necessidade
* ROL de reporte

O resultado dos diferentes critérios é utilizado para ordenar as oportunidades por aderência.

---

## Interface

A aplicação possui uma interface desenvolvida em **Streamlit**, permitindo:

* Carregar a base de vagas;
* Carregar a base de colaboradores;
* Selecionar um colaborador;
* Visualizar as oportunidades compatíveis;
* Consultar detalhes de cada vaga;
* Visualizar a pontuação de matching;
* Exportar as oportunidades encontradas para Excel.

---

## Tecnologias

| Tecnologia   | Utilização                                       |
| ------------ | ------------------------------------------------ |
| Python       | Desenvolvimento da aplicação                     |
| Streamlit    | Interface web                                    |
| Pandas       | Manipulação e tratamento dos dados               |
| Scikit-learn | TF-IDF e similaridade por cosseno                |
| OpenPyXL     | Geração de arquivos Excel                        |
| Regex        | Tratamento e interpretação de informações de ROL |

---

## Objetivo

O projeto tem como objetivo apoiar uma análise mais **ágil, estruturada e orientada por dados** na identificação de oportunidades internas.

A ferramenta considera simultaneamente aspectos **profissionais, financeiros, geográficos e contextuais** para apresentar oportunidades com maior potencial de aderência.

O sistema não substitui a avaliação profissional ou a decisão de alocação. Seu papel é atuar como um **mecanismo de apoio à análise**, destacando oportunidades que apresentam maior potencial de compatibilidade.
