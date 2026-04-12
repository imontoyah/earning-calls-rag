# Earnings Call Analyzer — Plan de Proyecto

## Contexto
Proyecto de aprendizaje para entender RAG (Retrieval-Augmented Generation) de forma práctica, usando earnings calls de empresas públicas. El objetivo es aprender los conceptos fundamentales de RAG y luego entender cómo se llevan a AWS.

## Tech Stack
- **Python 3.11+**
- **LangChain** — orquestación RAG
- **ChromaDB** — vector store local
- **sentence-transformers** (`all-MiniLM-L6-v2`) — embeddings locales (gratis, rápido)
- **Groq API** (gratis) — LLM para generación (Llama 3 o Mistral)
- **Jupyter Notebooks** — interfaz principal

## Estructura del Proyecto
```
ml_learning/
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_chunking_embeddings.ipynb
│   ├── 03_retrieval.ipynb
│   ├── 04_rag_pipeline.ipynb
│   └── 05_aws_migration.ipynb
├── src/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── chunking.py
│   ├── retrieval.py
│   └── rag_chain.py
├── data/
│   ├── raw/
│   └── processed/
├── chroma_db/
├── pyproject.toml          # uv project config + dependencies
├── .env.example
└── PLAN.md
```

---

## FASE 1: "Hello RAG" (Weekend 1)
**Objetivo:** Entender el loop básico de RAG con un solo transcript.

### [x] Paso 1.1 — Setup del proyecto
- `uv init` para inicializar el proyecto
- `uv add` para instalar dependencias
- `.env.example` con template para API keys
- Obtener API key gratuita de Groq (https://console.groq.com)

### [x] Paso 1.2 — Ingesta de datos (notebook 01)
- Descargar 1 earnings call transcript (ej. Apple Q4 2024)
- Parsear el transcript: separar por speaker turns (CEO, CFO, Analyst)
- Guardar como JSON estructurado:
  ```json
  {
    "company": "AAPL",
    "quarter": "Q4-2024",
    "date": "2024-10-31",
    "turns": [
      {"speaker": "Tim Cook", "role": "CEO", "text": "..."},
      {"speaker": "Luca Maestri", "role": "CFO", "text": "..."}
    ]
  }
  ```

### [x] Paso 1.3 — Chunking + Embeddings (notebook 02)
- Chunk por speaker turn (cada intervención = un chunk)
- Metadata por chunk: `company`, `quarter`, `speaker`, `role`, `date`
- Embeddings con `all-MiniLM-L6-v2` (corre local, no necesita API)
- Guardar en ChromaDB con metadata

### [x] Paso 1.4 — Query básico (notebook 03)
- Query semántica simple: "What did the CFO say about margins?"
- ChromaDB retrieval con `where` filter por metadata
- Ver los chunks recuperados (sin LLM todavía — solo retrieval)

### [x] Paso 1.5 — RAG completo (notebook 04)
- Conectar Groq (Llama 3 70B) como LLM via LangChain
- Prompt template que incluye contexto + pregunta
- Primera pregunta end-to-end funcionando

---

## FASE 2: "RAG Real" (Weekend 2-3)
**Objetivo:** Múltiples documentos, metadata filtering, comparaciones temporales.

### [ ] Paso 2.1 — Multi-document ingestion
- Descargar 4-8 transcripts (ej. AAPL últimos 4 quarters)
- Automatizar el pipeline de ingesta
- Re-indexar todo en ChromaDB

### [ ] Paso 2.2 — Metadata filtering avanzado
- Queries con filtros: "What did the CFO say about margins in Q3 2024?"
  - Filter: `role == "CFO"` AND `quarter == "Q3-2024"`
- `SelfQueryRetriever` de LangChain (el LLM extrae filtros de la pregunta)

### [ ] Paso 2.3 — Comparación temporal
- Query: "How has guidance on revenue changed across the last 4 quarters?"
- Retrieval por cada quarter → agregar contextos → prompt de comparación
- Chain custom en LangChain

### [ ] Paso 2.4 — Evaluación básica
- 5-10 pares pregunta/respuesta esperada
- Medir retrieval precision y answer faithfulness

---

## FASE 3: "Llevarlo a AWS" (Weekend 4-5)
**Objetivo:** Migrar componentes clave a AWS.

### [ ] Paso 3.1 — Setup AWS
- Configurar AWS CLI + IAM user con permisos mínimos
- Instalar boto3

### [ ] Paso 3.2 — S3 como storage
- Subir transcripts a S3
- Leer desde S3 en vez de disco local

### [ ] Paso 3.3 — Bedrock como LLM (reemplaza Groq)
- Activar acceso a modelos en Bedrock
- `ChatGroq` → `ChatBedrock` en LangChain

### [ ] Paso 3.4 — Bedrock Embeddings (reemplaza sentence-transformers)
- Titan Embeddings de Bedrock
- Comparar calidad vs. modelo local

### [ ] Paso 3.5 — (Opcional) OpenSearch Serverless
- Reemplazar ChromaDB por OpenSearch Serverless con k-NN

### Mapeo Local → AWS
| Local | AWS | Cambio en código |
|---|---|---|
| Archivos en `data/` | S3 bucket | `open()` → `boto3 s3.get_object()` |
| `sentence-transformers` | Bedrock Titan Embeddings | `HuggingFaceEmbeddings()` → `BedrockEmbeddings()` |
| Groq API | Bedrock (Claude/Mistral) | `ChatGroq()` → `ChatBedrock()` |
| ChromaDB | OpenSearch Serverless | `Chroma()` → `OpenSearchVectorSearch()` |

---

## FASE 4: Stretch Goals (Opcional)
- [ ] Pipeline automático con EventBridge
- [ ] Streamlit UI
- [ ] Más empresas / sectores
- [ ] Fine-tuning de embeddings para dominio financiero

---

## Notas
- **Costo AWS:** Bedrock cobra por token. Para aprendizaje, <$5/mes
- **Groq API:** Gratis con rate limit de 30 req/min
- **Embeddings locales:** `all-MiniLM-L6-v2` es gratis y rápido
