# RAG-Anything 项目架构流程图

## 整体架构概览

```mermaid
graph TB
    subgraph "用户接口层"
        A[用户] --> B[RAGAnything 主类]
    end

    subgraph "核心模块层"
        B --> C[ProcessorMixin<br/>文档处理]
        B --> D[QueryMixin<br/>查询处理]
        B --> E[BatchMixin<br/>批处理]
        B --> F[Config<br/>配置管理]
    end

    subgraph "解析器层"
        C --> G[MineruParser]
        C --> H[DoclingParser]
        G --> I[MinerU 引擎]
        H --> J[Docling 引擎]
    end

    subgraph "多模态处理层"
        C --> K[ImageModalProcessor<br/>图像处理器]
        C --> L[TableModalProcessor<br/>表格处理器]
        C --> M[EquationModalProcessor<br/>公式处理器]
        C --> N[GenericModalProcessor<br/>通用处理器]

        K --> O[ContextExtractor<br/>上下文提取器]
        L --> O
        M --> O
        N --> O
    end

    subgraph "RAG引擎层"
        C --> P[LightRAG]
        D --> P
        P --> Q[文本分块]
        P --> R[实体提取]
        P --> S[关系提取]
    end

    subgraph "存储层"
        P --> T[(text_chunks<br/>文本块存储)]
        P --> U[(chunks_vdb<br/>块向量库)]
        P --> V[(entities_vdb<br/>实体向量库)]
        P --> W[(relationships_vdb<br/>关系向量库)]
        P --> X[(知识图谱)]
        P --> Y[(doc_status<br/>文档状态)]
        C --> Z[(parse_cache<br/>解析缓存)]
    end

    subgraph "AI模型层"
        K --> AA[Vision Model<br/>视觉模型]
        L --> AB[LLM Model<br/>语言模型]
        M --> AB
        P --> AB
        P --> AC[Embedding Model<br/>嵌入模型]
    end

    style B fill:#e1f5ff
    style P fill:#fff4e1
    style K fill:#f0e1ff
    style L fill:#f0e1ff
    style M fill:#f0e1ff
```

## 文档处理完整流程

```mermaid
flowchart TD
    Start([开始: 用户提供文档]) --> CheckCache{检查解析缓存}

    CheckCache -->|缓存命中| LoadCache[加载缓存的 content_list]
    CheckCache -->|缓存未命中| FileType{识别文件类型}

    FileType -->|PDF| ParsePDF[MinerU/Docling 解析 PDF]
    FileType -->|图像| ParseImage[MinerU 图像解析]
    FileType -->|Office<br/>doc/ppt/xls| ConvertOffice[LibreOffice 转 PDF]
    FileType -->|文本<br/>txt/md| ConvertText[ReportLab 转 PDF]

    ConvertOffice --> ParsePDF
    ConvertText --> ParsePDF

    ParsePDF --> ContentList[生成 content_list]
    ParseImage --> ContentList

    ContentList --> SaveCache[保存到解析缓存]
    LoadCache --> Separate
    SaveCache --> Separate

    Separate[分离内容<br/>separate_content] --> TextContent[text_content<br/>纯文本内容]
    Separate --> MultimodalItems[multimodal_items<br/>多模态内容列表]

    TextContent --> InsertText[插入文本到 LightRAG<br/>ainsert]

    InsertText --> TextChunk[文本分块]
    TextChunk --> TextEntity[提取文本实体和关系]
    TextEntity --> TextKG[构建文本知识图谱]

    MultimodalItems --> ProcessMultimodal{处理多模态内容<br/>按类型分组}

    ProcessMultimodal -->|image| ProcessImage[ImageModalProcessor]
    ProcessMultimodal -->|table| ProcessTable[TableModalProcessor]
    ProcessMultimodal -->|equation| ProcessEquation[EquationModalProcessor]
    ProcessMultimodal -->|other| ProcessGeneric[GenericModalProcessor]

    ProcessImage --> ExtractContext1[提取周围文本上下文]
    ProcessTable --> ExtractContext2[提取周围文本上下文]
    ProcessEquation --> ExtractContext3[提取周围文本上下文]

    ExtractContext1 --> GenerateDesc1[生成视觉描述<br/>Vision Model]
    ExtractContext2 --> GenerateDesc2[生成表格分析<br/>LLM]
    ExtractContext3 --> GenerateDesc3[生成公式解释<br/>LLM]

    GenerateDesc1 --> CreateChunks[转换为 LightRAG 块格式]
    GenerateDesc2 --> CreateChunks
    GenerateDesc3 --> CreateChunks
    ProcessGeneric --> CreateChunks

    CreateChunks --> StoreChunks[存储块到<br/>text_chunks & chunks_vdb]
    StoreChunks --> StoreEntity[存储主实体到<br/>entities_vdb & full_entities]
    StoreEntity --> ExtractEntity[批量提取子实体和关系]

    ExtractEntity --> AddBelongsTo[添加 belongs_to 关系<br/>子实体 → 主实体]
    AddBelongsTo --> MergeKG[批量合并到知识图谱<br/>merge_nodes_and_edges]

    MergeKG --> UpdateStatus[更新文档状态<br/>multimodal_processed = true]
    TextKG --> UpdateStatus

    UpdateStatus --> End([处理完成])

    style Start fill:#e8f5e9
    style End fill:#e8f5e9
    style CheckCache fill:#fff9c4
    style TextContent fill:#e3f2fd
    style MultimodalItems fill:#f3e5f5
    style UpdateStatus fill:#c8e6c9
```

## 查询流程架构

```mermaid
flowchart TD
    QueryStart([用户查询]) --> QueryType{查询类型}

    QueryType -->|纯文本查询| TextQuery[aquery]
    QueryType -->|VLM 增强查询| VLMQuery[aquery_vlm_enhanced]
    QueryType -->|多模态查询| MultimodalQuery[aquery_with_multimodal]

    TextQuery --> LightRAGQuery[LightRAG.aquery<br/>选择查询模式]

    LightRAGQuery --> Mode{查询模式}
    Mode -->|local| LocalSearch[局部搜索<br/>基于实体的上下文检索]
    Mode -->|global| GlobalSearch[全局搜索<br/>基于社区报告的检索]
    Mode -->|hybrid| HybridSearch[混合搜索<br/>结合局部和全局]
    Mode -->|naive| NaiveSearch[朴素搜索<br/>向量相似度检索]

    VLMQuery --> GetContext[获取检索提示词]
    GetContext --> RetrieveContext[检索相关上下文]
    RetrieveContext --> ExtractImages[提取图像路径]
    ExtractImages --> EncodeImages[编码图像为 base64]
    EncodeImages --> BuildVLMMessage[构建 VLM 消息<br/>图像 + 文本上下文]
    BuildVLMMessage --> CallVLM[调用 Vision Model]

    MultimodalQuery --> HasMultimodal{包含多模态内容?}
    HasMultimodal -->|是| ProcessModalContent[处理多模态内容<br/>生成描述]
    HasMultimodal -->|否| DirectQuery
    ProcessModalContent --> EnhanceQuery[增强查询文本<br/>合并描述]
    EnhanceQuery --> DirectQuery[调用 aquery]

    DirectQuery --> LightRAGQuery

    LocalSearch --> GenerateAnswer[LLM 生成答案]
    GlobalSearch --> GenerateAnswer
    HybridSearch --> GenerateAnswer
    NaiveSearch --> GenerateAnswer
    CallVLM --> GenerateAnswer

    GenerateAnswer --> QueryResult([返回查询结果])

    style QueryStart fill:#e8f5e9
    style QueryResult fill:#e8f5e9
    style QueryType fill:#fff9c4
    style Mode fill:#fff9c4
    style CallVLM fill:#f3e5f5
    style GenerateAnswer fill:#e3f2fd
```

## 数据流架构

```mermaid
flowchart LR
    subgraph "输入层"
        A1[原始文档<br/>PDF/图像/Office/文本]
    end

    subgraph "解析层"
        B1[解析器输出<br/>content_list]
        B2[text 对象]
        B3[image 对象]
        B4[table 对象]
        B5[equation 对象]
    end

    subgraph "处理层"
        C1[文本内容<br/>字符串]
        C2[多模态内容<br/>列表]
    end

    subgraph "知识表示层"
        D1[文本块<br/>text_chunks]
        D2[多模态块<br/>multimodal_chunks]
        D3[文本实体<br/>entities]
        D4[多模态主实体<br/>modal_entities]
        D5[子实体<br/>sub_entities]
        D6[普通关系<br/>relations]
        D7[belongs_to 关系]
    end

    subgraph "存储层"
        E1[(text_chunks.json)]
        E2[(chunks_vdb)]
        E3[(entities_vdb)]
        E4[(full_entities.json)]
        E5[(relationships_vdb)]
        E6[(full_relations.json)]
        E7[(知识图谱<br/>graphml)]
        E8[(doc_status.json)]
    end

    A1 --> B1
    B1 --> B2
    B1 --> B3
    B1 --> B4
    B1 --> B5

    B2 --> C1
    B3 --> C2
    B4 --> C2
    B5 --> C2

    C1 --> D1
    C1 --> D3
    C1 --> D6

    C2 --> D2
    C2 --> D4
    C2 --> D5
    C2 --> D7

    D1 --> E1
    D1 --> E2
    D2 --> E1
    D2 --> E2

    D3 --> E3
    D3 --> E4
    D4 --> E3
    D4 --> E4
    D5 --> E3
    D5 --> E4

    D6 --> E5
    D6 --> E6
    D7 --> E5
    D7 --> E6

    D3 --> E7
    D4 --> E7
    D5 --> E7
    D6 --> E7
    D7 --> E7

    E1 --> E8

    style A1 fill:#ffebee
    style B1 fill:#fff3e0
    style C1 fill:#e3f2fd
    style C2 fill:#f3e5f5
    style E7 fill:#c8e6c9
    style E8 fill:#c8e6c9
```

## 批处理流程

```mermaid
flowchart TD
    BatchStart([批处理开始<br/>文件夹路径]) --> ScanFolder[扫描文件夹<br/>递归或非递归]

    ScanFolder --> FilterFiles[过滤支持的文件类型<br/>PDF/图像/Office/文本]

    FilterFiles --> Parallel{并发处理}

    Parallel -->|文件1| Process1[process_document_complete]
    Parallel -->|文件2| Process2[process_document_complete]
    Parallel -->|文件3| Process3[process_document_complete]
    Parallel -->|文件N| ProcessN[process_document_complete]

    Process1 --> Semaphore[Semaphore 控制并发数<br/>max_concurrent_files]
    Process2 --> Semaphore
    Process3 --> Semaphore
    ProcessN --> Semaphore

    Semaphore --> Success{处理结果}

    Success -->|成功| CollectSuccess[收集成功结果<br/>文件路径 + 统计信息]
    Success -->|失败| CollectError[收集错误信息<br/>文件路径 + 错误详情]

    CollectSuccess --> Report[生成处理报告<br/>成功数/失败数/总耗时]
    CollectError --> Report

    Report --> BatchEnd([批处理完成])

    style BatchStart fill:#e8f5e9
    style BatchEnd fill:#e8f5e9
    style Semaphore fill:#fff9c4
    style Report fill:#e3f2fd
```

## 多模态内容处理详细流程

```mermaid
sequenceDiagram
    participant U as 用户
    participant RA as RAGAnything
    participant P as ModalProcessor
    participant CE as ContextExtractor
    participant VM as Vision/LLM Model
    participant LR as LightRAG
    participant KG as 知识图谱

    U->>RA: 提供包含多模态内容的文档
    RA->>RA: 解析文档 → content_list
    RA->>RA: 分离内容 → multimodal_items

    loop 处理每个多模态项
        RA->>P: 发送多模态内容
        P->>CE: 请求提取上下文
        CE->>P: 返回周围文本上下文

        alt 图像类型
            P->>VM: 调用 Vision Model<br/>图像 + 上下文
            VM->>P: 返回视觉描述
        else 表格/公式类型
            P->>VM: 调用 LLM<br/>内容 + 上下文
            VM->>P: 返回分析/解释
        end

        P->>RA: 返回描述 + 实体信息
    end

    RA->>RA: 转换为 LightRAG 块格式
    RA->>LR: 存储块到 text_chunks & chunks_vdb
    RA->>LR: 存储主实体到 entities_vdb

    RA->>LR: 批量提取子实体和关系
    LR->>RA: 返回实体和关系

    RA->>RA: 添加 belongs_to 关系
    RA->>KG: 批量合并节点和边
    KG->>RA: 更新完成

    RA->>U: 返回处理结果
```

## 知识图谱结构

```mermaid
graph LR
    subgraph "文本实体层"
        TE1[概念实体]
        TE2[人物实体]
        TE3[组织实体]
    end

    subgraph "多模态主实体层"
        ME1[Figure 1<br/>image]
        ME2[Table 2<br/>table]
        ME3[Equation 3<br/>equation]
    end

    subgraph "多模态子实体层"
        SE1[图像中的对象]
        SE2[表格中的数据点]
        SE3[公式中的变量]
    end

    TE1 -->|related_to| TE2
    TE2 -->|works_at| TE3

    TE1 -.->|illustrated_by| ME1
    TE1 -.->|quantified_by| ME2
    TE3 -.->|formulated_by| ME3

    SE1 -->|belongs_to| ME1
    SE2 -->|belongs_to| ME2
    SE3 -->|belongs_to| ME3

    SE1 -.->|represents| TE1
    SE2 -.->|measures| TE1

    style ME1 fill:#f3e5f5
    style ME2 fill:#f3e5f5
    style ME3 fill:#f3e5f5
    style SE1 fill:#fce4ec
    style SE2 fill:#fce4ec
    style SE3 fill:#fce4ec
```

## 配置系统架构

```mermaid
flowchart TD
    subgraph "配置来源"
        A1[环境变量<br/>env]
        A2[配置文件<br/>config.py]
        A3[运行时参数<br/>kwargs]
    end

    subgraph "配置层次"
        B1[RAGAnythingConfig<br/>主配置]
        B2[LightRAG Config<br/>RAG 引擎配置]
        B3[Context Config<br/>上下文配置]
        B4[Parser Config<br/>解析器配置]
    end

    subgraph "配置项"
        C1[目录配置<br/>working_dir/output_dir]
        C2[解析器配置<br/>parser/parse_method]
        C3[多模态开关<br/>enable_image/table/equation]
        C4[批处理配置<br/>max_concurrent_files]
        C5[上下文配置<br/>context_window/mode]
        C6[模型配置<br/>llm/vision/embedding]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B1

    B1 --> B2
    B1 --> B3
    B1 --> B4

    B1 --> C1
    B1 --> C2
    B1 --> C3
    B1 --> C4
    B1 --> C5
    B1 --> C6

    style B1 fill:#e3f2fd
    style C1 fill:#fff9c4
    style C2 fill:#fff9c4
    style C3 fill:#fff9c4
```

## 缓存系统架构

```mermaid
flowchart LR
    subgraph "输入"
        A[文档 + 配置]
    end

    subgraph "缓存检查"
        B{计算哈希<br/>文件内容 + 配置}
        C{检查解析缓存}
        D{检查 LLM 缓存}
    end

    subgraph "缓存存储"
        E[(parse_cache.json<br/>解析结果缓存)]
        F[(llm_response_cache.json<br/>LLM 响应缓存)]
    end

    subgraph "处理流程"
        G[解析文档]
        H[调用 LLM/Vision Model]
    end

    A --> B
    B --> C
    C -->|缓存命中| E
    C -->|缓存未命中| G
    G --> E
    E --> D
    D -->|缓存命中| F
    D -->|缓存未命中| H
    H --> F

    style E fill:#c8e6c9
    style F fill:#c8e6c9
    style C fill:#fff9c4
    style D fill:#fff9c4
```

## 项目关键技术点

### 1. Mixin 模式设计
- **QueryMixin**: 查询功能
- **ProcessorMixin**: 文档处理功能
- **BatchMixin**: 批处理功能
- **优势**: 模块化、可扩展、易维护

### 2. 多模态处理器架构
- **统一接口**: BaseModalProcessor
- **类型专用**: Image/Table/Equation/Generic
- **上下文感知**: ContextExtractor 提取周围文本
- **可扩展**: 易于添加新的模态处理器

### 3. 知识图谱结构
- **三层实体**: 文本实体 + 多模态主实体 + 子实体
- **关系类型**: 普通关系 + belongs_to 关系
- **双向链接**: 文本实体 ↔ 多模态实体

### 4. 缓存机制
- **解析缓存**: 基于文件内容 + 配置哈希
- **LLM 缓存**: LightRAG 内置
- **性能提升**: 避免重复解析和 API 调用

### 5. 批处理优化
- **并发控制**: Semaphore 限制并发数
- **进度显示**: tqdm 进度条
- **错误处理**: 收集和报告所有错误

### 6. 查询增强
- **VLM 集成**: 自动加载图像供 Vision Model 分析
- **多模态查询**: 支持查询时提供额外的多模态内容
- **混合模式**: 结合局部和全局检索

## 技术栈总结

| 层级 | 技术 | 用途 |
|------|------|------|
| **AI 模型** | OpenAI GPT-4/GPT-4V | LLM 和 Vision 推理 |
| **RAG 引擎** | LightRAG | 文本 RAG 和知识图谱 |
| **文档解析** | MinerU/Docling | 多模态文档解析 |
| **向量存储** | Nano Vector DB | 向量数据库 |
| **图存储** | NetworkX + GraphML | 知识图谱存储 |
| **并发处理** | asyncio + Semaphore | 异步并发控制 |
| **进度显示** | tqdm | 批处理进度条 |
| **配置管理** | 环境变量 + dataclass | 配置系统 |

## 项目优势

1. **全能多模态**: 统一处理文本、图像、表格、公式等
2. **上下文感知**: 处理多模态内容时考虑周围文本上下文
3. **高性能**: 缓存机制 + 批处理 + 并发处理
4. **易扩展**: Mixin 模式 + 插件式处理器架构
5. **用户友好**: 简洁 API + 详细文档 + 进度显示
6. **生产就绪**: 错误处理 + 日志记录 + 配置管理
