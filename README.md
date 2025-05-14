# 商业合同审查系统

本仓库包含基于知识图谱的商业合同审查系统的完整代码。

## 合同审查 - 基于知识图谱的方法

该方法超越了传统的基于文本块的检索增强生成(RAG)，专注于从合同中提取目标信息(LLM + 提示)，创建知识图谱表示(LLM + Neo4J)，一组简单的数据检索函数(使用Python的Cypher、Text to Cypher、向量搜索检索器)，最终构建一个能够处理复杂问题的问答代理(Semantic Kernel)。

下图说明了这种方法：

![4阶段方法](./images/4-stage-approach%20.png)
4阶段方法：从基于问题的提取 -> 知识图谱模型 -> 图谱检索 -> 问答代理

四个步骤是：
1. 从合同中提取相关信息(LLM + 合同)
2. 将提取的信息存储到知识图谱中(Neo4j)
3. 开发简单的知识图谱数据检索函数(Python)
4. 构建处理复杂问题的问答代理(Semantic Kernel、LLM、Neo4j)

# 你需要什么？

- 获取API密钥:
    - [OpenAI token](https://platform.openai.com/api-keys) 或
    - [Google AI (Gemini) API key](https://ai.google.dev/)
- Python 3.9+ 和 Python 虚拟环境
- 访问 Neo4j 数据库
    - Docker, Aura 或自托管
    - 数据库上运行的 GenAI 插件（在 Aura 中自动可用）

# 设置

- 克隆仓库
```
git clone https://github.com/secpo/contract-review.git
cd contract-review
```
- 创建Python虚拟环境
```
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或者在Windows上
# .venv\Scripts\activate
pip install -r requirements.txt
```
- 运行本地Neo4j实例(可选)
我们将使用Docker

但您也可以使用Aura或自托管的Neo4j实例。
如果这样做，可以跳过此步骤。只需确保您有URL、用户名和密码来访问您的Neo4j数据库

```
docker run \
    --restart always --env NEO4J_AUTH=neo4j/yourpassword \
    --publish=7474:7474 --publish=7687:7687 \
    --env NEO4J_PLUGINS='["genai","apoc"]'  neo4j:latest
```

确保将**yourpassword**替换为访问此数据库的密码

## 设置环境变量
复制 `.env.example` 文件并重命名为 `.env`，然后根据您的需求编辑该文件：

```
# 选择AI服务类型: openai 或 gemini
AI_SERVICE_TYPE=openai

# OpenAI配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL_ID=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Gemini配置
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL_ID=gemini-1.5-pro

# Neo4j配置
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
```

# 第1步：从合同中提取相关信息(LLM + 合同)

在[data](./data/input)文件夹中，您将找到3个真实的商业合同PDF文件。

这些合同来自公开可用的[Contract Understanding Atticus Dataset](https://www.atticusprojectai.org/cuad)。

我们的第一步是运行一个程序，该程序将提示AI模型（OpenAI或Gemini）回答每个合同的40多个问题。

提示将包含将提取的信息以JSON格式存储在[data/output](./data/output)下的指令。

完整提示可以在[这里](./prompts/contract_extraction_prompt.txt)找到。

## 从文档到JSON
现在支持多种文档格式：PDF、DOCX、DOC和TXT。

运行以下命令：
```
python convert-document-to-json.py
```
每个文档处理时间约为60秒

您可以查看[data/output文件夹下生成的任何json文件](./data/output/)

如果LLM生成无效的JSON，您可以在[data/debug](./data/debug/)文件夹下找到LLM返回的信息。

# 第2步：将提取的信息存储到知识图谱中(Neo4j)

将每个合同作为JSON文件后，下一步是在Neo4j中创建知识图谱。

在此之前，我们需要设计一个适合表示从合同中提取的信息的知识图谱数据模型。

## 适合我们合同的知识图谱数据模型

在我们的案例中，一个合适的知识图谱数据模型包括我们的主要实体：协议(合同)、它们的条款、合同的当事方(组织)以及它们之间的关系。

![合适的知识图谱模型](./images/schema.png)

## 主要节点和关系的一些有用属性
```
Agreement {agreement_type: STRING, contract_id: INTEGER,
          effective_date: STRING,
          renewal_term: STRING, name: STRING}
ContractClause {name: STRING, type: STRING}
ClauseType {name: STRING}
Country {name: STRING}
Excerpt {text: STRING}
Organization {name: STRING}

关系属性:
IS_PARTY_TO {role: STRING}
GOVERNED_BY_LAW {state: STRING}
HAS_CLAUSE {type: STRING}
INCORPORATED_IN {state: STRING}
```

现在，让我们从`./data/output/*.json`中的JSON文件创建知识图谱：

```
python create_graph_from_json.py
```

`create_graph_from_json` Python脚本相对容易理解。

复杂性主要在于`CREATE_GRAPH_STATEMENT`。这个CYPHER语句接收合同JSON并在Neo4j中创建相关节点和关系。

您将看到类似以下的输出：
```
Index excerptTextIndex created.
Index agreementTypeTextIndex created.
Index clauseTypeNameTextIndex created.
Index clauseNameTextIndex created.
Index organizationNameTextIndex created.
Creating index: contractIdIndex
Generating Embeddings for Contract Excerpts...
```

嵌入向量的生成大约需要1分钟完成

Python脚本完成后：
- 每个合同JSON都已上传到Neo4J知识图谱
- Agreement、ClauseTypes、Organization（Party）上的关键属性都有全文索引
- 通过使用`genai.vector.encode(excerpt.text)`生成了一个新属性Excerpt.embedding
    - 根据配置的AI服务类型，这将调用：
        - OpenAI的`text-embedding-3-small`模型（如果使用OpenAI）
        - Google的`embedding-001`模型（如果使用Gemini）
- 为Excerpt.embedding创建了一个新的向量索引

3个合同的摘录嵌入总数在30-40之间（取决于LLM在第1步中检测到的相关摘录数量）。

知识图谱中一个合同的可视化表示如下：

![单个合同作为知识图谱的可视化表示](./images/contract_graph.png)

如果您使用Docker运行Neo4j实例，可以使用[浏览器工具](http://localhost:7474/browser/)确认数据已加载。

如果您使用Aura或自托管Neo4j实例，可以使用[新Neo4j控制台](https://console-preview.neo4j.io/tools/query)中的查询工具。您可能需要登录Aura实例或手动添加到自托管数据库的连接。

# 第3步：开发简单的知识图谱数据检索函数(Python)

将合同表示在知识图谱中后，下一步是构建一些基本的数据检索函数。

这些函数是使我们能够在下一节中构建问答代理的基本构建块。

让我们定义一些基本的数据检索函数：

- 检索合同的基本详细信息（给定合同ID）
- 查找涉及特定组织的合同（给定部分组织名称）
- 查找不包含特定条款类型的合同
- 查找包含特定类型条款的合同
- 基于条款中文本（摘录）的语义相似性查找合同（例如，提到"禁止物品"使用的合同）
- 对数据库中的所有合同运行自然语言查询。例如，计算"数据库中有多少合同"的聚合查询。

建议您探索[ContractPlugin.py](./ContractPlugin.py)了解定义，以及[ContractService.py](./ContractService.py)了解每个数据检索函数的实现。

数据检索函数有三种不同风格：
- 基于Cypher的数据检索函数
    - `get_contract(self, contract_id: int) -> Annotated[Agreement, "A contract"]:`
    - `get_contracts_without_clause(self, clause_type: ClauseType) -> List[Agreement]:`
    - 这两个数据检索都是围绕简单的CYPHER语句构建的
- 向量搜索+图遍历数据检索函数
    - `get_contracts_similar_text(self, clause_text: str) -> Annotated[List[Agreement], "A list of contracts with similar text in one of their clauses"]:`
    - 此函数利用Neo4j GraphRAG包
    - 它还依赖于在"Excerpt"节点上定义的向量索引
- 文本到Cypher(T2C)数据检索函数
    - `answer_aggregation_question(self, user_question: str) -> Annotated[str, "An answer to user_question"]:`
    - 此函数利用Neo4j GraphRAG包
    - 它使用配置的AI模型（OpenAI或Gemini）生成将针对数据库执行的CYPHER语句

# 第4步：构建处理复杂问题的问答代理(Semantic Kernel、LLM、Neo4j)

有了我们的知识图谱数据检索函数，我们已准备好构建一个基于图谱的问答代理！

我们将使用Microsoft Semantic Kernel，这是一个允许开发人员将LLM函数调用与现有API和数据检索函数集成的框架。该框架使用称为`Plugins`的概念来表示内核可以执行的特定功能。在我们的案例中，"ContractPlugin"中定义的所有数据检索函数都可以被LLM用来回答有关Neo4J数据库中合同的问题。
此外，Semantic Kernel使用`Memory`概念来保存用户和代理之间的所有交互。这包括任何已调用/执行的函数的详细信息（包括所有输入和输出数据）。

可以用几行代码实现一个非常简单的基于终端的代理。

运行：
```
python test_agent.py
```
您可以尝试以下问题的变体来练习不同的数据检索函数：

- 获取有价格限制但没有保险的合同
    - 查看日志INFO，注意这需要调用我们的2个数据检索函数
- 获取有关此合同的更多详细信息
- 获取AT&T的合同
- 获取Mount Knowledge的合同
- 获取合同3
- 获取提到100个产品单位的合同
- 每个合同的平均摘录数是多少？

您可以输入`exit`结束与代理的会话。

您可以查看[test_agent.py](./test_agent.py)的完整代码。
您将找到练习每个检索函数的函数（已注释）。

对于更好看的UI，您可以尝试使用streamlit：
```
streamlit run app.py
```
浏览器显示：

![Streamlit中的代理](./images/streamlit_view.png)

# 致谢 - 合同理解Atticus数据集

本演示得以实现要感谢Contract Understanding Atticus Dataset (CUAD) v1提供的宝贵资源。

这是由The Atticus Project策划和维护的数据集。CUAD在510份商业法律合同中超过13,000个标签的广泛语料库，在经验丰富的律师监督下手动注释，对于识别合同审查中的关键法律条款，特别是在公司交易（如并购）中，起到了重要作用。

我们认可并感谢CUAD对推进法律合同分析领域的NLP研究和开发的贡献。

# 未来改进
在本演示中，我们没有微调LLM来增强其识别相关摘录的基本能力。

CUAD确实提供了标记的条款/摘录，可用于微调模型以识别这些条款的存在/缺失。
