from neo4j import GraphDatabase
from typing import List
from AgreementSchema import Agreement, ClauseType,Party, ContractClause
from neo4j_graphrag.retrievers import VectorCypherRetriever,Text2CypherRetriever
from formatters import my_vector_search_excerpt_record_formatter
import os
from AIService import get_ai_service
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class ContractSearchService:
    def __init__(self, uri, user, pwd):
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        self._driver = driver

        # 获取AI服务实例
        self.ai_service = get_ai_service()

        # 设置嵌入和LLM服务
        self._setup_embeddings_and_llm()


    async def get_contract(self, contract_id: int) -> Agreement:

        GET_CONTRACT_BY_ID_QUERY = """
            MATCH (a:Agreement {contract_id: $contract_id})-[:HAS_CLAUSE]->(clause:ContractClause)
            WITH a, collect(clause) as clauses
            MATCH (country:Country)-[i:INCORPORATED_IN]-(p:Organization)-[r:IS_PARTY_TO]-(a)
            WITH a, clauses, collect(p) as parties, collect(country) as countries, collect(r) as roles, collect(i) as states
            RETURN a as agreement, clauses, parties, countries, roles, states
        """

        agreement_node = {}

        records, _, _  = self._driver.execute_query(GET_CONTRACT_BY_ID_QUERY,{'contract_id':contract_id})


        if (len(records)==1):
            agreement_node =    records[0].get('agreement')
            party_list =        records[0].get('parties')
            role_list =         records[0].get('roles')
            country_list =      records[0].get('countries')
            state_list =        records[0].get('states')
            clause_list =       records[0].get('clauses')

        return await self._get_agreement(
            agreement_node, format="long",
            party_list=party_list, role_list=role_list,
            country_list=country_list,state_list=state_list,
            clause_list=clause_list
        )

    async def get_contracts(self, organization_name: str) -> List[Agreement]:
        GET_CONTRACTS_BY_PARTY_NAME = """
            CALL db.index.fulltext.queryNodes('organizationNameTextIndex', $organization_name)
            YIELD node AS o, score
            WITH o, score
            ORDER BY score DESC
            LIMIT 1
            WITH o
            MATCH (o)-[:IS_PARTY_TO]->(a:Agreement)
            WITH a
            MATCH (country:Country)-[i:INCORPORATED_IN]-(p:Organization)-[r:IS_PARTY_TO]-(a:Agreement)
            RETURN a as agreement, collect(p) as parties, collect(r) as roles, collect(country) as countries, collect(i) as states
        """

        #run the Cypher query
        records, _ , _ = self._driver.execute_query(GET_CONTRACTS_BY_PARTY_NAME,{'organization_name':organization_name})

        #Build the result
        all_aggrements = []
        for row in records:
            agreement_node =  row['agreement']
            party_list =  row['parties']
            role_list =  row['roles']
            country_list = row['countries']
            state_list = row['states']

            agreement : Agreement = await self._get_agreement(
                format="short",
                agreement_node=agreement_node,
                party_list=party_list,
                role_list=role_list,
                country_list=country_list,
                state_list=state_list
            )
            all_aggrements.append(agreement)

        return all_aggrements

    async def get_contracts_with_clause_type(self, clause_type: ClauseType) -> List[Agreement]:
        GET_CONTRACT_WITH_CLAUSE_TYPE_QUERY = """
            MATCH (a:Agreement)-[:HAS_CLAUSE]->(cc:ContractClause {type: $clause_type})
            WITH a
            MATCH (country:Country)-[i:INCORPORATED_IN]-(p:Organization)-[r:IS_PARTY_TO]-(a:Agreement)
            RETURN a as agreement, collect(p) as parties, collect(r) as roles, collect(country) as countries, collect(i) as states

        """
        #run the Cypher query
        records, _ , _ = self._driver.execute_query(GET_CONTRACT_WITH_CLAUSE_TYPE_QUERY,{'clause_type': str(clause_type.value)})
        # Process the results

        all_agreements = []
        for row in records:
            agreement_node =  row['agreement']
            party_list =  row['parties']
            role_list =  row['roles']
            country_list = row['countries']
            state_list = row['states']
            agreement : Agreement = await self._get_agreement(
                format="short",
                agreement_node=agreement_node,
                party_list=party_list,
                role_list=role_list,
                country_list=country_list,
                state_list=state_list
            )

            all_agreements.append(agreement)

        return all_agreements

    async def get_contracts_without_clause(self, clause_type: ClauseType) -> List[Agreement]:
        GET_CONTRACT_WITHOUT_CLAUSE_TYPE_QUERY = """
            MATCH (a:Agreement)
            OPTIONAL MATCH (a)-[:HAS_CLAUSE]->(cc:ContractClause {type: $clause_type})
            WITH a,cc
            WHERE cc is NULL
            WITH a
            MATCH (country:Country)-[i:INCORPORATED_IN]-(p:Organization)-[r:IS_PARTY_TO]-(a)
            RETURN a as agreement, collect(p) as parties, collect(r) as roles, collect(country) as countries, collect(i) as states
        """

        #run the Cypher query
        records, _ , _ = self._driver.execute_query(GET_CONTRACT_WITHOUT_CLAUSE_TYPE_QUERY,{'clause_type':clause_type.value})

        all_agreements = []
        for row in records:
            agreement_node =  row['agreement']
            party_list =  row['parties']
            role_list =  row['roles']
            country_list = row['countries']
            state_list = row['states']
            agreement : Agreement = await self._get_agreement(
                format="short",
                agreement_node=agreement_node,
                party_list=party_list,
                role_list=role_list,
                country_list=country_list,
                state_list=state_list
            )
            all_agreements.append(agreement)
        return all_agreements

    def _setup_embeddings_and_llm(self):
        """设置嵌入和LLM服务"""
        # 根据AI服务类型设置相应的嵌入和LLM服务
        service_type = os.getenv('AI_SERVICE_TYPE', 'openai').lower()

        if service_type == 'openai':
            from neo4j_graphrag.embeddings import OpenAIEmbeddings
            from neo4j_graphrag.llm import OpenAILLM

            model_id = os.getenv('OPENAI_MODEL_ID', 'gpt-4o')
            embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')

            self._embedder = OpenAIEmbeddings(model=embedding_model)
            self._llm = OpenAILLM(model_name=model_id, model_params={"temperature": 0})

        elif service_type == 'gemini':
            # 为Gemini创建自定义嵌入和LLM服务
            from neo4j_graphrag.embeddings import EmbeddingsBase
            from neo4j_graphrag.llm import LLMBase

            class GeminiEmbeddings(EmbeddingsBase):
                def __init__(self):
                    import google.generativeai as genai
                    self.api_key = os.getenv('GEMINI_API_KEY')
                    genai.configure(api_key=self.api_key)
                    self.client = genai

                def embed(self, text):
                    embedding_model = self.client.get_model('embedding-001')
                    result = embedding_model.embed_content(text)
                    return result.embedding

            class GeminiLLM(LLMBase):
                def __init__(self):
                    import google.generativeai as genai
                    self.api_key = os.getenv('GEMINI_API_KEY')
                    self.model_id = os.getenv('GEMINI_MODEL_ID', 'gemini-1.5-pro')
                    genai.configure(api_key=self.api_key)
                    self.model = genai.GenerativeModel(self.model_id)

                def generate(self, prompt, **kwargs):
                    response = self.model.generate_content(prompt)
                    return response.text

            self._embedder = GeminiEmbeddings()
            self._llm = GeminiLLM()

        else:
            raise ValueError(f"不支持的AI服务类型: {service_type}")

    async def get_contracts_similar_text(self, clause_text: str) -> List[Agreement]:
        """根据语义相似性查找合同"""
        #Cypher to traverse from the semantically similar excerpts back to the agreement
        EXCERPT_TO_AGREEMENT_TRAVERSAL_QUERY="""
            MATCH (a:Agreement)-[:HAS_CLAUSE]->(cc:ContractClause)-[:HAS_EXCERPT]-(node)
            RETURN a.name as agreement_name, a.contract_id as contract_id, cc.type as clause_type, node.text as excerpt
        """

        #Set up vector Cypher retriever
        retriever = VectorCypherRetriever(
            driver= self._driver,
            index_name="excerpt_embedding",
            embedder=self._embedder,
            retrieval_query=EXCERPT_TO_AGREEMENT_TRAVERSAL_QUERY,
            result_formatter=my_vector_search_excerpt_record_formatter
        )

        # run vector search query on excerpts and get results containing the relevant agreement and clause
        retriever_result = retriever.search(query_text=clause_text, top_k=3)

        #set up List of Agreements (with partial data) to be returned
        agreements = []
        for item in retriever_result.items:
            content = item.content
            a : Agreement = {
                'agreement_name': content['agreement_name'],
                'contract_id': content['contract_id']
            }
            c : ContractClause = {
                "clause_type": content['clause_type'],
                "excerpts" : [content['excerpt']]
            }
            a['clauses'] = [c]
            agreements.append(a)

        return agreements

    async def answer_aggregation_question(self, user_question) -> str:
        """使用Text2Cypher回答聚合问题"""
        answer = ""

        NEO4J_SCHEMA = """
            Node properties:
            Agreement {agreement_type: STRING, contract_id: INTEGER,effective_date: STRING,renewal_term: STRING, name: STRING}
            ContractClause {type: STRING}
            ClauseType {name: STRING}
            Country {name: STRING}
            Excerpt {text: STRING}
            Organization {name: STRING}

            Relationship properties:
            IS_PARTY_TO {role: STRING}
            GOVERNED_BY_LAW {state: STRING}
            HAS_CLAUSE {type: STRING}
            INCORPORATED_IN {state: STRING}

            The relationships:
            (:Agreement)-[:HAS_CLAUSE]->(:ContractClause)
            (:ContractClause)-[:HAS_EXCERPT]->(:Excerpt)
            (:ContractClause)-[:HAS_TYPE]->(:ClauseType)
            (:Agreement)-[:GOVERNED_BY_LAW]->(:Country)
            (:Organization)-[:IS_PARTY_TO]->(:Agreement)
            (:Organization)-[:INCORPORATED_IN]->(:Country)

        """

        # Initialize the retriever
        retriever = Text2CypherRetriever(
            driver=self._driver,
            llm=self._llm,
            neo4j_schema=NEO4J_SCHEMA
        )

        # Generate a Cypher query using the LLM, send it to the Neo4j database, and return the results
        retriever_result = retriever.search(query_text=user_question)

        for item in retriever_result.items:
            content = str(item.content)
            if content:
                answer += content + '\n\n'

        return answer

    async def _get_agreement (self,agreement_node, format="short", party_list=None, role_list=None,country_list=None,
                              state_list=None,clause_list=None,clause_dict=None):
        agreement : Agreement = {}

        if format == "short" and agreement_node:
            agreement: Agreement = {
                "contract_id" : agreement_node.get('contract_id'),
                "name" : agreement_node.get('name'),
                "agreement_type": agreement_node.get('agreement_type')
            }
            agreement['parties']= await self._get_parties (
                party_list=party_list,
                role_list=role_list,
                country_list=country_list,
                state_list=state_list)

        elif format=="long" and agreement_node:
            agreement: Agreement = {
                "contract_id" : agreement_node.get('contract_id'),
                "name" : agreement_node.get('name'),
                "agreement_type": agreement_node.get('agreement_type'),
                "agreement_date": agreement_node.get('agreement_date'),
                "expiration_date":  agreement_node.get('expiration_date'),
                "renewal_term": agreement_node.get('renewal_term')
            }
            agreement['parties'] = await self._get_parties (
                party_list=party_list,
                role_list=role_list,
                country_list=country_list,
                state_list=state_list)

            clauses = []
            if clause_list:
                for clause in clause_list:
                    clause : ContractClause = {"clause_type": clause.get('type')}
                    clauses.append(clause)

            elif clause_dict:

                for clause_type_key in clause_dict:
                    clause : ContractClause = {"clause_type": clause_type_key,"excerpts": clause_dict[clause_type_key]}
                    clauses.append(clause)

            agreement['clauses'] = clauses



        return agreement

    async def _get_parties (self, party_list=None, role_list=None,country_list=None,state_list=None):
        parties = []
        if party_list:
            for i in range(len(party_list)):
                p: Party = {
                    "name":  party_list[i].get('name'),
                    "role":  role_list[i].get('role'),
                    "incorporation_country": country_list[i].get('name'),
                    "incorporation_state": state_list[i].get('state')
                }
                parties.append(p)

        return parties

    async def get_contract_excerpts (self, contract_id:int):

        GET_CONTRACT_CLAUSES_QUERY = """
        MATCH (a:Agreement {contract_id: $contract_id})-[:HAS_CLAUSE]->(cc:ContractClause)-[:HAS_EXCERPT]->(e:Excerpt)
        RETURN a as agreement, cc.type as contract_clause_type, collect(e.text) as excerpts
        """
        #run CYPHER query
        clause_records, _, _  = self._driver.execute_query(GET_CONTRACT_CLAUSES_QUERY,{'contract_id':contract_id})

        #get a dict d[clause_type]=list(Excerpt)
        clause_dict = {}
        for row in clause_records:
            agreement_node = row['agreement']
            clause_type = row['contract_clause_type']
            relevant_excerpts =  row['excerpts']
            clause_dict[clause_type] = relevant_excerpts

        #Agreement to return
        agreement = await self._get_agreement(
            format="long",agreement_node=agreement_node,
            clause_dict=clause_dict)

        return agreement



