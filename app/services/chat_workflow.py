"""
LangGraph Chat Workflow
Save as: app/services/chat_workflow.py

This implements the complete flow using LangGraph
"""

from typing import TypedDict, Literal, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from pathlib import Path
import os
import json
from datetime import datetime
from config import settings
from dotenv import load_dotenv
from services.description_vectorizer import DescriptionVectorizer
from services.document_service import DocumentService
from services.query_classifier import QueryClassifier
from services.memory_manager import ConversationMemoryManager
from services.query_decomposer import QueryDecomposer
from schemas import QueryType


# ============= STATE DEFINITION =============
class ChatState(TypedDict):
    """State that flows through the graph"""
    query: str
    department: str
    user_id: str  # Individual user identifier
    query_type: str  # "sql", "rag", or "hybrid"
    has_tables: bool  # Whether there are tables to query
    relevant_files: List[dict]
    rag_chunks: List[dict]  # Store RAG chunks separately for hybrid queries
    generated_sql: str
    sql_results: List[dict]
    rag_response: str  # Store RAG answer separately for hybrid queries
    final_response: str
    sources: List[str]
    error: str
    messages: List[dict]  # Conversation history
    conversation_summary: str  # Summarized conversation context
    conversation_context: dict  # Extracted context from conversation
    thread_id: str  # Thread/conversation ID for this user
    decomposed_query: dict  # For multi-part hybrid queries


# ============= LANGGRAPH WORKFLOW =============
class ChatWorkflow:
    """LangGraph workflow for chat"""
    
    def __init__(self):
        # Initialize LLM (Groq - FREE and fast!)
        # self.llm = ChatGoogleGenerativeAI(
        #     model="gemini-2.5-flash",
        #     temperature=0.0,
        #     google_api_key=settings.GOOGLE_API_KEY
        # )
    
        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model="llama-3.1-8b-instant",  # Fast and free
            temperature=0
        )
        
        # Initialize services
        self.query_classifier = QueryClassifier(self.llm)
        self.memory_manager = ConversationMemoryManager(self.llm)
        self.query_decomposer = QueryDecomposer(self.llm)
        self.desc_vectorizer = DescriptionVectorizer()
        self.doc_service = DocumentService()
        
        # Initialize persistent memory (JSON-based)
        app_dir = Path(__file__).parent.parent  # Go up to app/ directory
        self.memory_dir = app_dir / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.memory_dir / "conversations.json"
        
        # Load existing conversations
        self.conversations = self._load_conversations()
        print("✅ Persistent memory initialized at", self.memory_file)
        
        # Build graph
        self.graph = self._build_graph()
    
    
    def _load_conversations(self) -> dict:
        """Load conversations from JSON file"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    
    def _save_conversations(self):
        """Save conversations to JSON file"""
        with open(self.memory_file, 'w') as f:
            json.dump(self.conversations, f, indent=2, default=str)
    
    
    # ============= MEMORY MANAGEMENT =============
    def get_conversation_history(self, thread_id: str) -> List[dict]:
        """
        Retrieve conversation history from memory
        
        Args:
            thread_id: Conversation thread ID
            
        Returns:
            List of previous messages in the conversation
        """
        return self.conversations.get(thread_id, {}).get("messages", [])
    
    
    def save_message(self, thread_id: str, role: str, content: str):
        """
        Save individual message to conversation history
        
        Args:
            thread_id: Conversation thread ID
            role: "user" or "assistant"
            content: Message content
        """
        if thread_id not in self.conversations:
            self.conversations[thread_id] = {
                "messages": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        self.conversations[thread_id]["messages"].append(message)
        self.conversations[thread_id]["updated_at"] = datetime.now().isoformat()
        
        # Save to disk immediately
        self._save_conversations()
        print(f"💾 Message saved to thread {thread_id}: {role}")
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("load_memory", self.load_memory)
        workflow.add_node("process_memory", self.process_memory)
        workflow.add_node("check_meta_question", self.check_meta_question)
        workflow.add_node("classify_query", self.classify_query)
        workflow.add_node("retrieve_files", self.retrieve_files)
        workflow.add_node("generate_sql", self.generate_sql)
        workflow.add_node("execute_sql", self.execute_sql)
        workflow.add_node("format_response", self.format_response)
        workflow.add_node("handle_rag", self.handle_rag)
        workflow.add_node("generate_rag_answer", self.generate_rag_answer)
        workflow.add_node("handle_hybrid_query", self.handle_hybrid_query)
        workflow.add_node("execute_hybrid_sql", self.execute_hybrid_sql)
        workflow.add_node("generate_hybrid_answer", self.generate_hybrid_answer)
        
        # Set entry point
        workflow.set_entry_point("load_memory")
        
        # Load memory first, process it, then check for meta-questions
        workflow.add_edge("load_memory", "process_memory")
        workflow.add_edge("process_memory", "check_meta_question")
        
        # Check if it's a meta-question and route accordingly
        workflow.add_conditional_edges(
            "check_meta_question",
            self.route_after_meta_check,
            {
                "is_meta": END,  # Meta-question answered, done
                "not_meta": "classify_query"  # Regular query, proceed with classification
            }
        )
        
        # Add conditional routing
        workflow.add_conditional_edges(
            "classify_query",
            self.route_query_type,
            {
                "sql_with_tables": "retrieve_files",
                "rag": "handle_rag",
                "hybrid": "handle_hybrid_query"
            }
        )
        
        # SQL flow with fallback to RAG if no files found
        workflow.add_conditional_edges(
            "retrieve_files",
            self.route_after_file_retrieval,
            {
                "generate_sql": "generate_sql",
                "handle_rag": "handle_rag"
            }
        )
        workflow.add_edge("generate_sql", "execute_sql")
        workflow.add_edge("execute_sql", "format_response")
        workflow.add_edge("format_response", END)
        
        # RAG flow
        workflow.add_edge("handle_rag", "generate_rag_answer")
        workflow.add_edge("generate_rag_answer", END)
        
        # Hybrid flow
        workflow.add_edge("handle_hybrid_query", "execute_hybrid_sql")
        workflow.add_edge("execute_hybrid_sql", "generate_hybrid_answer")
        workflow.add_edge("generate_hybrid_answer", END)
        
        # Compile without checkpointer (using JSON-based memory)
        return workflow.compile()
    
    
    # ============= NODE 0: LOAD MEMORY =============
    def load_memory(self, state: ChatState) -> ChatState:
        """Load conversation history from database"""
        print("\n" + "="*60)
        print("NODE 0: LOADING CONVERSATION HISTORY")
        print("="*60)
        
        # Get thread_id from state (will be passed via config)
        thread_id = state.get("thread_id", "default_thread")
        
        # Retrieve conversation history from SQLite
        history = self.get_conversation_history(thread_id)
        state["messages"] = history
        
        if history:
            print(f"✅ Loaded {len(history)} previous messages from thread: {thread_id}")
        else:
            print(f"📝 Starting new conversation thread: {thread_id}")
        
        return state
    
    
    # ============= NODE 0.5: PROCESS MEMORY =============
    def process_memory(self, state: ChatState) -> ChatState:
        """Process and extract context from conversation memory"""
        print("\n" + "="*60)
        print("NODE 0.5: PROCESSING CONVERSATION MEMORY")
        print("="*60)
        
        # Extract context from conversation history
        context = self.memory_manager.process_conversation_memory(state["messages"])
        
        state["conversation_summary"] = context["summary"]
        state["conversation_context"] = context
        
        return state
    
    # ============= NODE 0.75: CHECK FOR META-QUESTIONS =============
    def check_meta_question(self, state: ChatState) -> ChatState:
        """
        Check if query is asking about conversation history (meta-question)
        If yes, answer directly without going through classification/retrieval
        """
        print("\n" + "="*60)
        print("NODE 0.75: CHECKING IF QUERY IS ABOUT CONVERSATION HISTORY")
        print("="*60)
        
        # Check if this is a meta-question
        is_meta = self.memory_manager.is_meta_question(state["query"])
        
        if is_meta:
            print("🔍 Detected meta-question about conversation history")
            print("⏭️  Skipping classification and retrieval, answering from context only\n")
            
            # Generate response from conversation context only
            response = self.memory_manager.generate_meta_response(state["conversation_context"])
            
            state["final_response"] = response
            state["query_type"] = "meta"  # Special type to indicate this was a meta-question
            state["sources"] = ["Conversation History"]
            
            return state
        
        return state
    # ============= NODE 1: CLASSIFY QUERY =============
    def classify_query(self, state: ChatState) -> ChatState:
        """Classify if query is SQL, RAG, or HYBRID (both) type"""
        print("\n" + "="*60)
        print("NODE 1: CLASSIFYING QUERY")
        print("="*60)
        
        # Use the QueryClassifier service with conversation context
        result = self.query_classifier.classify(
            query=state["query"],
            conversation_context=state.get("conversation_context")
        )
        
        state["query_type"] = result["query_type"]
        state["has_tables"] = result["has_tables"]
        
        return state
    
    
    # ============= CONDITIONAL ROUTING =============
    def route_after_meta_check(self, state: ChatState) -> Literal["is_meta", "not_meta"]:
        """Route based on whether query is a meta-question"""
        if state.get("query_type") == "meta":
            return "is_meta"
        return "not_meta"
    
    def route_query_type(self, state: ChatState) -> Literal["sql_no_tables", "sql_with_tables", "rag", "hybrid"]:
        """Route based on query type and available tables"""
        query_type = state["query_type"]
        has_tables = state.get("has_tables", False)
        
        if query_type == "sql":
            return "sql_with_tables"
        
        elif query_type == "hybrid":
            return "hybrid"
        else:  # rag
            return "rag"
    
    
    def route_after_file_retrieval(self, state: ChatState) -> Literal["generate_sql", "handle_rag"]:
        """Route based on whether files were actually retrieved"""
        # If no files found during retrieval, fallback to RAG
        if not state.get("relevant_files"):
            print("\n⚠️  No relevant files found for SQL query, falling back to RAG...\n")
            return "handle_rag"
        else:
            return "generate_sql"
    
    
    # ============= NODE 2: RETRIEVE FILES =============
    def retrieve_files(self, state: ChatState) -> ChatState:
        """Retrieve relevant CSV/XLSX files using description embeddings"""
        print("\n" + "="*60)
        print("NODE 2: RETRIEVING RELEVANT FILES")
        print("="*60)
        
        # Search descriptions to find relevant files
        relevant_files = self.desc_vectorizer.search_relevant_files(
            department=state["department"],
            query=state["query"],
            top_k=2  # Top 2 most relevant files
        )
        
        state["relevant_files"] = relevant_files
        state["sources"] = [f"{f['filename']} ({f.get('department', 'Unknown')})" for f in relevant_files]
        
        print(f"📁 Found {len(relevant_files)} relevant files:")
        for f in relevant_files:
            print(f"  - {f['filename']} (Department: {f.get('department', 'Unknown')})")
        print()
        
        return state
    
    
    # ============= NODE 3: GENERATE SQL =============
    def generate_sql(self, state: ChatState) -> ChatState:
        """Generate pandas code to query the CSV"""
        print("\n" + "="*60)
        print("NODE 3: GENERATING PANDAS CODE")
        print("="*60)
        
        if not state["relevant_files"]:
            state["error"] = "No relevant files found"
            return state
        
        # Get the most relevant file
        target_file = state["relevant_files"][0]
        
        # Load CSV to inspect columns
        df = pd.read_csv(target_file["file_path"])
        
        # Create DataFrame info for LLM
        df_info = f"""
File: {target_file['filename']}
Description: {target_file['description']}

Columns: {', '.join(df.columns)}
Total Rows: {len(df)}

Sample Data (first 3 rows):
{df.head(3).to_string()}
"""
        
        prompt = ChatPromptTemplate.from_template("""
You are a Python/Pandas expert. Generate code to answer this query.

{df_info}

User Query: {query}

Rules:
1. Use variable 'df' for the DataFrame (already loaded)
2. Store result in variable 'result'
3. Return ONLY executable Python/pandas code
4. No imports, no comments, just code
5. Handle missing values appropriately

Example:
result = df[df['salary'] > 50000][['full_name', 'salary', 'department']]

Your code:
""")
        
        chain = prompt | self.llm
        response = chain.invoke({"df_info": df_info, "query": state["query"]})
        
        # Clean code
        code = response.content.strip()
        code = code.replace("```python", "").replace("```", "").strip()
        
        state["generated_sql"] = code
        print(f"📝 Generated code:\n{code}\n")
        
        return state
    
    
    # ============= NODE 4: EXECUTE SQL =============
    def execute_sql(self, state: ChatState) -> ChatState:
        """Execute the generated pandas code"""
        print("\n" + "="*60)
        print("NODE 4: EXECUTING PANDAS CODE")
        print("="*60)
        
        try:
            # Load CSV
            target_file = state["relevant_files"][0]
            file_path = target_file["file_path"]
            
            # Handle both relative and absolute paths
            file_path_obj = Path(file_path)
            if not file_path_obj.is_absolute():
                # Convert relative path to absolute
                file_path_obj = file_path_obj.resolve()
            
            df = pd.read_csv(str(file_path_obj))
            
            # Execute code safely
            namespace = {'df': df, 'pd': pd, 'result': None}
            exec(state["generated_sql"], namespace)
            result = namespace['result']
            
            # Convert to JSON
            if isinstance(result, pd.DataFrame):
                state["sql_results"] = result.head(10).to_dict(orient='records')
                print(f"✅ Query executed: {len(result)} rows returned")
                print(f"📊 Showing first 10 rows\n")
            elif isinstance(result, pd.Series):
                # Convert Series to list of values with index
                state["sql_results"] = [{"index": idx, "value": val} for idx, val in result.items()]
                print(f"✅ Query executed: Series returned ({len(result)} values)\n")
            else:
                state["sql_results"] = [{"result": str(result)}]
                print(f"✅ Query executed: {result}\n")
            
        except Exception as e:
            state["error"] = f"Execution error: {str(e)}"
            print(f"❌ Error: {e}\n")
        
        return state
    
    
    # ============= NODE 5: FORMAT RESPONSE =============
    def format_response(self, state: ChatState) -> ChatState:
        """Format SQL results into natural language"""
        print("\n" + "="*60)
        print("NODE 5: FORMATTING RESPONSE")
        print("="*60)
        
        if state.get("error"):
            state["final_response"] = f"I encountered an error: {state['error']}"
            return state
        
        prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Convert these query results into a natural, friendly response.

User Query: {query}
Data Results: {results}

Provide a clear, concise answer. If it's a table, summarize key findings.
""")
        
        chain = prompt | self.llm
        response = chain.invoke({
            "query": state["query"],
            "results": state["sql_results"]
        })
        
        state["final_response"] = response.content
        print(f"✅ Response formatted\n")
        
        return state
    
    
    # ============= NODE 6: HANDLE RAG =============
    def handle_rag(self, state: ChatState) -> ChatState:
        """Handle RAG queries (for non-SQL questions)"""
        print("\n" + "="*60)
        print("NODE 6: HANDLING RAG QUERY")
        print("="*60)
        
        # Search for similar chunks in documents
        chunks = self.doc_service.search_similar_chunks(
            department=state["department"],
            query=state["query"],
            top_k=3
        )
        
        state["relevant_files"] = chunks
        state["sources"] = [f"{chunk.get('source', 'Unknown')} ({chunk.get('department', 'Unknown')})" for chunk in chunks]
        
        print(f"✅ Retrieved {len(chunks)} similar chunks\n")
        
        return state
    
    
    # ============= NODE 7: GENERATE RAG ANSWER =============
    def generate_rag_answer(self, state: ChatState) -> ChatState:
        """Generate answer from RAG retrieved chunks using LLM"""
        print("\n" + "="*60)
        print("NODE 7: GENERATING RAG ANSWER")
        print("="*60)
        
        if not state["relevant_files"]:
            state["final_response"] = "No relevant information found in the knowledge base."
            return state
        
        # Combine chunks into context
        context = "\n\n".join([
            f"[Source: {chunk.get('source', 'Unknown')}]\n{chunk['text']}"
            for chunk in state["relevant_files"]
        ])
        
        prompt = ChatPromptTemplate.from_template("""You are a helpful AI assistant. Based on the following retrieved documents, provide a clear and concise answer to the user's question.

Retrieved Documents:
{context}

User Question: {query}

Provide a well-summarized answer based on the retrieved information. If the information is not directly relevant, say so and explain what you found instead.""")
        
        chain = prompt | self.llm
        response = chain.invoke({
            "context": context,
            "query": state["query"]
        })
        
        state["final_response"] = response.content
        print(f"✅ RAG answer generated\n")
        
        return state
    
    
    # ============= NODE 8: HANDLE HYBRID QUERIES =============
    def handle_hybrid_query(self, state: ChatState) -> ChatState:
        """Handle HYBRID queries - intelligently decompose and handle multi-part queries"""
        print("\n" + "="*60)
        print("NODE 8: HANDLING HYBRID QUERY")
        print("="*60)
        print("🔄 Analyzing query for multiple parts...\n")
        
        # Decompose the query into parts
        available_depts = ["hr", "finance", "marketing", "engineering", "general"] if state["department"] == "all" else [state["department"]]
        
        decomposed = self.query_decomposer.decompose_hybrid_query(state["query"], available_depts)
        state["decomposed_query"] = decomposed
        
        # If single part, handle as before
        if not decomposed["is_multi_part"]:
            print("✅ Single unified query, searching all relevant sources\n")
            
            relevant_files = self.desc_vectorizer.search_relevant_files(
                department=state["department"],
                query=state["query"],
                top_k=2
            )
            
            state["relevant_files"] = relevant_files
            
            rag_chunks = self.doc_service.search_similar_chunks(
                department=state["department"],
                query=state["query"],
                top_k=3
            )
            
            state["rag_chunks"] = rag_chunks
            state["sources"] = list(set(
                [f"{f['filename']} ({f.get('department', 'Unknown')})" for f in relevant_files] + 
                [f"{chunk.get('source', 'Unknown')} ({chunk.get('department', 'Unknown')})" for chunk in rag_chunks]
            ))
            
            print(f"📁 Found {len(relevant_files)} relevant files")
            print(f"📚 Retrieved {len(rag_chunks)} similar chunks\n")
            
            return state
        
        # Multi-part query: search each part separately
        print(f"🔀 MULTI-PART QUERY DETECTED: {len(decomposed['parts'])} parts identified\n")
        
        all_relevant_files = []
        all_rag_chunks = []
        all_sources = set()
        
        for i, part in enumerate(decomposed["parts"], 1):
            print(f"  Part {i}: {part['query'][:80]}...")
            print(f"  ├─ Type: {part['query_type']}")
            print(f"  └─ Departments: {', '.join(part['relevant_departments'])}\n")
            
            # Determine department for this part
            part_dept = part['relevant_departments'][0] if part['relevant_departments'] else state["department"]
            
            # Search based on query type
            if part['query_type'] in ['sql', 'sql_and_rag']:
                # Search for structured data
                files = self.desc_vectorizer.search_relevant_files(
                    department=part_dept,
                    query=part['query'],
                    top_k=1  # Fewer per part since we're splitting
                )
                all_relevant_files.extend(files)
                all_sources.update([f"{f['filename']} ({f.get('department', 'Unknown')})" for f in files])
            
            if part['query_type'] in ['rag', 'sql_and_rag']:
                # Search for documents
                chunks = self.doc_service.search_similar_chunks(
                    department=part_dept,
                    query=part['query'],
                    top_k=2  # Fewer per part since we're splitting
                )
                all_rag_chunks.extend(chunks)
                all_sources.update([f"{chunk.get('source', 'Unknown')} ({chunk.get('department', 'Unknown')})" for chunk in chunks])
        
        state["relevant_files"] = all_relevant_files
        state["rag_chunks"] = all_rag_chunks
        state["sources"] = list(all_sources)
        
        print(f"✅ AGGREGATED RESULTS:")
        print(f"  📁 {len(all_relevant_files)} files across parts")
        print(f"  📚 {len(all_rag_chunks)} chunks across parts\n")
        
        return state
    
    
    # ============= NODE 9: EXECUTE HYBRID SQL =============
    def execute_hybrid_sql(self, state: ChatState) -> ChatState:
        """Execute SQL query in hybrid mode"""
        print("\n" + "="*60)
        print("NODE 9: EXECUTING SQL IN HYBRID MODE")
        print("="*60)
        
        # For multi-part queries, skip global execution
        # Each part will handle its own execution in generate_hybrid_answer
        decomposed = state.get("decomposed_query", {})
        if decomposed.get("is_multi_part"):
            print("⏭️  Multi-part query detected, skipping global SQL execution")
            print("   (Each part will execute its own SQL independently)\n")
            state["sql_results"] = []
            return state
        
        # Single unified query: execute SQL
        if not state["relevant_files"]:
            print("⚠️  No relevant files for SQL query\n")
            state["sql_results"] = []
            return state
        
        # Generate SQL for unified query
        target_file = state["relevant_files"][0]
        file_path = target_file["file_path"]
        
        # Handle both relative and absolute paths
        file_path_obj = Path(file_path)
        if not file_path_obj.is_absolute():
            # Convert relative path to absolute
            file_path_obj = file_path_obj.resolve()
        
        df = pd.read_csv(str(file_path_obj))
        
        df_info = f"""
File: {target_file['filename']}
Description: {target_file['description']}

Columns: {', '.join(df.columns)}
Total Rows: {len(df)}

Sample Data (first 3 rows):
{df.head(3).to_string()}
"""
        
        prompt = ChatPromptTemplate.from_template("""
You are a Python/Pandas expert. Generate code to answer this query.

{df_info}

User Query: {query}

Rules:
1. Use variable 'df' for the DataFrame (already loaded)
2. Store result in variable 'result'
3. Return ONLY executable Python/pandas code
4. No imports, no comments, just code
5. Handle missing values appropriately

Your code:
""")
        
        chain = prompt | self.llm
        response = chain.invoke({"df_info": df_info, "query": state["query"]})
        
        code = response.content.strip()
        code = code.replace("```python", "").replace("```", "").strip()
        
        state["generated_sql"] = code
        print(f"📝 Generated SQL code:\n{code}\n")
        
        # Execute the code
        try:
            namespace = {'df': df, 'pd': pd, 'result': None}
            exec(code, namespace)
            result = namespace['result']
            
            if isinstance(result, pd.DataFrame):
                state["sql_results"] = result.head(10).to_dict(orient='records')
                print(f"✅ SQL executed: {len(result)} rows returned\n")
            elif isinstance(result, pd.Series):
                # Convert Series to list of values with index
                state["sql_results"] = [{"index": idx, "value": val} for idx, val in result.items()]
                print(f"✅ SQL executed: Series returned ({len(result)} values)\n")
            else:
                state["sql_results"] = [{"result": str(result)}]
                print(f"✅ SQL executed: {result}\n")
        
        except Exception as e:
            print(f"❌ SQL Execution error: {str(e)}\n")
            state["sql_results"] = []
        
        return state
    
    
    # ============= NODE 10: GENERATE HYBRID ANSWER =============
    def generate_hybrid_answer(self, state: ChatState) -> ChatState:
        """Generate answer combining both SQL results and RAG information"""
        print("\n" + "="*60)
        print("NODE 10: GENERATING HYBRID ANSWER")
        print("="*60)
        
        decomposed = state.get("decomposed_query", {})
        
        # If multi-part, process each part and combine
        if decomposed.get("is_multi_part"):
            print(f"🔀 Generating answers for {len(decomposed['parts'])} query parts...\n")
            
            parts_results = []
            
            for i, part in enumerate(decomposed["parts"], 1):
                part_result = self._generate_part_answer(
                    part_query=part['query'],
                    part_type=part['query_type'],
                    relevant_files=[f for f in state["relevant_files"] if f.get('department') in part['relevant_departments']],
                    rag_chunks=[c for c in state["rag_chunks"] if c.get('department') in part['relevant_departments']]
                )
                parts_results.append(part_result)
            
            # Combine results from all parts
            combined_response = self.query_decomposer.combine_results(parts_results)
            state["final_response"] = combined_response
            
            print(f"✅ Hybrid answer generated from {len(parts_results)} parts\n")
            
            return state
        
        # Single unified hybrid query (original behavior)
        print("✅ Single unified hybrid query, combining all sources\n")
        
        # Get SQL data with department info from relevant_files
        sql_department_info = ""
        if state.get("relevant_files"):
            sql_sources = list(set([f.get('department', 'Unknown') for f in state["relevant_files"]]))
            sql_department_info = f"(Source Department(s): {', '.join(sql_sources)})"
        
        # Format SQL results with department context
        sql_results_text = str(state.get("sql_results", [])) if state.get("sql_results") else "No SQL results"
        sql_section = f"""DATABASE QUERY RESULTS {sql_department_info}:
{sql_results_text}"""
        
        # Get RAG context with department info
        rag_chunks_with_dept = []
        if state.get("rag_chunks"):
            for chunk in state["rag_chunks"]:
                dept = chunk.get('department', 'Unknown')
                source = chunk.get('source', 'Unknown')
                rag_chunks_with_dept.append({
                    "text": chunk['text'],
                    "source": source,
                    "department": dept
                })
        
        rag_context = "\n\n".join([
            f"[Source: {chunk['source']} | Department: {chunk['department']}]\n{chunk['text']}"
            for chunk in rag_chunks_with_dept
        ]) if rag_chunks_with_dept else "No additional context available"
        
        rag_section = f"""DOCUMENT/KNOWLEDGE BASE CONTEXT:
{rag_context}"""
        
        prompt = ChatPromptTemplate.from_template("""You are a helpful AI assistant. The user asked a comprehensive question that requires information from MULTIPLE DATA SOURCES across different departments.

{sql_section}

{rag_section}

Original User Question: {query}

IMPORTANT: This is a CROSS-DEPARTMENTAL query. You have:
- STRUCTURED DATA (tables/CSV) from one department
- CONTEXTUAL INFORMATION (documents) from another department

Your task:
1. Use the DATABASE RESULTS to answer the data-related aspects of the question
2. Use the DOCUMENT CONTEXT to answer the theoretical/informational aspects
3. Connect insights from BOTH sources to provide a complete answer
4. Clearly indicate the department and source type for each piece of information

Do NOT say "the database doesn't contain X" if the answer is in the DOCUMENT CONTEXT section - they are from different sources!
Do NOT say "documents don't contain Y" if the answer is in the DATABASE RESULTS - they are from different sources!

Be comprehensive, clear, and well-organized.""")
        
        chain = prompt | self.llm
        response = chain.invoke({
            "sql_section": sql_section,
            "rag_section": rag_section,
            "query": state["query"]
        })
        
        state["final_response"] = response.content
        print(f"✅ Hybrid answer generated\n")
        
        return state
    
    def _generate_part_answer(self, part_query: str, part_type: str, relevant_files: List[dict], rag_chunks: List[dict]) -> dict:
        """
        Generate answer for a single part of a decomposed query
        Properly execute SQL and/or RAG independently for each part
        
        Returns:
            {
                "part_query": "...",
                "part_response": "...",
                "part_type": "sql|rag|sql_and_rag",
                "sql_results": [...] or None,
                "sources_used": [...]
            }
        """
        print(f"\n    Processing: {part_query[:70]}...")
        print(f"    Type: {part_type}")
        
        sql_results = None
        rag_context = ""
        sources_used = []
        
        # PART A: Execute SQL if needed
        if part_type in ["sql", "sql_and_rag"]:
            if not relevant_files:
                print(f"    ⚠️  No SQL data files found for this part")
            else:
                print(f"    🔍 Executing SQL on: {relevant_files[0]['filename']}")
                
                try:
                    target_file = relevant_files[0]
                    file_path = target_file["file_path"]
                    
                    # Handle both relative and absolute paths
                    file_path_obj = Path(file_path)
                    if not file_path_obj.is_absolute():
                        # Convert relative path to absolute
                        file_path_obj = file_path_obj.resolve()
                    
                    df = pd.read_csv(str(file_path_obj))
                    
                    df_info = f"""File: {target_file['filename']}
Columns: {', '.join(df.columns)}
Rows: {len(df)}
Sample:
{df.head(2).to_string()}"""
                    
                    # Generate pandas code for THIS specific part
                    prompt = ChatPromptTemplate.from_template("""You are a Python/Pandas expert. Generate code to answer this query.

{df_info}

User Query: {question}

Rules:
1. Use variable 'df' for the DataFrame (already loaded)
2. Store result in variable 'result'
3. Return ONLY executable Python/pandas code
4. No imports, no comments, just code
5. Handle missing values appropriately

Your code:""")
                    
                    chain = prompt | self.llm
                    response = chain.invoke({"question": part_query, "df_info": df_info})
                    
                    code = response.content.strip()
                    code = code.replace("```python", "").replace("```", "").strip()
                    code = code.replace("```", "").strip()
                    
                    print(f"    📝 Code: {code[:60]}...")
                    
                    # Execute safely
                    namespace = {'df': df, 'pd': pd, 'result': None}
                    exec(code, namespace)
                    result = namespace['result']
                    
                    # Convert result
                    if isinstance(result, pd.DataFrame):
                        sql_results = result.head(5).to_dict(orient='records')
                        print(f"    ✅ SQL result: {len(result)} rows")
                    elif isinstance(result, pd.Series):
                        sql_results = result.to_dict()
                        print(f"    ✅ SQL result: Series with {len(result)} values")
                    else:
                        sql_results = {"value": str(result)}
                        print(f"    ✅ SQL result: {result}")
                    
                    sources_used.append(f"{target_file['filename']} (SQL)")
                    
                except Exception as e:
                    print(f"    ❌ SQL execution failed: {str(e)[:50]}")
                    sql_results = None
        
        # PART B: Get RAG context if needed
        if part_type in ["rag", "sql_and_rag"]:
            if not rag_chunks:
                print(f"    ⚠️  No documents found for this part")
            else:
                print(f"    📚 Using {len(rag_chunks)} document chunks")
                
                # Combine relevant chunks
                rag_context = "\n\n".join([
                    f"[{chunk.get('source', 'Unknown')}]\n{chunk['text']}"
                    for chunk in rag_chunks[:3]  # Use top 3 chunks
                ])
                
                sources_used.extend([c.get('source', 'Unknown') for c in rag_chunks[:2]])
        
        # PART C: Generate answer combining SQL + RAG
        if part_type == "sql" and sql_results is not None:
            # SQL only: format results as answer
            prompt = ChatPromptTemplate.from_template("""Answer this question based on the data:

Question: {question}
Data: {data}

Provide a clear, concise answer.""")
            
            chain = prompt | self.llm
            response = chain.invoke({
                "question": part_query,
                "data": str(sql_results)
            })
            part_response = response.content
        
        elif part_type == "rag" and rag_context:
            # RAG only: answer from documents
            prompt = ChatPromptTemplate.from_template("""Answer based on this information:

{context}

Question: {question}

Provide a comprehensive answer.""")
            
            chain = prompt | self.llm
            response = chain.invoke({
                "context": rag_context,
                "question": part_query
            })
            part_response = response.content
        
        elif part_type == "sql_and_rag":
            # Hybrid: combine SQL data with RAG context
            sql_text = f"Data: {str(sql_results)}" if sql_results else "No SQL data available"
            rag_text = f"Context: {rag_context}" if rag_context else "No documents available"
            
            prompt = ChatPromptTemplate.from_template("""Answer this question using both data and context:

{sql_text}

{rag_text}

Question: {question}

Provide a complete answer that uses both sources.""")
            
            chain = prompt | self.llm
            response = chain.invoke({
                "sql_text": sql_text,
                "rag_text": rag_text,
                "question": part_query
            })
            part_response = response.content
        
        else:
            # Fallback if nothing matched
            part_response = f"Could not generate answer for: {part_query[:50]}... (no data or documents found)"
        
        return {
            "part_query": part_query,
            "part_type": part_type,
            "part_response": part_response,
            "sql_results": sql_results,
            "sources_used": list(set(sources_used))
        }
    
    # ============= RUN WORKFLOW =============
    def run(self, query: str, department: str, user_id: str, thread_id: str = None) -> ChatState:
        """
        Run the complete workflow with persistent JSON-based memory per user
        
        Args:
            query: User's question
            department: Department context
            user_id: REQUIRED - Unique identifier for the user (from JWT token or session)
            thread_id: Optional - Specific conversation thread. If None, uses user_id
        
        Returns:
            ChatState with the final response
        """
        if not user_id:
            raise ValueError("user_id is required for persistent memory!")
        
        # Use user_id as thread_id if not specified (each user has separate memory)
        thread_id = thread_id or f"user_{user_id}"
        
        # Retrieve conversation history for initial state
        history = self.get_conversation_history(thread_id)
        
        initial_state = {
            "query": query,
            "department": department,
            "query_type": "",
            "has_tables": False,
            "relevant_files": [],
            "rag_chunks": [],
            "generated_sql": "",
            "sql_results": [],
            "rag_response": "",
            "final_response": "",
            "sources": [],
            "error": "",
            "messages": history,
            "conversation_summary": "",
            "conversation_context": {},
            "thread_id": thread_id,
            "user_id": user_id,
            "decomposed_query": {}
        }
        
        # Run the workflow (no config needed with JSON memory)
        result = self.graph.invoke(initial_state)
        
        # Save messages to history
        self.save_message(thread_id, "user", query)
        self.save_message(thread_id, "assistant", result.get("final_response", ""))
        
        print(f"✅ Conversation saved (user: {user_id}, thread: {thread_id})")
        
        return result