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


# ============= LANGGRAPH WORKFLOW =============
class ChatWorkflow:
    """LangGraph workflow for chat"""
    
    def __init__(self):
        # Initialize LLM (Groq - FREE and fast!)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0,
            google_api_key=settings.GOOGLE_API_KEY
        )
        
        # Initialize services
        self.query_classifier = QueryClassifier(self.llm)
        self.memory_manager = ConversationMemoryManager(self.llm)
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
        
        # Load memory first, process it, then classify
        workflow.add_edge("load_memory", "process_memory")
        workflow.add_edge("process_memory", "classify_query")
        
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
    
    # ============= NODE 1: CLASSIFY QUERY =============
    def classify_query(self, state: ChatState) -> ChatState:
        """Classify if query is SQL, RAG, or HYBRID (both) type"""
        print("\n" + "="*60)
        print("NODE 1: CLASSIFYING QUERY")
        print("="*60)
        
        # Use the QueryClassifier service
        result = self.query_classifier.classify(state["query"])
        
        state["query_type"] = result["query_type"]
        state["has_tables"] = result["has_tables"]
        
        return state
    
    
    # ============= CONDITIONAL ROUTING =============
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
        state["sources"] = [f["filename"] for f in relevant_files]
        
        print(f"📁 Found {len(relevant_files)} relevant files:")
        for f in relevant_files:
            print(f"  - {f['filename']}")
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
            df = pd.read_csv(target_file["file_path"])
            
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
        state["sources"] = [chunk.get("source", "Unknown") for chunk in chunks]
        
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
        """Handle HYBRID queries - queries that need both SQL and RAG"""
        print("\n" + "="*60)
        print("NODE 8: HANDLING HYBRID QUERY")
        print("="*60)
        print("🔄 This query requires BOTH data analysis AND contextual information\n")
        
        # First, check if we have relevant files/tables to query
        relevant_files = self.desc_vectorizer.search_relevant_files(
            department=state["department"],
            query=state["query"],
            top_k=2
        )
        
        state["relevant_files"] = relevant_files
        
        # Also retrieve RAG chunks
        rag_chunks = self.doc_service.search_similar_chunks(
            department=state["department"],
            query=state["query"],
            top_k=3
        )
        
        state["rag_chunks"] = rag_chunks
        state["sources"] = list(set([f["filename"] for f in relevant_files] + [chunk.get("source", "Unknown") for chunk in rag_chunks]))
        
        print(f"📁 Found {len(relevant_files)} relevant files")
        print(f"📚 Retrieved {len(rag_chunks)} similar chunks\n")
        
        return state
    
    
    # ============= NODE 9: EXECUTE HYBRID SQL =============
    def execute_hybrid_sql(self, state: ChatState) -> ChatState:
        """Execute SQL query in hybrid mode"""
        print("\n" + "="*60)
        print("NODE 9: EXECUTING SQL IN HYBRID MODE")
        print("="*60)
        
        if not state["relevant_files"]:
            print("⚠️  No relevant files for SQL query\n")
            state["sql_results"] = []
            return state
        
        # Generate SQL for hybrid mode
        target_file = state["relevant_files"][0]
        df = pd.read_csv(target_file["file_path"])
        
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
        
        # Get RAG context
        rag_context = "\n\n".join([
            f"[Source: {chunk.get('source', 'Unknown')}]\n{chunk['text']}"
            for chunk in state.get("rag_chunks", [])
        ]) if state.get("rag_chunks") else "No additional context available"
        
        # Format SQL results
        sql_results_text = str(state.get("sql_results", [])) if state.get("sql_results") else "No SQL results"
        
        prompt = ChatPromptTemplate.from_template("""You are a helpful AI assistant. The user asked a question that requires both data analysis and contextual information.

Here are the RESULTS from the database query:
{sql_results}

Here is CONTEXTUAL INFORMATION from documents:
{rag_context}

Original User Question: {query}

Provide a comprehensive answer that:
1. Summarizes the key findings from the data query
2. Provides relevant context from the documents
3. Connects the data with the contextual information to give a complete answer
4. Clearly indicates which insights come from data and which come from documentation

Be clear, concise, and well-organized.""")
        
        chain = prompt | self.llm
        response = chain.invoke({
            "sql_results": sql_results_text,
            "rag_context": rag_context,
            "query": state["query"]
        })
        
        state["final_response"] = response.content
        print(f"✅ Hybrid answer generated\n")
        
        return state
    
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
            "user_id": user_id
        }
        
        # Run the workflow (no config needed with JSON memory)
        result = self.graph.invoke(initial_state)
        
        # Save messages to history
        self.save_message(thread_id, "user", query)
        self.save_message(thread_id, "assistant", result.get("final_response", ""))
        
        print(f"✅ Conversation saved (user: {user_id}, thread: {thread_id})")
        
        return result