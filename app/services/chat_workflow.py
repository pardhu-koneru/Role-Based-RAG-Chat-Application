"""
LangGraph Chat Workflow
Save as: app/services/chat_workflow.py

This implements the complete flow using LangGraph
"""

from typing import TypedDict, Literal, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from pathlib import Path
import os
import json
from datetime import datetime
from config import settings
from services.description_vectorizer import DescriptionVectorizer
from services.document_service import DocumentService
from schemas import QueryType


# ============= STATE DEFINITION =============
class ChatState(TypedDict):
    """State that flows through the graph"""
    query: str
    department: str
    user_id: str  # Individual user identifier
    query_type: str  # "sql" or "rag"
    relevant_files: List[dict]
    generated_sql: str
    sql_results: List[dict]
    final_response: str
    sources: List[str]
    error: str
    messages: List[dict]  # Conversation history
    thread_id: str  # Thread/conversation ID for this user


# ============= LANGGRAPH WORKFLOW =============
class ChatWorkflow:
    """LangGraph workflow for chat"""
    
    def __init__(self):
        # Initialize LLM (Groq - FREE and fast!)
        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model="llama-3.1-8b-instant",  # Fast and free
            temperature=0
        )
        
        # Initialize services
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
        workflow.add_node("classify_query", self.classify_query)
        workflow.add_node("retrieve_files", self.retrieve_files)
        workflow.add_node("generate_sql", self.generate_sql)
        workflow.add_node("execute_sql", self.execute_sql)
        workflow.add_node("format_response", self.format_response)
        workflow.add_node("handle_rag", self.handle_rag)
        workflow.add_node("generate_rag_answer", self.generate_rag_answer)
        
        # Set entry point
        workflow.set_entry_point("load_memory")
        
        # Load memory first, then classify
        workflow.add_edge("load_memory", "classify_query")
        
        # Add conditional routing
        workflow.add_conditional_edges(
            "classify_query",
            self.route_query_type,
            {
                "sql": "retrieve_files",
                "rag": "handle_rag"
            }
        )
        
        # SQL flow
        workflow.add_edge("retrieve_files", "generate_sql")
        workflow.add_edge("generate_sql", "execute_sql")
        workflow.add_edge("execute_sql", "format_response")
        workflow.add_edge("format_response", END)
        
        # RAG flow
        workflow.add_edge("handle_rag", "generate_rag_answer")
        workflow.add_edge("generate_rag_answer", END)
        
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
    
    
    # ============= NODE 1: CLASSIFY QUERY =============
    def classify_query(self, state: ChatState) -> ChatState:
        """Classify if query is SQL or RAG type"""
        print("\n" + "="*60)
        print("NODE 1: CLASSIFYING QUERY")
        print("="*60)
        
        prompt = ChatPromptTemplate.from_template("""
You are a query classifier. Determine if the user wants:
1. DATA ANALYSIS (sql) - filtering, counting, aggregating data
2. GENERAL INFORMATION (rag) - explanations, policies, concepts

SQL indicators: show, list, count, how many, filter, average, sum, total, compare, find employees, get data
RAG indicators: what is, explain, tell me about, why, how does, describe, policy

Query: {query}

Return JSON:
{{
    "type": "sql" or "rag",
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}}
""")
        
        chain = prompt | self.llm | JsonOutputParser()
        result = chain.invoke({"query": state["query"]})
        
        state["query_type"] = result["type"]
        print(f"✅ Classification: {result['type']} (confidence: {result['confidence']})")
        print(f"💭 Reasoning: {result['reasoning']}\n")
        
        return state
    
    
    # ============= CONDITIONAL ROUTING =============
    def route_query_type(self, state: ChatState) -> Literal["sql", "rag"]:
        """Route based on query type"""
        return state["query_type"]
    
    
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
                state["sql_results"] = result.to_dict()
                print(f"✅ Query executed: Series returned\n")
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
            "relevant_files": [],
            "generated_sql": "",
            "sql_results": [],
            "final_response": "",
            "sources": [],
            "error": "",
            "messages": history,
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