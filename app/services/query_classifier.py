"""
Query Classifier Service
Handles classification of user queries into SQL, RAG, or HYBRID types
"""

import json
from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate


class QueryClassificationResult(TypedDict):
    """Result of query classification"""
    query_type: str  # "sql", "rag", or "hybrid"
    has_tables: bool
    confidence: float
    reasoning: str


class QueryClassifier:
    """Classifies user queries into SQL, RAG, or HYBRID types"""
    
    def __init__(self, llm):
        """
        Initialize QueryClassifier
        
        Args:
            llm: Language model instance (e.g., ChatGroq)
        """
        self.llm = llm
    
    def classify(self, query: str) -> QueryClassificationResult:
        """
        Classify if query is SQL, RAG, or HYBRID (both) type
        
        Args:
            query: User's query string
            
        Returns:
            QueryClassificationResult with classification details
        """
        print("\n" + "="*60)
        print("CLASSIFYING QUERY")
        print("="*60)
        
        prompt = ChatPromptTemplate.from_template("""You are a query classifier. Determine if the user wants:
1. DATA ANALYSIS (sql) - filtering, counting, aggregating data from tables
2. GENERAL INFORMATION (rag) - explanations, policies, concepts from documents
3. HYBRID (both) - queries that need BOTH data analysis AND document information

SQL indicators: show, list, count, how many, filter, average, sum, total, compare, find employees, get data
RAG indicators: what is, explain, tell me about, why, how does, describe, policy
HYBRID indicators: "show me employees and also explain", "list data and provide context", "get results and what does this mean"

Examples:
- "Show me all employees in finance department AND explain the company hiring policy" -> hybrid
- "How many employees are in HR?" -> sql
- "What is our company policy on remote work?" -> rag
- "List all employees earning above 100k AND provide context about salary bands" -> hybrid

Query: {query}

IMPORTANT: Return ONLY valid JSON, no markdown, no extra text.
{{
    "type": "sql",
    "confidence": 0.95,
    "reasoning": "User is asking to show/filter data",
    "has_tables": true
}}""")
        
        chain = prompt | self.llm
        response = chain.invoke({"query": query})
        
        # Extract and parse JSON more robustly
        try:
            response_text = response.content.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "")
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "")
            
            response_text = response_text.strip()
            
            # Parse JSON
            result = json.loads(response_text)
            
            query_type = result.get("type", "rag")
            has_tables = result.get("has_tables", False)
            confidence = result.get("confidence", 0)
            reasoning = result.get("reasoning", "N/A")
            
            print(f"✅ Classification: {query_type} (confidence: {confidence})")
            print(f"💭 Reasoning: {reasoning}")
            print(f"📊 Has tables to query: {has_tables}\n")
            
            return {
                "query_type": query_type,
                "has_tables": has_tables,
                "confidence": confidence,
                "reasoning": reasoning
            }
            
        except json.JSONDecodeError as e:
            # Fallback: detect based on keywords
            print(f"⚠️  JSON parsing failed, using keyword detection")
            query_lower = query.lower()
            
            if any(word in query_lower for word in ["show", "list", "count", "how many", "filter", "total", "sum", "average"]):
                query_type = "sql"
                has_tables = True
            elif any(word in query_lower for word in ["and also", "and explain", "provide context", "what does"]):
                query_type = "hybrid"
                has_tables = True
            else:
                query_type = "rag"
                has_tables = False
            
            print(f"✅ Classification (fallback): {query_type}")
            print(f"📊 Has tables: {has_tables}\n")
            
            return {
                "query_type": query_type,
                "has_tables": has_tables,
                "confidence": 0.0,
                "reasoning": "Fallback keyword detection"
            }
