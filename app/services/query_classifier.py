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
        
        prompt = ChatPromptTemplate.from_template("""
You are a QUERY CLASSIFIER in a multi-stage data system.

SYSTEM FLOW (IMPORTANT CONTEXT):
1. You ONLY classify the user query.
2. If classified as "sql", a separate RAG step will retrieve relevant CSV file
   descriptions and schemas before writing and executing a pandas/SQL query.
3. You MUST NOT assume table or column names at this stage unless explicitly mentioned.

────────────────────────────────────
YOUR TASK:
Classify the query based on the INTENT of the user, not keywords.

Choose exactly ONE type:

1. "sql"
   - User intends to retrieve, filter, aggregate, or compute structured data
   - Query expects rows, columns, counts, sums, comparisons, or exact values
   - Even if the exact file or table is unknown at this stage

2. "rag"
   - User asks for explanations, summaries, reasoning, strategies, policies,
     objectives, or conceptual information
   - Answer comes from document text, NOT computed data

3. "hybrid"
   - User asks for BOTH:
     a) structured data retrieval AND
     b) explanation, reasoning, or interpretation
   - OR query is ambiguous and could require both paths

────────────────────────────────────
STRICT CLASSIFICATION RULES:

• If the expected answer is a NUMBER, LIST, ROW, or COMPUTED VALUE → SQL
• If the expected answer is TEXTUAL EXPLANATION → RAG
• If the expected answer is DATA + CONTEXT → HYBRID
• Do NOT decide based on where data exists — decide based on what the user wants
• If unsure → HYBRID with lower confidence

────────────────────────────────────
EXAMPLES:

"What is the salary of Isha Nair?" → sql
"List employees in the finance department" → sql
"How many customers were acquired in Q1?" → sql
"Explain the marketing strategy for Europe" → rag
"Why did customer acquisition drop in France?" → rag
"Show Q1 revenue and explain the shortfall" → hybrid
"Compare planned vs actual revenue and give insights" → hybrid

────────────────────────────────────
USER QUERY:
{query}

────────────────────────────────────
RETURN ONLY VALID JSON:

{{
  "type": "sql | rag | hybrid",
  "confidence": 0.0-1.0,
  "reasoning": "Short explanation focused on user intent"
}}
""")


        
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
