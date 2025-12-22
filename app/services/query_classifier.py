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
    
    def classify(self, query: str, conversation_context: dict = None) -> QueryClassificationResult:
        """
        Classify if query is SQL, RAG, or HYBRID (both) type
        
        Args:
            query: User's query string
            conversation_context: Optional conversation context for follow-up queries
            
        Returns:
            QueryClassificationResult with classification details
        """
        print("\n" + "="*60)
        print("CLASSIFYING QUERY")
        print("="*60)
        
        # Format conversation context for prompt
        context_str = ""
        if conversation_context and conversation_context.get("should_use_context"):
            context_parts = []
            if conversation_context.get("previous_queries"):
                context_parts.append(f"Previous user queries: {'; '.join(conversation_context['previous_queries'][-2:])}")
            if conversation_context.get("summary"):
                context_parts.append(f"Conversation summary: {conversation_context['summary']}")
            if context_parts:
                context_str = "\n\nCONVERSATION CONTEXT:\n" + "\n".join(context_parts)
        
        prompt = ChatPromptTemplate.from_template("""
You are a QUERY CLASSIFIER in a multi-stage data system.

SYSTEM FLOW (IMPORTANT CONTEXT):
1. You ONLY classify the user query.
2. If classified as "sql", a separate RAG step will retrieve relevant CSV file
   descriptions and schemas before writing and executing a pandas/SQL query.
3. You MUST NOT assume table or column names at this stage unless explicitly mentioned.
4. If the user refers to PREVIOUS QUERIES (e.g., "what did I just ask", "show me that again", "tell me more"),
   consider the CONVERSATION CONTEXT provided below and infer that the new query has the SAME classification
   as the previous query.

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

⚠️  **CRITICAL GUARDRAILS - DO NOT VIOLATE**:
1. NEVER assume or hallucinate column names or table structures
2. NEVER suggest database operations for questions asking about conversation history
3. NEVER make up data or execute queries without explicit user request for data retrieval
4. If query is vague or about the conversation itself, classify as "rag" to handle through documents only

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
{context}

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
        response = chain.invoke({"query": query, "context": context_str})
        
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
