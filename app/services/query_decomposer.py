"""
Query Decomposer Service
Decomposes complex hybrid queries into separate data retrieval and RAG parts
Intelligently routes each part to the appropriate department/data source
"""

from typing import TypedDict, List
from langchain_core.prompts import ChatPromptTemplate


class QueryPart(TypedDict):
    """A part of a decomposed query"""
    query: str  # The sub-query text
    query_type: str  # "sql", "rag", or "sql_and_rag"
    relevant_departments: List[str]  # Which departments to search
    priority: int  # 1=high, 2=medium, 3=low


class DecomposedQuery(TypedDict):
    """Result of decomposing a complex query"""
    is_multi_part: bool  # Whether query has multiple parts
    parts: List[QueryPart]  # List of decomposed parts
    reasoning: str  # Why it was decomposed this way


class QueryDecomposer:
    """Decomposes complex hybrid queries into manageable parts"""
    
    def __init__(self, llm):
        """
        Initialize QueryDecomposer
        
        Args:
            llm: Language model instance
        """
        self.llm = llm
    
    def decompose_hybrid_query(self, query: str, available_departments: List[str]) -> DecomposedQuery:
        """
        Decompose a hybrid query into separate data retrieval and RAG parts
        Intelligently identify which parts need SQL and which need RAG
        
        Args:
            query: The user's hybrid query
            available_departments: List of available departments to search
            
        Returns:
            DecomposedQuery with identified parts and routing suggestions
        """
        print("\n" + "="*60)
        print("DECOMPOSING HYBRID QUERY")
        print("="*60)
        
        prompt = ChatPromptTemplate.from_template("""You are a query analyzer. Decompose this hybrid query into separate parts.

Available Departments: {departments}

Department Data Types:
- HR: Employee records, salary data, department assignments (CSV/Excel)
- Finance: Revenue, spending, quarterly reports, financial metrics (CSV/Excel)
- Marketing: Marketing strategies, reports, summaries, campaign data (Documents/Markdown)
- Engineering: Technical documentation, project specs (Documents/Markdown)
- General: Company policies, handbooks (Documents/Markdown)

User Query: {query}

Analyze this query and determine:
1. Does it ask for MULTIPLE different things? (multi-part)
2. For each part, should it search DATA (SQL) or DOCUMENTS (RAG)?
3. Which specific departments are relevant for each part?

Return JSON with this structure:
{{
  "is_multi_part": true/false,
  "parts": [
    {{
      "query": "specific sub-question",
      "query_type": "sql" | "rag" | "sql_and_rag",
      "relevant_departments": ["hr", "finance", etc.],
      "priority": 1
    }}
  ],
  "reasoning": "Why decomposed this way"
}}

IMPORTANT:
- SQL queries: "how many", "list", "show", "count", "total", "average", "salary", "employees"
- RAG queries: "explain", "summary", "strategy", "executive summary", "why", "what is"
- Split the query ONLY if it genuinely asks for different types of information
- Keep departments to ACTUAL relevant ones, don't guess

Return ONLY valid JSON.""")
        
        chain = prompt | self.llm
        response = chain.invoke({
            "query": query,
            "departments": ", ".join(available_departments)
        })
        
        # Parse response
        try:
            response_text = response.content.strip()
            
            # Remove markdown if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            import json
            result = json.loads(response_text)
            
            is_multi_part = result.get("is_multi_part", False)
            parts = result.get("parts", [])
            reasoning = result.get("reasoning", "")
            
            print(f"📊 Query is {'MULTI-PART' if is_multi_part else 'SINGLE QUERY'}")
            
            if is_multi_part:
                print(f"📋 Identified {len(parts)} parts:\n")
                for i, part in enumerate(parts, 1):
                    print(f"  Part {i}: {part['query'][:80]}...")
                    print(f"    Type: {part['query_type']}")
                    print(f"    Departments: {', '.join(part['relevant_departments'])}")
                    print()
            else:
                print("✅ Single unified query, will process as-is\n")
            
            return {
                "is_multi_part": is_multi_part,
                "parts": parts,
                "reasoning": reasoning
            }
            
        except Exception as e:
            print(f"⚠️  Decomposition failed: {e}")
            print(f"Treating as single query\n")
            
            # Fallback: treat as single query
            return {
                "is_multi_part": False,
                "parts": [{
                    "query": query,
                    "query_type": "hybrid",
                    "relevant_departments": available_departments,
                    "priority": 1
                }],
                "reasoning": "Fallback: treating as single query"
            }
    
    def combine_results(self, parts_results: List[dict]) -> str:
        """
        Combine results from multiple query parts into a cohesive response
        
        Args:
            parts_results: List of results from each part
                [{
                    "part_query": "...",
                    "part_type": "sql|rag|sql_and_rag",
                    "part_response": "...",
                    "sql_results": [...] or None,
                    "sources_used": [...]
                }]
        
        Returns:
            Combined response text
        """
        if len(parts_results) == 1:
            result = parts_results[0]
            print(f"✅ Single part response from {result.get('part_type', 'unknown')}")
            return result.get("part_response", "No response generated")
        
        print("\n" + "="*60)
        print(f"COMBINING RESULTS FROM {len(parts_results)} PARTS")
        print("="*60)
        
        # Build comprehensive context with all part results
        parts_info = []
        
        for i, result in enumerate(parts_results, 1):
            part_header = f"\n**Part {i}: {result['part_query'][:80]}...**"
            parts_info.append(part_header)
            
            # Include source information
            if result.get("sources_used"):
                parts_info.append(f"Sources: {', '.join(result['sources_used'])}")
            
            # Include SQL results if available
            if result.get("sql_results"):
                parts_info.append(f"Data: {str(result['sql_results'])}")
            
            # Include the generated answer
            if result.get("part_response"):
                parts_info.append(f"Answer: {result['part_response']}")
        
        combined_content = "\n".join(parts_info)
        
        prompt = ChatPromptTemplate.from_template("""You are a response synthesizer. Combine these separate answers into ONE comprehensive response.

PARTS AND THEIR ANSWERS:
{parts_content}

YOUR TASK:
1. Read all parts carefully
2. Combine them into a SINGLE coherent response (not numbered lists of answers)
3. Use clear formatting and transitions between ideas
4. Each part addresses a DIFFERENT aspect of the original question
5. Maintain accuracy - do NOT paraphrase or change the data
6. Flow naturally without seeming like separate answers concatenated

OUTPUT:
Provide ONE unified answer that addresses all parts naturally and coherently.""")
        
        chain = prompt | self.llm
        response = chain.invoke({"parts_content": combined_content})
        
        print(f"✅ Results combined from {len(parts_results)} parts\n")
        return response.content
