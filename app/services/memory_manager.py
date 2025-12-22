"""
Conversation Memory Manager
Handles intelligent summarization and context extraction from conversation history
"""

import json
from typing import List, Dict, Optional, TypedDict
from langchain_core.prompts import ChatPromptTemplate


class ConversationContext(TypedDict):
    """Extracted conversation context"""
    summary: str  # Brief summary of conversation
    previous_queries: List[str]  # Previous user queries
    previous_findings: List[str]  # Key findings from previous answers
    relevant_departments: List[str]  # Departments discussed
    should_use_context: bool  # Whether context should be used


class ConversationMemoryManager:
    """
    Manages conversation history with intelligent summarization
    Extracts and maintains context from previous messages
    """
    
    # Meta-question patterns that ask about conversation history, not data
    META_QUESTION_KEYWORDS = {
        # Asking to repeat/recall previous queries
        "what did i just ask", "what did i ask", "what was my last question",
        "repeat that", "say that again", "tell me again", "what did you answer",
        "what was my previous question", "previous question", "last question",
        "what was i asking", "remind me what i asked", "summarize our conversation",
        "what have we discussed", "what did we talk about", "recap",
        
        # Asking for clarification on what was retrieved
        "what data did you retrieve", "what files did you access", "what documents",
        "what was in that query", "what did that search find", "what results",
        
        # Direct references to conversation
        "in our previous conversation", "earlier you said", "as we discussed",
    }
    
    def __init__(self, llm):
        """
        Initialize ConversationMemoryManager
        
        Args:
            llm: Language model instance (e.g., ChatGroq)
        """
        self.llm = llm
    
    def is_meta_question(self, query: str) -> bool:
        """
        Detect if query is asking about the conversation history itself
        (meta-question) rather than requesting new data
        
        Args:
            query: User's query
            
        Returns:
            bool: True if this is a meta-question about conversation history
        """
        query_lower = query.lower().strip()
        
        # Check if query matches any meta-question patterns
        for pattern in self.META_QUESTION_KEYWORDS:
            if pattern in query_lower:
                return True
        
        # Additional heuristics
        # If very short and starts with question word, likely meta
        if len(query) < 30 and query_lower.startswith(('what', 'tell', 'remind', 'repeat', 'recap')):
            return True
        
        return False
    
    def generate_meta_response(self, context: ConversationContext) -> str:
        """
        Generate response to meta-questions about conversation history
        WITHOUT executing any new queries or hallucinating data
        
        Args:
            context: ConversationContext extracted from conversation history
            
        Returns:
            Formatted summary of previous conversation
        """
        if not context["previous_queries"]:
            return "You haven't asked any questions yet in this conversation."
        
        response_parts = []
        
        # Present ONLY what was actually asked and found
        response_parts.append("📋 **Your Previous Queries:**")
        for i, query in enumerate(context["previous_queries"], 1):
            response_parts.append(f"{i}. {query}")
        
        # Include conversation summary if available
        if context["summary"]:
            response_parts.append("\n📝 **Conversation Summary:**")
            response_parts.append(context["summary"])
        
        # Include departments that were relevant
        if context["relevant_departments"]:
            response_parts.append("\n🏢 **Departments Discussed:**")
            response_parts.append(", ".join(context["relevant_departments"]))
        
        response_parts.append("\n💡 **Next Step:** Ask me a new question or request more details about the above topics.")
        
        return "\n".join(response_parts)
    
    def should_summarize(self, messages: List[dict]) -> bool:
        """
        Decide if conversation needs summarization
        
        Only summarize longer conversations to save tokens
        
        Args:
            messages: List of conversation messages
            
        Returns:
            bool: True if summarization is needed
        """
        # Only summarize if more than 5 messages (conversation is getting long)
        if len(messages) > 5:
            return True
        return False
    
    def extract_key_facts(self, messages: List[dict]) -> List[str]:
        """
        Extract key facts from recent messages without LLM call
        
        Args:
            messages: List of conversation messages
            
        Returns:
            List of key facts/findings
        """
        facts = []
        
        # Look at last 4 messages for recent context
        recent_messages = messages[-4:] if len(messages) > 4 else messages
        
        for msg in recent_messages:
            if msg["role"] == "assistant":
                content = msg.get("content", "")
                # Extract numbers and key phrases
                if any(word in content.lower() for word in ["employees", "salary", "q1", "q2", "q3", "q4", "spending", "total", "average"]):
                    # Keep first 150 chars of meaningful findings
                    facts.append(content[:150])
        
        return facts
    
    def summarize_conversation(self, messages: List[dict]) -> str:
        """
        Generate intelligent summary of conversation using LLM
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Summarized conversation context
        """
        if not messages:
            return ""
        
        # Format messages for LLM
        message_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content'][:200]}"  # Limit to 200 chars per message
            for msg in messages[-6:]  # Only use last 6 messages to save tokens
        ])
        
        prompt = ChatPromptTemplate.from_template("""You are a conversation analyst. Summarize the key points from this conversation in 2-3 sentences.

Previous Conversation:
{messages}

Focus on:
1. What the user asked about
2. What data/information was retrieved
3. Any specific departments or time periods mentioned

Keep it concise and factual.""")
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({"messages": message_text})
            return response.content.strip()
        except Exception as e:
            print(f"⚠️  Summarization failed: {e}")
            return ""
    
    def get_previous_queries(self, messages: List[dict]) -> List[str]:
        """
        Extract previous user queries from conversation history
        
        Args:
            messages: List of conversation messages
            
        Returns:
            List of previous user queries
        """
        queries = []
        for msg in messages:
            if msg["role"] == "user":
                query = msg.get("content", "").strip()
                if query and len(query) > 5:  # Only meaningful queries
                    queries.append(query)
        
        # Return last 3 queries to keep context manageable
        return queries[-3:] if len(queries) > 3 else queries
    
    def process_conversation_memory(self, messages: List[dict]) -> ConversationContext:
        """
        Process conversation memory and extract useful context
        
        Args:
            messages: List of conversation messages
            
        Returns:
            ConversationContext with summarized information
        """
        print("\n" + "="*60)
        print("PROCESSING CONVERSATION MEMORY")
        print("="*60)
        
        if not messages:
            print("📝 No previous conversation history\n")
            return {
                "summary": "",
                "previous_queries": [],
                "previous_findings": [],
                "relevant_departments": [],
                "should_use_context": False
            }
        
        print(f"📚 Processing {len(messages)} messages from conversation history\n")
        
        # Extract previous queries
        previous_queries = self.get_previous_queries(messages)
        
        # Extract key facts without extra LLM call
        previous_findings = self.extract_key_facts(messages)
        
        # Decide if we need full summarization
        needs_summary = self.should_summarize(messages)
        summary = ""
        
        if needs_summary:
            print("🔄 Conversation is long, generating summary...")
            summary = self.summarize_conversation(messages)
            print(f"✅ Summary: {summary[:100]}...\n")
        else:
            print(f"ℹ️  Short conversation ({len(messages)} messages), using recent context\n")
        
        # Extract departments mentioned
        departments = self._extract_departments(messages)
        
        context = {
            "summary": summary,
            "previous_queries": previous_queries,
            "previous_findings": previous_findings,
            "relevant_departments": departments,
            "should_use_context": len(messages) > 2  # Use context if more than 2 messages
        }
        
        return context
    
    def _extract_departments(self, messages: List[dict]) -> List[str]:
        """
        Extract department names mentioned in conversation
        
        Args:
            messages: List of conversation messages
            
        Returns:
            List of relevant departments
        """
        departments = set()
        valid_departments = {'finance', 'marketing', 'hr', 'engineering', 'sales', 'design'}
        
        for msg in messages:
            content = msg.get("content", "").lower()
            for dept in valid_departments:
                if dept in content:
                    departments.add(dept)
        
        return list(departments)
    
    def format_context_for_prompt(self, context: ConversationContext) -> str:
        """
        Format extracted context for use in LLM prompts
        
        Args:
            context: ConversationContext from process_conversation_memory
            
        Returns:
            Formatted string for inclusion in prompts
        """
        if not context["should_use_context"]:
            return ""
        
        parts = []
        
        if context["summary"]:
            parts.append(f"Previous conversation: {context['summary']}")
        
        if context["previous_queries"]:
            parts.append(f"Previous queries: {', '.join(context['previous_queries'][-2:])}")
        
        if context["relevant_departments"]:
            parts.append(f"Departments in focus: {', '.join(context['relevant_departments'])}")
        
        if parts:
            return "\n".join(parts)
        return ""
