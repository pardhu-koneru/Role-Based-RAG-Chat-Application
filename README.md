# 🔐 Role-Based RAG Chat Application with Hybrid Query Processing

This project is an **enterprise-grade Retrieval-Augmented Generation (RAG) system** that supports **role-based access control (RBAC)** and **hybrid query execution** over both **structured (CSV / tabular)** and **unstructured (documents)** data.

Unlike traditional RAG systems that treat all inputs as unstructured text, this system dynamically **understands the nature of a user query** and routes it to the most appropriate data source—while strictly enforcing access permissions.

---

## 🚀 Key Capabilities

- 🔐 JWT-based authentication and authorization  
- 👥 Role- and department-based data access control  
- 🧠 LLM-powered query intent classification  
- 🔀 Hybrid query processing (SQL + RAG)  
- ⚡ Parallel execution of multi-part queries  
- 📊 Safe execution of SQL-like analytics using Pandas  
- 📚 Vector-based document retrieval (RAG)  
- 🧩 Answer synthesis with confidence handling  
- 💾 Conversation context and memory management  

---

## 🧠 Problem This System Solves

Most RAG implementations assume:
- All data is unstructured
- All users have equal access
- One retrieval strategy fits all queries

In real organizations:
- Business metrics live in **structured data** (HR, Finance, Sales)
- Policies and reports live in **documents**
- Access is restricted by **roles and departments**

This system is built to reflect those real-world constraints and requirements.

---

## 📊 Types of Data Supported

### Structured Data
- CSV files
- Tabular datasets
- Metrics and aggregates
- Queried using Pandas (SQL-like operations)

### Unstructured Data
- Reports
- Policies
- PDFs / Markdown / Text documents
- Retrieved using vector embeddings (RAG)

---

## 👑 Role-Based Access Control (RBAC)

Access to data is strictly controlled:

- **Admin & C-level users**
  - Can access data across all departments
  - Can execute full hybrid queries (SQL + RAG)

- **Other users**
  - Can only access data belonging to their own department
  - Queries are automatically restricted at retrieval time

This ensures **enterprise-grade security** and prevents unauthorized data leakage.

---

## ⚙️ High-Level System Flow

1. A user submits a query
2. The system authenticates the user using **JWT**
3. Role and department permissions are validated
4. Previous conversation context is loaded (if available)
5. The query intent is classified as:
   - SQL
   - RAG
   - HYBRID
6. The query is routed dynamically based on intent
7. Data is retrieved and/or computed
8. Results are synthesized into a single response
9. The response is returned with context awareness
10. Conversation state is stored for future interactions

---

## 🔍 Query Classification Logic

The system uses an LLM to classify each query into one of three categories:

### SQL Query
- Requires numerical computation or aggregation
- Example: *“What is the average salary of the Sales department?”*

### RAG Query
- Requires document understanding
- Example: *“What is the executive summary of the Marketing report 2024?”*

### HYBRID Query
- Requires both structured and unstructured data
- Example: *“What is the average salary of the Sales department and the executive summary of the Marketing report 2024?”*

---

## 🔀 Hybrid Query Execution

For hybrid queries, the system:

1. Decomposes the query into independent sub-queries
2. Executes each sub-query in parallel:
   - SQL path → Pandas computation on CSV data
   - RAG path → Vector search + document summarization
3. Combines outputs into a single, coherent response

This allows the system to answer complex business questions accurately.

---

## 🔍 Example Query Walkthrough

**User Query:**

> *What is the average salary of the Sales department, and what is the executive summary of the Marketing report 2024?*

### Internal Processing:
- Query classified as **HYBRID**
- Query split into two parts:
  - SQL path → Compute average salary from HR CSV data
  - RAG path → Retrieve and summarize Marketing report
- Both paths executed in parallel
- Outputs merged into a unified answer

---

## 🧩 Answer Synthesis

After retrieval and computation:
- Structured results and document insights are combined
- The LLM generates a grounded response
- Context from previous conversation is preserved
- The final answer feels natural and coherent to the user

---

## 🛠️ Technology Stack

- **FastAPI** – API layer, request handling, authentication
- **LangGraph** – State management, conditional routing, hybrid execution
- **Large Language Models (LLMs)** – Query understanding and synthesis
- **Vector Store** – Document indexing and retrieval
- **Pandas** – Structured data analysis
- **JWT** – Secure authentication and authorization

---

## 📈 Reliability & Safety

- Safe execution of generated Pandas code
- Strict department-level data filtering
- No direct raw SQL execution
- Controlled access to document retrieval
- Confidence-aware answer generation

---

## 🚧 Future Enhancements

- Frontend UI for better user interaction
- Streaming responses for real-time feedback
- Dockerization and cloud deployment
- Improved confidence scoring and observability
- Advanced analytics and monitoring

---

## 🎯 Why This Project Matters

This project demonstrates:
- Real-world LLM orchestration
- Secure enterprise data access
- Intelligent hybrid reasoning
- Production-style stateful workflows

It moves beyond simple RAG demos toward **practical, deployable AI systems**.

---

## 🔗 Repository

GitHub:  
https://github.com/pardhu-koneru/Role-Based-RAG-Chat-Application

---

## 🙌 Final Note

This system is intentionally designed to reflect **how real organizations work**, rather than simplifying assumptions often made in tutorials.

If you're interested in **RAG, LangGraph, or enterprise AI system design**, this project is a practical reference implementation.
