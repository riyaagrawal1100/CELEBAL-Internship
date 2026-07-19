# 🤖 Single Agent System & Agent Pipeline

## Overview

This project demonstrates a basic **Single Agent System** capable of routing user queries to different tools based on their intent.

The agent performs conditional routing and selects the appropriate tool to execute the requested task.

---

## Features

- Calculator Tool
- Keyword Extraction Tool
- General Response Handler
- Rule-Based Conditional Routing
- Basic Error Handling

---

## Project Structure

```
Week8_Single_Agent_Pipeline.ipynb
README.md
requirements.txt
```

---

## Workflow

```
              User Query
                   │
                   ▼
          Single Agent Router
                   │
      ┌────────────┼────────────┐
      │            │            │
      ▼            ▼            ▼
 Calculator   Keyword Tool   General Response
      │            │            │
      └────────────┼────────────┘
                   ▼
             Final Response
```

---

## Tools Used

### Calculator Tool
Performs arithmetic calculations.

Example:
```
Calculate 20 + 5
```

Output:
```
25
```

---

### Keyword Extraction Tool

Extracts important words from a sentence.

Example:

```
Extract keywords from Artificial Intelligence is transforming industries
```

Output:

```
['artificial', 'intelligence', 'transforming', 'industries']
```

---

### General Response

Handles queries that do not match any predefined tool.

Example:

```
What is Machine Learning?
```

---

## Technologies Used

- Python 3
- Google Colab / Jupyter Notebook
- Regular Expressions (re)

---

## Concepts Covered

- Single Agent Systems
- Agent Pipelines
- Conditional Routing
- Tool Calling
- Error Handling
- Modular Programming

---

## How to Run

1. Open the notebook in Google Colab or Jupyter Notebook.
2. Run all cells.
3. Execute the test queries.
4. Observe how the agent routes requests to different tools.

---

## Example Queries

```
Calculate 15 * 8
```

```
Extract keywords from Machine Learning is changing healthcare
```

```
What is Artificial Intelligence?
```

---

## Author

Riya Agrawal
