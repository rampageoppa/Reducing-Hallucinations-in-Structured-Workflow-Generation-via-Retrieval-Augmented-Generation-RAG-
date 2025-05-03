BASE_TEMPLATE = """You are an assistant that converts a plain‑English requirement
into a JSON workflow following this schema:
{{"trigger": {{"type": "on_insert", "table": "<table>"}},
  "steps": [{{"id": "<string>", "name": "<step_name>", "parent": "<id_or_null>"}}]}}
Return valid JSON ONLY.

Requirement:
{nl}
JSON:
"""

RAG_TEMPLATE = """### SuggestedSteps
{steps}
### SuggestedTables
{tables}
### UserRequirement
{nl}
### GenerateWorkflowJSON
"""
