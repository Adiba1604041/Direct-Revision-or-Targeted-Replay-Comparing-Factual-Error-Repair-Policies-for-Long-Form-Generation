import json
import os
import time
import traceback
import pandas as pd
import requests
from jsonschema import validate
import autogen
import re
import hashlib
# =========================================================
# Configuration
# =========================================================
MODEL_NAME = "gemma3:64k"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
TIMEOUT = 900

# =========================================================
# Source article helpers for HDM
# =========================================================

SOURCE_ARTICLE_COLUMNS = [
    "full_source_text_1",
    "full_source_text_2",
    "full_source_text_3"
]


def get_source_articles(row):
    articles = []

    for col in SOURCE_ARTICLE_COLUMNS:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            articles.append(str(row[col]))

    return articles


def build_source_articles_text(row):
    articles = get_source_articles(row)

    if not articles:
        return "No source articles provided."

    return "\n\n".join(
        [f"Source Article {i+1}:\n{article}" for i, article in enumerate(articles)]
    )
    
# =========================================================
# Shared Memory
# =========================================================
import json

class SharedMemory:
    def __init__(self, main_task):
        self.memory = {
            "main_task": main_task,
            "planner_output": None,
            "agent_architect_output": None,
            "subtasks": {},
            "final_output": None,
            "judge_output": None
        }

    def set_planner_output(self, planner_output):
        self.memory["planner_output"] = planner_output

    def add_subtasks_from_plan(self, subtasks):
        for subtask in subtasks:
            subtask_id = subtask["subtask_id"]
            self.memory["subtasks"][subtask_id] = {
                "subtask_description": subtask["subtask_description"],
                "assigned_agent": None,
                "status": "pending",
                "executor_output": None,
                "confidence": None
            }

    def set_agent_architect_output(self, architect_output):
        self.memory["agent_architect_output"] = architect_output

    def assign_agents(self, agent_allocation_map):
        for agent_id, subtask_ids in agent_allocation_map.items():
            for subtask_id in subtask_ids:
                if subtask_id in self.memory["subtasks"]:
                    self.memory["subtasks"][subtask_id]["assigned_agent"] = agent_id

    def update_executor_output(self, subtask_id, task_result, confidence=None):
        if subtask_id in self.memory["subtasks"]:
            self.memory["subtasks"][subtask_id]["executor_output"] = task_result
            self.memory["subtasks"][subtask_id]["confidence"] = confidence
            self.memory["subtasks"][subtask_id]["status"] = "completed"

    def get_previous_subtask_outputs(self, current_subtask_id):
        context = {}
        for subtask_id, info in self.memory["subtasks"].items():
            if subtask_id == current_subtask_id:
                break
            if info["executor_output"] is not None:
                context[subtask_id] = info["executor_output"]
        return context

    def set_final_output(self, final_output):
        self.memory["final_output"] = final_output

    def set_judge_output(self, judge_output):
        self.memory["judge_output"] = judge_output

    def to_json(self):
        return json.dumps(self.memory, indent=2, ensure_ascii=False)
# =========================================================
# Schemas (Keep your existing schemas, they are well-defined)
# =========================================================
milestone_item_schema = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "milestone_id": {"type": "string"},
        "priority": {
            "type": "string",
            "enum": ["MANDATORY", "OPTIONAL"]
        },
        "description": {"type": "string"}
    },
    "required": ["milestone_id", "priority", "description"]
}

planner_schema = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "Subtasks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "subtask_id": {"type": "string"},
                    "subtask_description": {"type": "string"}
                },
                "required": ["subtask_id", "subtask_description"]
            },
            "minItems": 1
        },
        "FinalExpectedMilestones": {
            "type": "array",
            "items": milestone_item_schema,
            "minItems": 1
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
        }

    },
    "required": ["Subtasks", "FinalExpectedMilestones", "confidence"]
}

# (Other schemas: architect_schema, executor_schema, synthesizer_schema, judge_schema 
# remain the same as your original code)
architect_schema = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "agent_allocation_map": {
            "type": "object",
            "patternProperties": {
                "^A\\d+$": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "pattern": "^T\\d+$"
                    },
                    "minItems": 1
                }
            },
            "additionalProperties": False,
            "minProperties": 1
        },
        "agent_specs": {
            "type": "object",
            "patternProperties": {
                "^A\\d+$": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "actor_type": {"type": "string"},
                        "skills": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1
                        },
                        "rationale": {"type": "string"}
                    },
                    "required": ["actor_type", "skills", "rationale"]
                }
            },
            "additionalProperties": False,
            "minProperties": 1
        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        }
    },
    "required": ["agent_allocation_map", "agent_specs", "confidence"]
}
executor_schema = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "subtask_id": {"type": "string"},
        "task_result": {"type": "string"},
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
        }

    },
    "required": ["subtask_id", "task_result", "confidence"]
}

synthesizer_schema = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "task_result": {"type": "string"},
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1
            }
    },
    "required": ["task_result", "confidence"]
}

judge_schema = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "event_title": {"type": "string"},
        "milestone_judgment": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "milestone_scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "milestone_id": {"type": "string"},
                            "priority": {
                                "type": "string",
                                "enum": ["MANDATORY", "OPTIONAL"]
                            },
                            "description": {"type": "string"},
                            "achieved": {"type": "boolean"},
                            "score_1_to_5": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            },
                            "reason": {"type": "string"}
                        },
                        "required": [
                            "milestone_id",
                            "priority",
                            "description",
                            "achieved",
                            "score_1_to_5",
                            "reason"
                        ]
                    }
                }
            },
            "required": ["milestone_scores"]
        },
        "dimension_scores": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "core_event_fidelity": {"type": "number"},
            "required_entity_coverage": {"type": "number"},
            "key_information_coverage": {"type": "number"},
            "cross_source_synthesis": {"type": "number"},
            "structural_coverage": {"type": "number"},
            "statistics_and_claims_accuracy": {"type": "number"},
            "narrative_quality": {"type": "number"}
        },
        "required": [
            "core_event_fidelity",
            "required_entity_coverage",
            "key_information_coverage",
            "cross_source_synthesis",
            "structural_coverage",
            "statistics_and_claims_accuracy",
            "narrative_quality"
        ]
    },
        "quality_analysis": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "coherence": {"type": "number"},
            "readability": {"type": "number"},
            "conciseness": {"type": "number"}
        },
        "required": [
            "coherence",
            "readability",
            "conciseness"
        ]
    },
        "summary": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "strengths": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "weaknesses": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["strengths", "weaknesses"]
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
        }
    },
    "required": [
        "event_title",
        "milestone_judgment",
        "dimension_scores",
        "quality_analysis",
        "summary",
        "confidence"
    ]
}


# =========================================================
# HDM Schemas
# =========================================================

claim_extraction_schema = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "claim_id": {"type": "string"},
                    "claim": {"type": "string"},
                    "source_sentence": {"type": "string"},
                    "claim_type": {
                        "type": "string",
                        "enum": [
                            "entity",
                            "date",
                            "number",
                            "event",
                            "causal",
                            "quote",
                            "background",
                            "other"
                        ]
                    }
                },
                "required": [
                    "claim_id",
                    "claim",
                    "source_sentence",
                    "claim_type"
                ]
            }
        }
    },
    "required": ["claims"]
}


# One agent will do BOTH verification and provenance tracing.
claim_verification_provenance_schema = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "claim_verifications": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "claim_id": {"type": "string"},
                    "claim": {"type": "string"},
                    "label": {
                        "type": "string",
                        "enum": ["supported", "unsupported", "contradicted"]
                    },
                    "evidence": {"type": "string"},
                    "reason": {"type": "string"},

                    "accountable_stage": {
                        "type": "string",
                        "enum": [
                            "executor",
                            "synthesizer",
                            "unknown",
                            "not_applicable"
                        ]
                    },
                    "accountable_agent_id": {"type": "string"},
                    "accountable_subtask_id": {"type": "string"},
                    "agent_contribution": {
                        "type": "string",
                        "enum": [
                            "introduced",
                            "propagated",
                            "distorted",
                            "unknown",
                            "not_applicable"
                        ]
                    },
                    "trace_evidence": {"type": "string"},
                    "provenance_confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1
                    }
                },
                "required": [
                    "claim_id",
                    "claim",
                    "label",
                    "evidence",
                    "reason",
                    "accountable_stage",
                    "accountable_agent_id",
                    "accountable_subtask_id",
                    "agent_contribution",
                    "trace_evidence",
                    "provenance_confidence"
                ]
            }
        }
    },
    "required": ["claim_verifications"]
}
# =========================================================
# System messages (CORRECTED to match schemas)
# =========================================================
planner_system_msg = """
You are a Task Planner for a multi-agent public-event narrative synthesis
system for digital collections.

Your job is to take the main task and produce:
- A list of actionable subtasks
- Final expected milestones for the overall public-event narrative
- A confidence score for the plan from 0 to 1

Instructions:
1. Analyze the main task exactly as given.
2. Break it down into actionable, clear, and logically ordered subtasks.
3. Do not skip important subtasks or create unnecessary subtasks.
4. Do NOT define milestones for individual subtasks.
5. For each subtask, provide:
   - subtask_id
   - subtask_description
6. Define final expected milestones for the complete public-event narrative.
7. Milestones must describe observable conditions for successful completion.
8. Each milestone must include a milestone_id, priority, and description.
9. MANDATORY milestones must represent vital requirements for factual
   correctness and adequate completion.
10. OPTIONAL milestones should improve completeness, context, accessibility,
    nuance, or synthesis quality.
11. There must be at least one MANDATORY milestone.
12. Provide a confidence score from 0 to 1.

13. Provide the output in the following structured JSON format ONLY:

### EXAMPLE STRUCTURE
{
  "Subtasks": [
    {
      "subtask_id": "T1",
      "subtask_description": "Extract the central event details from the supplied metadata and source documents."
    },
    {
      "subtask_id": "T2",
      "subtask_description": "Identify key entities, claims, and any source-level uncertainty or disagreement."
    },
    {
      "subtask_id": "T3",
      "subtask_description": "Organize the verified information into a public-facing event narrative structure."
    }
  ],
  "FinalExpectedMilestones": [
    {
      "milestone_id": "M1",
      "priority": "MANDATORY",
      "description": "The narrative must accurately represent the central event using only the supplied metadata and source documents."
    },
    {
      "milestone_id": "M2",
      "priority": "MANDATORY",
      "description": "The narrative must cover the key entities and factual claims supported by the source documents."
    },
    {
      "milestone_id": "M3",
      "priority": "OPTIONAL",
      "description": "The narrative should explain uncertainty or disagreement across source documents when present."
    }
  ],
  "confidence": 0.9
}
"""

# Rest of system messages updated for strict JSON enforcement...
architect_system_msg = """
You are an Agent Architect in an autonomous multi-agent system.

You will receive a main task and a list of subtasks with their associated ID.  
Your job is to design agents and assign the subtasks to the agents efficiently.

You must produce the following things:
1. Agent Allocation Map : which agent handles which subtasks
2. Agent Specifications : details about each agent
3. Confidence score (0 to 1) for your overall agent design and allocation plan

Guidelines:
- Analyze all subtasks together.
- Group similar subtasks under the same agent when possible.
- Avoid creating unnecessary agents.
- Each subtask must be assigned to exactly one agent.
- Each agent must have a clear role and skills.

For each agent must provide:
- "actor_type" short role name (e.g. data_collector, data_analyzer, writer_agent etc.) (string)
- "skills" (array of strings)
- "rationale" (string)

### CRITICAL STRUCTURAL RULES:
1. **agent_allocation_map**: Must be an OBJECT where keys are Agent IDs (A1, A2...) and values are ARRAYS of Subtask IDs (T1, T2...).
2. **agent_specs**: Must be an OBJECT (not a list). The keys MUST be the same Agent IDs used in the map (A1, A2...).
3. Each agent specification must include "actor_type", "skills" and "rationale".
4. The output JSON must strictly follow the defined structure without any extra text or markdown.   

### EXAMPLE FORMAT (FOLLOW THIS JSON format strictly):
### EXAMPLE FORMAT
{
  "agent_allocation_map": {
    "A1": ["T1", "T2"],
    "A2": ["T3"],
    "A3": ["T4"]
  },
  "agent_specs": {
    "A1": {
      "actor_type": "Source Document Analyst",
      "skills": ["source-document analysis", "fact extraction", "entity identification"],
      "rationale": "This agent extracts central event details, key entities, and supported factual claims from the supplied collection materials."
    },
    "A2": {
      "actor_type": "Evidence Comparison Specialist",
      "skills": ["cross-source comparison", "uncertainty detection", "claim verification"],
      "rationale": "This agent compares the supplied source documents to identify overlapping evidence, disagreement, and missing information."
    },
    "A3": {
      "actor_type": "Narrative Organizer",
      "skills": ["public-facing synthesis", "narrative organization", "factual grounding"],
      "rationale": "This agent organizes verified information into a coherent structure for the final public-event narrative."
    }
  },
  "confidence": 0.9
}
"""
def build_executor_system_msg(actor_spec):
    return f"""
You are an Executor Agent in an autonomous multi-agent public-event narrative
synthesis system.

- Your assigned role is: {actor_spec['actor_type']}
- Your expertise and skills are: {actor_spec['skills']}

Your job is to execute the assigned subtask accurately using the supplied
event-centered digital collection.

Execution rules:
1. Stay strictly within your assigned subtask, role, and expertise.
2. Treat the supplied event metadata and source documents as the only evidence
   for factual claims.
3. Do not use outside knowledge, web search, or unsupported assumptions.
4. Use prior subtask outputs only as supporting context. Do not treat them as
   more authoritative than the source documents.
5. Identify uncertainty, disagreement, or missing information when relevant.
6. Do not invent facts, statistics, quotations, dates, entities, motivations,
   or outcomes.
7. Place the completed subtask result in the "task_result" field.
8. Provide a confidence score from 0 to 1.

Output rules:
- Confidence must be between 0 and 1.
- Output must be valid JSON matching the provided schema.
- Do not include explanation outside the JSON object.

Your output must be ONLY valid JSON.

### EXAMPLE:
{{
  "subtask_id": "T1",
  "task_result": "The FDA approved the Cepheid test on March 21, 2020.....",
  "confidence": 0.9
}}
"""


synthesizer_system_msg = """
You are the Final Synthesizer in a multi-agent public-event narrative synthesis
system for digital collections.

Your task is to synthesize the executor outputs into a readable, public-facing
event narrative of 400-600 words.

Grounding rules:
- Treat the supplied event metadata and source documents as the authoritative
  evidence.
- Use the executor outputs as analysis materials, but do not repeat claims that
  are unsupported by the source documents.
- Do not use outside knowledge or invent facts, statistics, quotations, dates,
  entities, motivations, or outcomes.
- If the source documents disagree or contain uncertainty, describe that
  cautiously.
- Synthesize relevant information across the available source documents rather
  than summarizing only one source.

Narrative rules:
- The narrative should help a reader understand the event-centered collection
  without reading every source document.
- The output should be coherent, readable, factual, accessible, and neutral.
- The output is not an opinion piece, newspaper column, or journalistic report.
- Place the final public-event narrative in the "task_result" field.
- Provide a confidence score from 0 to 1.

Return ONLY valid JSON matching the required schema.
Do not include markdown or extra text outside the JSON object.

### EXAMPLE:
{
  "task_result": "In a significant move for pandemic response, the FDA announced on March 21...",
  "confidence": 0.9
}
"""

judge_system_msg = """
You are an expert evaluator of public-event narratives generated from
event-centered digital collections.

You will be given:
1. Structured event metadata
2. Related source documents
3. A generated public-event narrative
4. A set of final expected milestones
5. A constraint-based ground truth

Your task is to judge whether the generated narrative is a faithful,
well-synthesized, and accessible public-facing knowledge artifact.

You must evaluate:
1. Milestone achievement
2. Core event fidelity
3. Required entity coverage
4. Key information coverage
5. Cross-source synthesis
6. Structural coverage
7. Statistics and claims accuracy
8. Narrative quality
9. Quality subdimensions:
   - coherence
   - readability
   - conciseness

Evaluation guidance:
- Do not judge based on wording overlap.
- Judge based on factual grounding, source-document fidelity, and constraint
  satisfaction.
- Treat the supplied metadata, source documents, and constraint-based ground
  truth as the authoritative evidence.
- Reward narratives that integrate relevant information across multiple source
  documents.
- A narrative that relies heavily on only one source document should receive a
  lower cross-source synthesis score when other relevant sources are available.
- Missing optional details should not be penalized heavily.
- Missing mandatory milestones or major factual content should reduce the
  score.
- Unsupported claims, invented numbers, invented entities, fabricated
  quotations, or overconfident treatment of disputed claims should reduce the
  score significantly.
- Be strict about factual grounding, but allow paraphrasing.
- Evaluate whether the narrative is understandable to a reader who has not
  read the source documents.
- Do not evaluate the output as journalism, an opinion column, or a newspaper
  article.

Milestone scoring rules:
- Judge each milestone independently.
- Score each milestone on a scale of 1 to 5.
- 5 = fully achieved
- 4 = mostly achieved
- 3 = partially achieved
- 2 = weakly achieved
- 1 = not achieved
- Mark achieved=true only if the milestone is adequately satisfied.
- Mark achieved=false if the milestone is not adequately satisfied.
- Do not compute aggregate milestone counts.
- Only provide milestone-level judgments.

Dimension score rules:
- Score each dimension from 0.0 to 1.0.
- 1.0 = fully satisfied.
- 0.5 = partially satisfied.
- 0.0 = not satisfied.
- Use intermediate values when appropriate.

Provide a confidence score from 0 to 1 for the evaluation.

Return ONLY valid JSON matching the required schema.
Do not include markdown or extra text outside the JSON object.

## EXAMPLE FORMAT
{
  "event_title": "Public Event Narrative Evaluation",
  "milestone_judgment": {
    "milestone_scores": [
      {
        "milestone_id": "M1",
        "priority": "MANDATORY",
        "description": "The narrative must accurately represent the central event using only the supplied metadata and source documents.",
        "achieved": true,
        "score_1_to_5": 5,
        "reason": "The narrative accurately represents the central event and does not introduce unsupported external information."
      },
      {
        "milestone_id": "M2",
        "priority": "MANDATORY",
        "description": "The narrative must cover the key entities and factual claims supported by the source documents.",
        "achieved": true,
        "score_1_to_5": 4,
        "reason": "The narrative covers the main entities and most supported factual claims, although a few source-supported details are underdeveloped."
      },
      {
        "milestone_id": "M3",
        "priority": "OPTIONAL",
        "description": "The narrative should explain uncertainty or disagreement across source documents when present.",
        "achieved": false,
        "score_1_to_5": 2,
        "reason": "The narrative is mostly faithful, but it does not clearly explain source-level uncertainty or disagreement where relevant."
      }
    ]
  },
  "dimension_scores": {
    "core_event_fidelity": 1.0,
    "required_entity_coverage": 0.9,
    "key_information_coverage": 0.85,
    "cross_source_synthesis": 0.8,
    "structural_coverage": 0.9,
    "statistics_and_claims_accuracy": 1.0,
    "narrative_quality": 0.85
  },
  "quality_analysis": {
    "coherence": 0.9,
    "readability": 0.9,
    "conciseness": 0.8
  },
  "summary": {
    "strengths": [
      "The narrative accurately represents the central event and key entities.",
      "The narrative avoids unsupported claims and remains grounded in the supplied source documents."
    ],
    "weaknesses": [
      "The narrative could explain source disagreement or uncertainty more clearly."
    ]
  },
  "confidence": 0.9
}
"""
# =========================================================
# HDM System Messages
# =========================================================

claim_extraction_system_msg = """
You are hdm_claim_extractor_agent, a claim extraction agent for public-event narratives.

Your task is to extract atomic factual claims from the generated public-event narrative.

Rules:
- Extract only factual claims.
- Split compound sentences into separate atomic claims.
- Do not extract opinions, vague commentary, or purely stylistic statements.
- Each claim must be independently checkable.
- Return ONLY valid JSON matching the schema.
Do not include markdown or extra text outside the JSON object.
"""

hdm_verification_provenance_system_msg = """
You are hdm_verification_provenance_agent.

Your task has two parts:

Part 1: Source-grounded claim verification
- Verify each extracted claim against ONLY the constraint-based ground truth and related source documents.
- Do not use internal knowledge, web search, or unsupported assumptions.
- Label each claim as:
  - supported: the claim is supported by the provided evidence.
  - unsupported: the claim is not supported by the provided evidence.
  - contradicted: the claim conflicts with the provided evidence.
- Provide a clear reason for each verification label.
- For unsupported or contradicted claims, do not provide the corrected claim unless the correction is explicitly supported by the provided evidence.
- Provide evidence from the ground truth or source documents that supports your verification label.

Part 2: Multi-agent provenance tracing
- For every unsupported or contradicted claim, trace where the claim likely originated in the multi-agent workflow.
- You will receive executor outputs, assigned agent IDs, subtask IDs, and the final synthesized public-event narrative.
- If the wrong claim appears in an executor output, identify that executor agent ID and corresponding subtask ID.
- If the wrong claim does not appear in executor outputs but appears in the final narrative, assign accountability to the synthesizer.
- If the accountable stage is synthesizer, keep accountable_agent_id and accountable_subtask_id as "not_applicable".
- If the origin cannot be determined, use unknown.
- Use semantic matching, not only exact string matching.

For supported claims:
- Use accountable_stage = "not_applicable"
- Use accountable_agent_id = "not_applicable"
- Use accountable_subtask_id = "not_applicable"
- Use agent_contribution = "not_applicable"
- Use trace_evidence = "Claim is supported; provenance tracing not required."

Rules:
- Be strict about numbers, dates, entities, causal claims, quotations, and source attribution.
- Do not guess beyond the provided source documents and execution trace.
- Do not evaluate the text as journalism or an opinion column.
- Return ONLY valid JSON matching the schema.
Do not include markdown or extra text outside the JSON object.
"""
# =========================================================
# Prompt builders
# =========================================================
def build_initial_task(row):
    source_documents = build_source_articles_text(row)

    return f"""
Generate a public-event narrative using a multi-agent workflow for the
following event-centered digital collection.

Event Metadata:
- Year: {row['year']}
- Month: {row['month']}
- Date: {row['date']}
- Event: {row['event']}

Related Source Documents:
{source_documents}

Overall objective:
Produce a readable, factual, neutral, and source-grounded public-event
narrative in 400-600 words.
"""

def build_planner_user_msg(row):
    return f"""
Main Task:
{build_initial_task(row)}

Produce:
- Subtasks with IDs T1, T2, and so on
- Final expected milestones with IDs M1, M2, and so on
- Confidence about the plan
"""

def build_architect_user_msg(row, planner_output):
    return f"""
Main Task:
{build_initial_task(row)}

Planner Output:
{json.dumps(planner_output, indent=2, ensure_ascii=False)}

Create:
1. agent_allocation_map
2. agent_specs
3. confidence about the agent design and allocation plan
"""

def build_executor_user_msg(row, subtask, context_dict):
    source_documents = build_source_articles_text(row)

    return f"""
Execute one subtask for the following event-centered digital collection.

Event Metadata:
- Year: {row['year']}
- Month: {row['month']}
- Date: {row['date']}
- Event: {row['event']}

Related Source Documents:
{source_documents}

Assigned Subtask:
{subtask['subtask_description']}

Subtask ID:
{subtask['subtask_id']}

Context from Previously Completed Subtasks:
{json.dumps(context_dict, indent=2, ensure_ascii=False)}

Complete only the assigned subtask.
"""


def build_synthesizer_user_msg(row, final_expected_milestones, execution_outputs):
    source_documents = build_source_articles_text(row)

    simplified_outputs = {
        subtask_id: output["executor_output"]["task_result"]
        for subtask_id, output in execution_outputs.items()
    }

    return f"""
Generate the final public-event narrative using the materials below.

Event Metadata:
- Year: {row['year']}
- Month: {row['month']}
- Date: {row['date']}
- Event: {row['event']}

Related Source Documents:
{source_documents}

Final Expected Milestones:
{json.dumps(final_expected_milestones, indent=2, ensure_ascii=False)}

Executor Outputs:
{json.dumps(simplified_outputs, indent=2, ensure_ascii=False)}

Return ONLY valid JSON with:
- task_result
- confidence
"""

def build_judge_user_msg(row, final_expected_milestones, task_result):
    source_documents = build_source_articles_text(row)

    return f"""
Evaluate the following public-event narrative generated from an event-centered
digital collection.

Event Metadata:
- Year: {row['year']}
- Month: {row['month']}
- Date: {row['date']}
- Event: {row['event']}

Related Source Documents:
{source_documents}

Final Expected Milestones:
{json.dumps(final_expected_milestones, indent=2, ensure_ascii=False)}

Generated Public-Event Narrative:
{task_result}

Constraint-Based Ground Truth:
{row['ground_truth_json']}

Return ONLY valid JSON with:
- event_title
- milestone_judgment
- dimension_scores
- quality_analysis
- summary
- confidence
"""

# =========================================================
# HDM Prompt Builders
# =========================================================

def build_claim_extraction_user_msg(task_result):
    return f"""
Generated Public-Event Narrative:
{task_result}

Extract atomic factual claims from this narrative.

Return ONLY valid JSON with:
- claims
"""


def build_execution_trace_text(shared_memory, final_article):
    blocks = []

    for subtask_id, info in shared_memory.memory["subtasks"].items():
        blocks.append(
            f"""
                [SUBTASK {subtask_id}]
                Assigned Agent: {info.get("assigned_agent", "unknown")}
                Subtask Description: {info.get("subtask_description", "")}
                Status: {info.get("status", "")}
                Executor Confidence: {info.get("confidence", "")}

                Executor Output:
                {info.get("executor_output", "")}
"""
        )

    blocks.append(
        f"""
[SYNTHESIZER FINAL OUTPUT]
{final_article}
"""
    )

    return "\n\n".join(blocks)


def build_verification_provenance_user_msg(row, extracted_claims, shared_memory, final_article):
    return f"""
Extracted Claims:
{json.dumps(extracted_claims, indent=2, ensure_ascii=False)}

Constraint-Based Ground Truth:
{row['ground_truth_json']}

Related Source Documents:
{build_source_articles_text(row)}

Multi-Agent Execution Trace:
{build_execution_trace_text(shared_memory, final_article)}

Verify every claim and, for unsupported or contradicted claims, trace the accountable agent or stage.

Return ONLY valid JSON with:
- claim_verifications
"""

# =========================================================
# AutoGen call with schema
# =========================================================

_AUTOGEN_AGENT_CACHE = {}


def extract_json_from_text(text):
    """
    AutoGen/local models may return markdown fences or extra text.
    This extracts the JSON object safely.
    """
    text = text.strip()

    # Remove Qwen-style thinking text if it appears
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Remove markdown fences if they appear
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Could not parse JSON from AutoGen response:\n{text}")


def temperature_label(value):
    """
    Converts temperatures such as 0.5 to safe labels such as 0p5.
    This avoids decimal points in AutoGen agent names and file names.
    """
    return str(value).replace(".", "p")


def build_llm_config(model=MODEL_NAME, timeout=TIMEOUT, temperature=0.0):
    """
    AutoGen config for Ollama's OpenAI-compatible endpoint.
    Make sure Ollama is running.
    """
    return {
        "config_list": [
            {
                "model": model,
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama",
                "timeout": timeout,
                "response_format": {"type": "json_object"}
            }
        ],
        "temperature": temperature,
        "cache_seed": None
    }


def get_agent_name_from_system_msg(system_msg, agent_id=None):
    """
    Assigns a stable AutoGen agent name.
    For executors, use the architect-assigned agent_id when available.
    """

    if system_msg == planner_system_msg:
        return "planner_agent"

    if system_msg == architect_system_msg:
        return "architect_agent"

    if system_msg == synthesizer_system_msg:
        return "synthesizer_agent"

    if system_msg == judge_system_msg:
        return "judge_agent"
    
    if system_msg == claim_extraction_system_msg:
        return "hdm_claim_extractor_agent"

    if system_msg == hdm_verification_provenance_system_msg:
        return "hdm_verification_provenance_agent"

    if system_msg.strip().startswith("You are an Executor Agent"):
        if agent_id is not None:
            return f"executor_agent_{agent_id}"
        digest = hashlib.md5(system_msg.encode("utf-8")).hexdigest()[:8]
        return f"executor_agent_{digest}"

    digest = hashlib.md5(system_msg.encode("utf-8")).hexdigest()[:8]
    return f"generic_agent_{digest}"

def get_autogen_agent(
    system_msg,
    model=MODEL_NAME,
    timeout=TIMEOUT,
    agent_id=None,
    temperature=0.0
):
    """
    Creates or retrieves a cached AutoGen AssistantAgent and UserProxyAgent.
    Each architect-assigned executor agent can have its own AutoGen object.
    """
    agent_name = get_agent_name_from_system_msg(system_msg, agent_id=agent_id)
    temp_label = temperature_label(temperature)
    cache_key = f"{model}_{timeout}_{agent_name}_T{temp_label}"

    if cache_key in _AUTOGEN_AGENT_CACHE:
        return _AUTOGEN_AGENT_CACHE[cache_key]

    llm_config = build_llm_config(
    model=model,
    timeout=timeout,
    temperature=temperature
)

    assistant = autogen.AssistantAgent(
        name=f"{agent_name}_T{temp_label}",
        system_message=system_msg,
        llm_config=llm_config
    )

    proxy = autogen.UserProxyAgent(
        name=f"{agent_name}_proxy",
        human_input_mode="NEVER",
        code_execution_config=False
    )

    _AUTOGEN_AGENT_CACHE[cache_key] = (assistant, proxy)

    return assistant, proxy


def get_autogen_response(agent, proxy, chat_result=None):
    """
    Gets the latest assistant response from an AutoGen chat.
    """
    try:
        msg = agent.last_message(proxy)
        if msg and "content" in msg:
            return msg["content"]
    except Exception:
        pass

    if chat_result is not None and hasattr(chat_result, "chat_history"):
        for msg in reversed(chat_result.chat_history):
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"]

    raise ValueError("Could not retrieve AutoGen assistant response.")


def call_ollama_with_schema(
    system_msg,
    user_msg,
    schema,
    model=MODEL_NAME,
    timeout=TIMEOUT,
    max_retries=3,
    agent_id=None,
    temperature=0.0
):
    """
    Same function name as before, so the rest of your code does not need to change.

    This version uses AutoGen agents instead of direct requests.post().
    Each module gets its own AutoGen AssistantAgent object:
    - planner_agent
    - architect_agent
    - executor_agent_*
    - synthesizer_agent
    - judge_agent
    """

    agent, proxy = get_autogen_agent(
    system_msg=system_msg,
    model=model,
    timeout=timeout,
    agent_id=agent_id,
    temperature=temperature
)

    agent_name = agent.name

    base_user_msg = (
        user_msg
        + "\n\nReturn JSON matching this schema exactly:\n"
        + json.dumps(schema, indent=2, ensure_ascii=False)
        + "\n\nReturn ONLY the JSON object. Do not include markdown or explanation."
    )

    last_error = None
    raw_content = ""

    for attempt in range(max_retries):
        if attempt == 0:
            message = base_user_msg
        else:
            message = f"""
Your previous response did not match the required JSON schema.

Previous invalid response:
{raw_content if raw_content else "No content"}

Error:
{str(last_error)}

Please generate again and follow the schema exactly.
Return ONLY valid JSON.

Original task:
{base_user_msg}
"""

        try:
            chat_result = proxy.initiate_chat(
                agent,
                message=message,
                max_turns=1,
                silent=True,
                clear_history=True
            )

            raw_content = get_autogen_response(agent, proxy, chat_result)
            parsed = extract_json_from_text(raw_content)
            validate(instance=parsed, schema=schema)
            return parsed

        except Exception as e:
            last_error = e

            print(f"--- ATTEMPT {attempt + 1} FAILED FOR {agent_name} ---")
            print(f"Raw Output: {raw_content if raw_content else 'No content'}")
            print(f"Error: {str(e)}")

    raise RuntimeError(
        f"AutoGen call failed for agent '{agent_name}' after {max_retries} attempts. "
        f"Last error: {type(last_error).__name__}: {last_error}"
    )
    

# =========================================================
# HDM for Multi-Agent System
# =========================================================

def run_hdm_multi_agent(
    row,
    task_result,
    shared_memory,
    model=MODEL_NAME,
    hdm_temperature=0.0
):
    """
    Multi-agent HDM:
    1. Extract claims from final synthesized article.
    2. Verify claims against ground truth + source articles.
    3. Trace unsupported/contradicted claims to executor/synthesizer.
    Verification and provenance are handled by the same HDM agent.
    """

    # Step 1: claim extraction
    claim_extraction_output = call_ollama_with_schema(
        system_msg=claim_extraction_system_msg,
        user_msg=build_claim_extraction_user_msg(task_result),
        schema=claim_extraction_schema,
        model=model,
        temperature=hdm_temperature
    )

    extracted_claims = claim_extraction_output["claims"]

    # Step 2: claim verification + provenance tracing using ONE agent
    verification_provenance_output = call_ollama_with_schema(
        system_msg=hdm_verification_provenance_system_msg,
        user_msg=build_verification_provenance_user_msg(
            row=row,
            extracted_claims=extracted_claims,
            shared_memory=shared_memory,
            final_article=task_result
        ),
        schema=claim_verification_provenance_schema,
        model=model,
        temperature=hdm_temperature
    )

    claim_verifications = verification_provenance_output["claim_verifications"]

    wrong_claims = [
        c for c in claim_verifications
        if c["label"] in ["unsupported", "contradicted"]
    ]

    total_claims = len(claim_verifications)
    wrong_claim_count = len(wrong_claims)
    hallucination_rate = safe_divide(wrong_claim_count, total_claims)

    accountable_agent_counts = {}
    accountable_subtask_counts = {}

    for claim in wrong_claims:
        agent_id = claim.get("accountable_agent_id", "unknown")
        subtask_id = claim.get("accountable_subtask_id", "unknown")

        accountable_agent_counts[agent_id] = accountable_agent_counts.get(agent_id, 0) + 1
        accountable_subtask_counts[subtask_id] = accountable_subtask_counts.get(subtask_id, 0) + 1

    return {
        "extracted_claims": extracted_claims,
        "claim_verifications": claim_verifications,
        "wrong_claims": wrong_claims,
        "total_claims": total_claims,
        "wrong_claim_count": wrong_claim_count,
        "hallucination_rate": hallucination_rate,
        "accountable_agent_counts": accountable_agent_counts,
        "accountable_subtask_counts": accountable_subtask_counts
    }

# =========================================================
# HDM-guided retriggering helpers
# =========================================================

def get_subtask_order_map(subtasks):
    """
    Returns a map like {"T1": 0, "T2": 1, ...}
    based on the original planner subtask order.
    """
    return {
        subtask["subtask_id"]: i
        for i, subtask in enumerate(subtasks)
    }
def is_valid_agent_subtask_pair(agent_id, subtask_id, agent_allocation_map):
    return (
        agent_id in agent_allocation_map
        and subtask_id in agent_allocation_map[agent_id]
    )


def filter_invalid_provenance_pairs(hdm_output, agent_allocation_map):
    all_wrong_claims = hdm_output.get("wrong_claims", [])

    valid_wrong_claims = []
    invalid_wrong_claims = []

    for claim in all_wrong_claims:
        # Keep synthesizer/unknown claims in repairable list.
        # They will be handled separately by your retrigger logic.
        if claim.get("accountable_stage") != "executor":
            valid_wrong_claims.append(claim)
            continue

        agent_id = claim.get("accountable_agent_id")
        subtask_id = claim.get("accountable_subtask_id")

        if is_valid_agent_subtask_pair(agent_id, subtask_id, agent_allocation_map):
            claim["provenance_pair_status"] = "valid"
            valid_wrong_claims.append(claim)
        else:
            claim["provenance_pair_status"] = "invalid"
            claim["excluded_from_retrigger"] = True
            invalid_wrong_claims.append(claim)

    # Put the lines here, before return
    hdm_output["all_wrong_claims"] = all_wrong_claims
    hdm_output["repairable_wrong_claims"] = valid_wrong_claims
    hdm_output["invalid_provenance_claims"] = invalid_wrong_claims
    hdm_output["invalid_provenance_claim_count"] = len(invalid_wrong_claims)

    return hdm_output

def is_valid_executor_target(claim):
    """
    Checks whether a wrong claim has a usable executor provenance target.
    """
    stage = claim.get("accountable_stage", "")
    agent_id = claim.get("accountable_agent_id", "")
    subtask_id = claim.get("accountable_subtask_id", "")

    invalid_values = ["", "none", "unknown", "not_applicable", None]

    return (
        stage == "executor"
        and agent_id not in invalid_values
        and subtask_id not in invalid_values
    )


def group_wrong_claims_by_target(hdm_output):
    """
    Groups wrong claims by accountable executor/subtask.

    Returns:
    - executor_claims_by_target: {(agent_id, subtask_id): [claims]}
    - synthesizer_claims: [claims]
    - unknown_claims: [claims]
    """
    executor_claims_by_target = {}
    synthesizer_claims = []
    unknown_claims = []

    for claim in hdm_output.get(
    "repairable_wrong_claims",
    hdm_output.get("wrong_claims", [])):
        stage = claim.get("accountable_stage", "unknown")

        if is_valid_executor_target(claim):
            key = (
                claim.get("accountable_agent_id"),
                claim.get("accountable_subtask_id")
            )
            executor_claims_by_target.setdefault(key, []).append(claim)

        elif stage == "synthesizer":
            synthesizer_claims.append(claim)

        else:
            unknown_claims.append(claim)

    return executor_claims_by_target, synthesizer_claims, unknown_claims


def format_claim_feedback(claims):
    """
    Builds a readable grouped feedback block from multiple wrong claims.
    """
    if not claims:
        return "No direct wrong claims were assigned to this agent/subtask."

    feedback_lines = []

    for i, claim in enumerate(claims, start=1):
        feedback_lines.append(f"{i}. Claim ID: {claim.get('claim_id', '')}")
        feedback_lines.append(f"   Claim: {claim.get('claim', '')}")
        feedback_lines.append(f"   Label: {claim.get('label', '')}")
        feedback_lines.append(f"   Reason: {claim.get('reason', '')}")
        feedback_lines.append("")

    return "\n".join(feedback_lines)


def build_executor_retrigger_user_msg(
    row,
    subtask,
    context_dict,
    previous_output,
    feedback_claims,
    is_accountable_agent
):
    """
    Builds the user message for retriggering an executor.

    If the executor is accountable, it gets claim-specific HDM feedback.
    If it is downstream, it gets neutral context-change feedback.
    """
    source_documents = build_source_articles_text(row)

    if is_accountable_agent:
        feedback_text = f"""
HDM found that your previous output contributed to the following unsupported or contradicted claim(s):

{format_claim_feedback(feedback_claims)}
"""
    else:
        feedback_text = """
A previous subtask output has been revised after HDM feedback.
You are being rerun only because your subtask may depend on updated previous context.
You are not directly blamed for a wrong claim.
Revise your output only if the updated context affects your previous answer.
Otherwise, you can keep your previous output as is.
"""

    return f"""
You are being retriggered for a repair round in a public-event narrative synthesis workflow.

Event Metadata:
- Year: {row['year']}
- Month: {row['month']}
- Date: {row['date']}
- Event: {row['event']}

Related Source Documents:
{source_documents}

Subtask ID:
{subtask['subtask_id']}

Assigned Subtask:
{subtask['subtask_description']}

Your Previous Output:
{previous_output}

Updated Context from Previously Completed Subtasks:
{json.dumps(context_dict, indent=2, ensure_ascii=False)}

HDM / Repair Feedback:
{feedback_text}

Repair Instructions:
- Do NOT restart from scratch.
- Modify only what is necessary based on the feedback, updated context, and source documents.
- Preserve all supported content from your previous output.
- Revise contradicted claims only when the correction is supported by the source documents.
- Remove unsupported claims unless they are clearly supported by the source documents.
- Do not introduce new unsupported facts, numbers, dates, quotations, entities, motivations, or outcomes.
- Keep the output useful for building a public-facing event narrative.
- Return ONLY valid JSON with:
  - subtask_id
  - task_result
  - confidence
"""


def build_synthesizer_retrigger_user_msg(
    row,
    final_expected_milestones,
    execution_outputs,
    previous_final_article,
    all_wrong_claims,
    repair_summary
):
    """
    Builds the user message for rerunning the synthesizer after executor repair.
    """
    source_documents = build_source_articles_text(row)

    simplified_outputs = {
        subtask_id: output["executor_output"]["task_result"]
        for subtask_id, output in execution_outputs.items()
    }

    return f"""
You are being retriggered for a repair round after HDM feedback.

Event Metadata:
- Year: {row['year']}
- Month: {row['month']}
- Date: {row['date']}
- Event: {row['event']}

Related Source Documents:
{source_documents}

Final Expected Milestones:
{json.dumps(final_expected_milestones, indent=2, ensure_ascii=False)}

Previous Final Public-Event Narrative:
{previous_final_article}

Updated Executor Outputs:
{json.dumps(simplified_outputs, indent=2, ensure_ascii=False)}

Repair Summary:
{repair_summary}

HDM Wrong Claims from Previous Final Narrative:
{json.dumps(all_wrong_claims, indent=2, ensure_ascii=False)}

Synthesizer Repair Instructions:
- Do NOT restart from scratch.
- Use the updated executor outputs and the related source documents.
- Preserve supported parts of the previous final narrative when possible.
- Remove or correct unsupported and contradicted claims.
- Do not use invalidated claims from old executor outputs.
- Do not introduce new unsupported facts, numbers, dates, quotations, entities, motivations, or outcomes.
- Keep the narrative readable, factual, neutral, and public-facing.
- Keep the narrative within 400-600 words.
- Return ONLY valid JSON with:
  - task_result
  - confidence
"""


def compute_hdm_change_status(hdm_before, hdm_after):
    before_rate = hdm_before.get("hallucination_rate", None)
    after_rate = hdm_after.get("hallucination_rate", None)

    before_wrong = hdm_before.get("wrong_claim_count", 0)
    after_wrong = hdm_after.get("wrong_claim_count", 0)

    if before_rate is None or after_rate is None:
        return {
            "hallucination_rate_delta": None,
            "hallucination_count_delta": None,
            "hallucination_reduction_rate": None,
            "hallucination_change_status": "unknown"
        }

    rate_delta = after_rate - before_rate
    count_delta = after_wrong - before_wrong

    if after_rate < before_rate:
        status = "reduced"
    elif after_rate > before_rate:
        status = "increased"
    else:
        status = "unchanged"

    if before_rate != 0:
        reduction_rate = (before_rate - after_rate) / before_rate
    else:
        reduction_rate = 0.0

    return {
        "hallucination_rate_delta": rate_delta,
        "hallucination_count_delta": count_delta,
        "hallucination_reduction_rate": reduction_rate,
        "hallucination_change_status": status
    }
    
def strip_evidence_from_claims(claims):
    cleaned_claims = []

    for claim in claims:
        cleaned_claim = {
            key: value
            for key, value in claim.items()
            if key not in ["evidence", "trace_evidence"]
        }
        cleaned_claims.append(cleaned_claim)

    return cleaned_claims
# =========================================================
# main retrigger function
# =========================================================    
def retrigger_after_hdm(
    row,
    subtasks,
    agent_allocation_map,
    agent_specifications,
    execution_outputs,
    final_expected_milestones,
    shared_memory,
    synthesizer_output,
    hdm_before,
    model=MODEL_NAME,
    repair_round=1,
    generation_temperature=0.0,
    hdm_temperature=0.0
):
    """
    Executes one HDM-guided repair round.

    Policy:
    1. Group wrong claims by accountable executor/subtask.
    2. Find earliest accountable executor subtask.
    3. Rerun from that earliest subtask onward.
    4. Accountable agents get claim-specific feedback.
    5. Downstream agents get neutral context-change feedback.
    6. Rerun synthesizer.
    7. Rerun HDM.
    """
    
    wrong_claims = hdm_before.get(
    "repairable_wrong_claims",
    hdm_before.get("wrong_claims", [])
)

    if not wrong_claims:
        return {
            "repair_performed": False,
            "reason": "HDM found no wrong claims.",
            "final_synthesizer_output": synthesizer_output,
            "final_task_result": synthesizer_output["task_result"],
            "hdm_after": hdm_before,
            "repair_report": {
                "repair_round": repair_round,
                "repair_performed": False,
                "reason": "No unsupported or contradicted claims found."
            }
        }
    wrong_claims_without_evidence = strip_evidence_from_claims(wrong_claims)

    (
        executor_claims_by_target,
        synthesizer_claims,
        unknown_claims
    ) = group_wrong_claims_by_target(hdm_before)

    subtask_order = get_subtask_order_map(subtasks)

    accountable_subtask_ids = [
        subtask_id
        for (_, subtask_id) in executor_claims_by_target.keys()
        if subtask_id in subtask_order
    ]

    retriggered_subtasks = []
    accountable_feedback_targets = []
    downstream_retriggered_subtasks = []

    # -----------------------------------------------------
    # Case 1: at least one executor is accountable
    # -----------------------------------------------------
    if accountable_subtask_ids:
        earliest_subtask_id = min(
            accountable_subtask_ids,
            key=lambda sid: subtask_order[sid]
        )
        earliest_index = subtask_order[earliest_subtask_id]

        # Rerun from earliest accountable subtask onward
        for idx in range(earliest_index, len(subtasks)):
            subtask = subtasks[idx]
            subtask_id = subtask["subtask_id"]

            agent_id = find_agent(subtask_id, agent_allocation_map)
            if agent_id is None:
                raise ValueError(f"No agent assigned for subtask {subtask_id}")

            agent_spec = find_agent_specification(agent_id, agent_specifications)
            if agent_spec is None:
                raise ValueError(f"No agent specification found for agent {agent_id}")

            target_key = (agent_id, subtask_id)
            feedback_claims = executor_claims_by_target.get(target_key, [])
            is_accountable_agent = len(feedback_claims) > 0

            if is_accountable_agent:
                accountable_feedback_targets.append({
                    "agent_id": agent_id,
                    "subtask_id": subtask_id,
                    "wrong_claim_count": len(feedback_claims),
                    "claim_ids": [
                        c.get("claim_id", "")
                        for c in feedback_claims
                    ]
                })
            else:
                downstream_retriggered_subtasks.append(subtask_id)

            previous_output = execution_outputs.get(
                subtask_id, {}
            ).get(
                "executor_output", {}
            ).get(
                "task_result", ""
            )

            context_dict = shared_memory.get_previous_subtask_outputs(subtask_id)

            retrigger_user_msg = build_executor_retrigger_user_msg(
                row=row,
                subtask=subtask,
                context_dict=context_dict,
                previous_output=previous_output,
                feedback_claims=feedback_claims,
                is_accountable_agent=is_accountable_agent
            )

            revised_executor_output = call_ollama_with_schema(
            system_msg=build_executor_system_msg(agent_spec),
            user_msg=retrigger_user_msg,
            schema=executor_schema,
            model=model,
            agent_id=agent_id,
            temperature=generation_temperature
        )

            # Version tracking in shared memory
            subtask_memory = shared_memory.memory["subtasks"][subtask_id]
            subtask_memory.setdefault("revision_history", []).append({
                "repair_round": repair_round,
                "old_output": previous_output,
                "new_output": revised_executor_output["task_result"],
                "was_accountable": is_accountable_agent,
                "feedback_claims": feedback_claims
            })

            # Update execution_outputs so synthesizer uses revised output
            execution_outputs[subtask_id]["executor_output"] = revised_executor_output
            execution_outputs[subtask_id]["executor_elapsed_sec_after_repair"] = None

            # Update shared memory active output
            shared_memory.update_executor_output(
                subtask_id=subtask_id,
                task_result=revised_executor_output["task_result"],
                confidence=revised_executor_output.get("confidence")
            )

            retriggered_subtasks.append(subtask_id)

    # -----------------------------------------------------
    # Case 2: no executor accountable
    # Only synthesizer / unknown claims exist
    # -----------------------------------------------------
    else:
        earliest_subtask_id = None
        earliest_index = None

    # -----------------------------------------------------
    # Always rerun synthesizer if HDM found wrong claims
    # -----------------------------------------------------
    repair_summary = {
        "repair_round": repair_round,
        "executor_claims_by_target": {
            f"{agent_id}/{subtask_id}": [
                c.get("claim_id", "")
                for c in claims
            ]
            for (agent_id, subtask_id), claims in executor_claims_by_target.items()
        },
        "synthesizer_claim_ids": [
            c.get("claim_id", "")
            for c in synthesizer_claims
        ],
        "unknown_claim_ids": [
            c.get("claim_id", "")
            for c in unknown_claims
        ],
        "earliest_retriggered_subtask": earliest_subtask_id,
        "retriggered_subtasks": retriggered_subtasks,
        "downstream_retriggered_subtasks": downstream_retriggered_subtasks
    }

    synthesizer_retrigger_user_msg = build_synthesizer_retrigger_user_msg(
        row=row,
        final_expected_milestones=final_expected_milestones,
        execution_outputs=execution_outputs,
        previous_final_article=synthesizer_output["task_result"],
        all_wrong_claims=wrong_claims_without_evidence,
        repair_summary=json.dumps(repair_summary, indent=2, ensure_ascii=False)
    )

    revised_synthesizer_output = call_ollama_with_schema(
    system_msg=synthesizer_system_msg,
    user_msg=synthesizer_retrigger_user_msg,
    schema=synthesizer_schema,
    model=model,
    temperature=generation_temperature
)

    revised_task_result = revised_synthesizer_output["task_result"]

    shared_memory.set_final_output(revised_synthesizer_output)

    # -----------------------------------------------------
    # Rerun HDM after repair
    # -----------------------------------------------------
    hdm_after = run_hdm_multi_agent(
    row=row,
    task_result=revised_task_result,
    shared_memory=shared_memory,
    model=model,
    hdm_temperature=hdm_temperature
)

    hdm_change = compute_hdm_change_status(
        hdm_before=hdm_before,
        hdm_after=hdm_after
    )

    repair_report = {
        "repair_round": repair_round,
        "repair_performed": True,
        "earliest_retriggered_subtask": earliest_subtask_id,
        "retriggered_subtasks": retriggered_subtasks,
        "accountable_feedback_targets": accountable_feedback_targets,
        "downstream_retriggered_subtasks": downstream_retriggered_subtasks,
        "synthesizer_retriggered": True,
        "synthesizer_claims": synthesizer_claims,
        "unknown_claims": unknown_claims,
        "accountable_stages": sorted(list(set(
        claim.get("accountable_stage", "unknown")
        for claim in wrong_claims
    ))),
        "hdm_change": hdm_change
    }

    shared_memory.memory.setdefault("repair_history", []).append(repair_report)
    shared_memory.memory["hdm_after_repair"] = hdm_after

    return {
        "repair_performed": True,
        "final_synthesizer_output": revised_synthesizer_output,
        "final_task_result": revised_task_result,
        "hdm_after": hdm_after,
        "repair_report": repair_report
    }    
    
    
# =========================================================
# Utility functions
# =========================================================
def safe_divide(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator

def find_agent(subtask_id, agent_allocation_map):
    for agent_id, tasks in agent_allocation_map.items():
        if subtask_id in tasks:
            return agent_id
    return None

def find_agent_specification(agent_id, agent_specifications):
    return agent_specifications.get(agent_id)

# =========================================================
# Single-row evaluation
# =========================================================
def evaluate_single_row_multi_agent(
    row,
    model=MODEL_NAME,
    generation_temperature=0.0,
    evaluator_temperature=0.0,
    hdm_temperature=0.0,
    run_id=1
):
    _AUTOGEN_AGENT_CACHE.clear()
    row_start = time.perf_counter()
    shared_memory = SharedMemory(main_task=row.get("event", ""))
    # Planner
    planner_start = time.perf_counter()
    planner_output = call_ollama_with_schema(
        system_msg=planner_system_msg,
        user_msg=build_planner_user_msg(row),
        schema=planner_schema,
        model=model,
        temperature=generation_temperature
    )
    #print("Planner output: \n", planner_output)
    planner_elapsed = time.perf_counter() - planner_start

    #print("Planner elapsed: ", planner_elapsed)
    
    subtasks = planner_output["Subtasks"]
    final_expected_milestones = planner_output["FinalExpectedMilestones"]
    
    shared_memory.set_planner_output(planner_output)
    shared_memory.add_subtasks_from_plan(subtasks)

    # Agent architect
    architect_start = time.perf_counter()
    agent_architect_output = call_ollama_with_schema(
        system_msg=architect_system_msg,
        user_msg=build_architect_user_msg(row, planner_output),
        schema=architect_schema,
        model=model,
        temperature=generation_temperature
    )
    architect_elapsed = time.perf_counter() - architect_start

    #print("Architect Output: \n", agent_architect_output)
    #print("Architect elapsed: ", architect_elapsed)
    
    agent_allocation_map = agent_architect_output["agent_allocation_map"]
    agent_specifications = agent_architect_output["agent_specs"]
    num_agents = len(agent_allocation_map)
    
    shared_memory.set_agent_architect_output(agent_architect_output)
    shared_memory.assign_agents(agent_allocation_map)
    
    subtask_id_list = [s["subtask_id"] for s in subtasks]
    execution_outputs = {}

    # Executors
    executor_total_start = time.perf_counter()

    for subtask in subtasks:
        subtask_id = subtask["subtask_id"]

        agent_id = find_agent(subtask_id, agent_allocation_map)
        if agent_id is None:
            raise ValueError(f"No agent assigned for subtask {subtask_id}")

        agent_spec = find_agent_specification(agent_id, agent_specifications)
        if agent_spec is None:
            raise ValueError(f"No agent specification found for agent {agent_id}")

        context_dict = shared_memory.get_previous_subtask_outputs(subtask_id)
        
        executor_start = time.perf_counter()
        executor_output = call_ollama_with_schema(
        system_msg=build_executor_system_msg(agent_spec),
        user_msg=build_executor_user_msg(row, subtask, context_dict),
        schema=executor_schema,
        model=model,
        agent_id=agent_id,
        temperature=generation_temperature
    )
        executor_elapsed = time.perf_counter() - executor_start

        #print("Executor elapsed: ", executor_elapsed)
        execution_outputs[subtask_id] = {
            "assigned_agent": agent_id,
            "agent_spec": agent_spec,
            "executor_output": executor_output,
            "executor_elapsed_sec": round(executor_elapsed, 3)
        }

        shared_memory.update_executor_output(
            subtask_id=subtask_id,
            task_result=executor_output["task_result"],
            confidence=executor_output.get("confidence")
        )
    executor_total_elapsed = time.perf_counter() - executor_total_start

    # Final writer / synthesizer
    synthesis_start = time.perf_counter()
    synthesizer_output = call_ollama_with_schema(
        system_msg=synthesizer_system_msg,
        user_msg=build_synthesizer_user_msg(row, final_expected_milestones, execution_outputs),
        schema=synthesizer_schema,
        model=model,
        temperature=generation_temperature
    )
    shared_memory.set_final_output(synthesizer_output)
    synthesis_elapsed = time.perf_counter() - synthesis_start

    task_result = synthesizer_output["task_result"]
    mandatory_total = sum(1 for m in final_expected_milestones if m.get("priority") == "MANDATORY")
    optional_total = sum(1 for m in final_expected_milestones if m.get("priority") == "OPTIONAL")

    # ---------------------------------------------------------
    # HDM before repair
    # ---------------------------------------------------------
    hdm_before_start = time.perf_counter()
    hdm_before = run_hdm_multi_agent(
    row=row,
    task_result=task_result,
    shared_memory=shared_memory,
    model=model,
    hdm_temperature=hdm_temperature
)

    # Filter invalid executor provenance pairs before retriggering
    hdm_before = filter_invalid_provenance_pairs(
        hdm_output=hdm_before,
        agent_allocation_map=agent_allocation_map
    )
    hdm_before_elapsed = time.perf_counter() - hdm_before_start

    shared_memory.memory["hdm_before_repair"] = hdm_before

    # ---------------------------------------------------------
    # HDM-guided retriggering / repair
    # ---------------------------------------------------------
    repair_start = time.perf_counter()
    repair_result = retrigger_after_hdm(
    row=row,
    subtasks=subtasks,
    agent_allocation_map=agent_allocation_map,
    agent_specifications=agent_specifications,
    execution_outputs=execution_outputs,
    final_expected_milestones=final_expected_milestones,
    shared_memory=shared_memory,
    synthesizer_output=synthesizer_output,
    hdm_before=hdm_before,
    model=model,
    repair_round=1,
    generation_temperature=generation_temperature,
    hdm_temperature=hdm_temperature
)
    repair_elapsed = time.perf_counter() - repair_start

    # Use revised output if repair happened; otherwise use original synthesized output
    synthesizer_output = repair_result["final_synthesizer_output"]
    task_result = repair_result["final_task_result"]
    hdm_after = repair_result["hdm_after"]
    repair_report = repair_result["repair_report"]

    shared_memory.memory["repair_report"] = repair_report

    # ---------------------------------------------------------
    # Judge final repaired article
    # ---------------------------------------------------------
    judge_start = time.perf_counter()
    judge_output = call_ollama_with_schema(
    system_msg=judge_system_msg,
    user_msg=build_judge_user_msg(row, final_expected_milestones, task_result),
    schema=judge_schema,
    model=model,
    temperature=evaluator_temperature
)
    shared_memory.set_judge_output(judge_output)
    judge_elapsed = time.perf_counter() - judge_start

    #print("Judge elapsed: ", judge_elapsed)
    total_elapsed = time.perf_counter() - row_start

    milestone_scores = judge_output.get("milestone_judgment", {}).get("milestone_scores", [])
    total_milestones = len(milestone_scores)
    achieved_milestones_count = sum(1 for m in milestone_scores if m.get("achieved", False))
    mandatory_milestones_achieved = sum(
        1 for m in milestone_scores
        if m.get("priority") == "MANDATORY" and m.get("achieved", False)
    )
    optional_milestones_achieved = sum(
        1 for m in milestone_scores
        if m.get("priority") == "OPTIONAL" and m.get("achieved", False)
    )
    achieved_milestone_ids = [
        m.get("milestone_id", "") for m in milestone_scores if m.get("achieved", False)
    ]

    dimension_scores = judge_output.get("dimension_scores", {})
    quality_analysis = judge_output.get("quality_analysis", {})
    summary = judge_output.get("summary", {})

    return {
        "row_id": row.get("row_id", ""),
        "model": model,
        "run_id": run_id,
        "generation_temperature": generation_temperature,
        "evaluator_temperature": evaluator_temperature,
        "hdm_temperature": hdm_temperature,
        "year": row.get("year", ""),
        "month": row.get("month", ""),
        "date": row.get("date", ""),
        "event": row.get("event", ""),
        "planner_elapsed_sec": round(planner_elapsed, 3),
        "architect_elapsed_sec": round(architect_elapsed, 3),
        "executor_total_elapsed_sec": round(executor_total_elapsed, 3),
        "synthesis_elapsed_sec": round(synthesis_elapsed, 3),
        "judge_elapsed_sec": round(judge_elapsed, 3),
        "hdm_before_elapsed_sec": round(hdm_before_elapsed, 3),
        "repair_elapsed_sec": round(repair_elapsed, 3),
        "total_elapsed_sec": round(total_elapsed, 3),

        "subtask_count": len(subtasks),
        "num_agents": num_agents,
        "total_milestones": total_milestones,
        "achieved_milestones_count": achieved_milestones_count,
        "achieved_milestone_ids_json": json.dumps(achieved_milestone_ids, ensure_ascii=False),
        "mandatory_milestones_achieved": mandatory_milestones_achieved,
        "optional_milestones_achieved": optional_milestones_achieved,
        "milestone_achievement_rate": safe_divide(achieved_milestones_count, total_milestones),
        "KPI": safe_divide(safe_divide(achieved_milestones_count, total_milestones), num_agents) if num_agents > 0 else 0,
        "mandatory_milestone_achievement_rate": safe_divide(mandatory_milestones_achieved, mandatory_total),
        "optional_milestone_achievement_rate": safe_divide(optional_milestones_achieved, optional_total),
        "total_milestone_score_normalized": safe_divide(
            sum(m.get("score_1_to_5", 0) for m in milestone_scores),
            total_milestones * 5
        ),

        "core_event_fidelity": dimension_scores.get("core_event_fidelity"),
        "required_entity_coverage": dimension_scores.get("required_entity_coverage"),
        "key_information_coverage": dimension_scores.get("key_information_coverage"),
        "cross_source_synthesis": dimension_scores.get("cross_source_synthesis"),
        "structural_coverage": dimension_scores.get("structural_coverage"),
        "statistics_and_claims_accuracy": dimension_scores.get("statistics_and_claims_accuracy"),
        "narrative_quality": dimension_scores.get("narrative_quality"),
        "total_dimension_score": safe_divide(
            sum(dimension_scores.get(dim, 0) for dim in [
                "core_event_fidelity",
                "required_entity_coverage",
                "key_information_coverage",
                "cross_source_synthesis",
                "structural_coverage",
                "statistics_and_claims_accuracy",
                "narrative_quality"
            ]),
            7
        ),

        "coherence": quality_analysis.get("coherence"),
        "readability": quality_analysis.get("readability"),
        "conciseness": quality_analysis.get("conciseness"),
        "total_quality_analysis_score": safe_divide(
            sum(quality_analysis.get(dim, 0) for dim in [
                "coherence",
                "readability",
                "conciseness"
            ]),
            3
        ),

        "strengths_json": json.dumps(summary.get("strengths", []), ensure_ascii=False),
        "weaknesses_json": json.dumps(summary.get("weaknesses", []), ensure_ascii=False),
        
        "planner_output_json": json.dumps(planner_output, ensure_ascii=False),
        "agent_architect_output_json": json.dumps(agent_architect_output, ensure_ascii=False),
        "execution_outputs_json": json.dumps(execution_outputs, ensure_ascii=False),
        "final_expected_milestones_json": json.dumps(final_expected_milestones, ensure_ascii=False),
        "generated_event_narrative": task_result,
        "synthesizer_output_json": json.dumps(synthesizer_output, ensure_ascii=False),
        "judge_output_json": json.dumps(judge_output, ensure_ascii=False),
        "hdm_wrong_claim_count_before": hdm_before["wrong_claim_count"],
        "hdm_wrong_claim_count_after": hdm_after["wrong_claim_count"],
        "hdm_total_claims_before": hdm_before["total_claims"],
        "hdm_total_claims_after": hdm_after["total_claims"],
        "hdm_hallucination_rate_before": hdm_before["hallucination_rate"],
        "hdm_hallucination_rate_after": hdm_after["hallucination_rate"],


        "invalid_provenance_claim_count": hdm_before.get("invalid_provenance_claim_count", 0),
        "invalid_provenance_claims_json": json.dumps(
            hdm_before.get("invalid_provenance_claims", []),
            ensure_ascii=False
        ),
        "repairable_wrong_claim_count": len(hdm_before.get("repairable_wrong_claims", [])),

        "repair_performed": repair_result["repair_performed"],
        "earliest_retriggered_subtask": repair_report.get(
        "earliest_retriggered_subtask",
        None
    ),
    "retriggered_subtasks_json": json.dumps(
        repair_report.get("retriggered_subtasks", []),
        ensure_ascii=False
    ),
    "downstream_retriggered_subtasks_json": json.dumps(
        repair_report.get("downstream_retriggered_subtasks", []),
        ensure_ascii=False
    ),
    "accountable_feedback_targets_json": json.dumps(
        repair_report.get("accountable_feedback_targets", []),
        ensure_ascii=False
    ),
    "accountable_stages_json": json.dumps(
        repair_report.get("accountable_stages", []),
        ensure_ascii=False
    ),
        "repair_report_json": json.dumps(repair_report, ensure_ascii=False),
        "hdm_before_json": json.dumps(hdm_before, ensure_ascii=False),
        "hdm_after_json": json.dumps(hdm_after, ensure_ascii=False),
        "shared_memory_json": shared_memory.to_json()
    }

# =========================================================
# Whole-dataset runner
# =========================================================
def run_multi_agent_hdm_on_dataset(
    csv_path,
    model=MODEL_NAME,
    generation_temperature=0.0,
    evaluator_temperature=0.0,
    hdm_temperature=0.0,
    run_id=1,
    max_retries=3,
    results_csv_path="MultiAgent+HDM_pns/MultiAgent+HDM_pns_gemma3/multiagent_hdm_pns_results_gemma3.csv"
):
    df = pd.read_csv(csv_path).fillna("")

    results = []

    os.makedirs(os.path.dirname(results_csv_path) or ".", exist_ok=True)

    dataset_start = time.perf_counter()

    for idx, row in df.iterrows():
        print(f"Processing row {idx + 1}/{len(df)} | row_id={row.get('row_id', idx)}")

        row_success = False
        last_error_msg = ""

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"Retrying row_index={idx} | retry {attempt}/{max_retries}")

                row_result = evaluate_single_row_multi_agent(
                    row,
                    model=model,
                    generation_temperature=generation_temperature,
                    evaluator_temperature=evaluator_temperature,
                    hdm_temperature=hdm_temperature,
                    run_id=run_id
                )

                row_result["row_index"] = idx
                row_result["retry_count"] = attempt
                row_result["run_status"] = "successful"

                results.append(row_result)
                row_success = True
                break

            except Exception as e:
                last_error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"Attempt {attempt}/{max_retries} failed for row_index={idx}: {last_error_msg}")

        if not row_success:
            results.append({
                "row_index": idx,
                "row_id": row.get("row_id", ""),
                "model": model,
                "run_id": run_id,
                "generation_temperature": generation_temperature,
                "evaluator_temperature": evaluator_temperature,
                "hdm_temperature": hdm_temperature,
                "year": row.get("year", ""),
                "month": row.get("month", ""),
                "date": row.get("date", ""),
                "event": row.get("event", ""),
                "retry_count": max_retries,
                "run_status": last_error_msg
            })

    dataset_elapsed = time.perf_counter() - dataset_start

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_csv_path, index=False)

    print("\nFinished dataset run.")
    print(f"Total dataset elapsed time: {round(dataset_elapsed, 3)} seconds")
    print(f"Total rows saved: {len(results_df)}")
    print(f"Successful rows: {(results_df['run_status'] == 'successful').sum()}")
    print(f"Errored rows after retries: {(results_df['run_status'] != 'successful').sum()}")
    print(f"Results saved to: {results_csv_path}")

    if len(results_df) > 0 and "planner_elapsed_sec" in results_df.columns:
        successful_df = results_df[results_df["run_status"] == "successful"]

        if len(successful_df) > 0:
            print(f"planner time: {round(successful_df['planner_elapsed_sec'].mean(), 3)} sec")
            print(f"architect time: {round(successful_df['architect_elapsed_sec'].mean(), 3)} sec")
            print(f"executor time: {round(successful_df['executor_total_elapsed_sec'].mean(), 3)} sec")
            print(f"synthesis time: {round(successful_df['synthesis_elapsed_sec'].mean(), 3)} sec")
            print(f"HDM before time: {round(successful_df['hdm_before_elapsed_sec'].mean(), 3)} sec")
            print(f"repair time: {round(successful_df['repair_elapsed_sec'].mean(), 3)} sec")
            print(f"judge time: {round(successful_df['judge_elapsed_sec'].mean(), 3)} sec")
            print(f"total time per row: {round(successful_df['total_elapsed_sec'].mean(), 3)} sec")

    return results_df, dataset_elapsed

if __name__ == "__main__":

    CSV_PATH = "public_narrative_synthesis_dataset.csv"

    TEMPERATURES = [0.0, 0.5, 1.0]
    RUNS_PER_TEMPERATURE = 3
    MAX_RETRIES = 3

    BASE_OUTPUT_DIR = "MultiAgent+HDM_pns/MultiAgent+HDM_pns_gemma3"

    all_results = []

    for temp in TEMPERATURES:
        for run_id in range(1, RUNS_PER_TEMPERATURE + 1):

            temp_label = temperature_label(temp)

            results_csv_path = os.path.join(
                BASE_OUTPUT_DIR,
                f"multiagent_hdm_pns_gemma3_generationT{temp_label}_judgeT0_hdmT0_run{run_id}.csv"
            )

            print("\n" + "=" * 80)
            print(
                f"Running MultiAgent+HDM PNS | model=gemma3:64k | "
                f"generation_temperature={temp} | "
                f"evaluator_temperature=0.0 | "
                f"hdm_temperature=0.0 | "
                f"run_id={run_id} | max_retries={MAX_RETRIES}"
            )
            print("=" * 80)

            results_df, dataset_elapsed = run_multi_agent_hdm_on_dataset(
                csv_path=CSV_PATH,
                model="gemma3:64k",
                generation_temperature=temp,
                evaluator_temperature=0.0,
                hdm_temperature=0.0,
                run_id=run_id,
                max_retries=MAX_RETRIES,
                results_csv_path=results_csv_path
            )

            all_results.append(results_df)

    if not all_results:
        raise RuntimeError("No result dataframes were created. Check earlier errors before concatenating.")

    combined_results_df = pd.concat(all_results, ignore_index=True)

    combined_results_path = os.path.join(
        BASE_OUTPUT_DIR,
        "multiagent_hdm_pns_gemma3_all_temperatures_combined_with_rerun.csv"
    )

    combined_results_df.to_csv(combined_results_path, index=False)

    print("\nTemperature experiment finished.")
    print(f"Combined results saved to: {combined_results_path}")
    print(f"Total rows: {len(combined_results_df)}")
    print(f"Successful rows: {(combined_results_df['run_status'] == 'successful').sum()}")
    print(f"Errored rows after retries: {(combined_results_df['run_status'] != 'successful').sum()}")
