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
import ast
# =========================================================
# Configuration
# =========================================================
MODEL_NAME = "gemma3:64k"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
TIMEOUT = 900

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
        "related_work_title": {"type": "string"},
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
                "source_grounding": {"type": "number"},
                "citation_faithfulness": {"type": "number"},
                "reference_coverage": {"type": "number"},
                "synthesis_quality": {"type": "number"},
                "organization": {"type": "number"},
                "specificity": {"type": "number"},
                "writing_quality": {"type": "number"}
            },
            "required": [
                "source_grounding",
                "citation_faithfulness",
                "reference_coverage",
                "synthesis_quality",
                "organization",
                "specificity",
                "writing_quality"
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
            "required": ["coherence", "readability", "conciseness"]
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
        "related_work_title",
        "milestone_judgment",
        "dimension_scores",
        "quality_analysis",
        "summary",
        "confidence"
    ]
}


# =========================================================
# HDM Schemas for Multi-Agent Related-Work Generation
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
                            "method",
                            "dataset",
                            "task",
                            "result",
                            "comparison",
                            "citation_attribution",
                            "entity",
                            "number",
                            "background",
                            "other"
                        ]
                    },
                    "cited_ids": {
                        "type": "array",
                        "items": {"type": "string"}
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
You are a Task Planner for a multi-agent scientific related-work generation system.

Your job is to take the main task and produce:
- A list of actionable subtasks
- Final expected milestones for the overall related-work section
- Confidence score for the plan from 0 to 1

The task is related-work section generation:
Given a query paper abstract and multiple cited reference abstracts, generate a scholarly related-work section grounded in the provided sources.

Instructions:
1. Analyze the query paper abstract and cited reference abstracts.
2. Break the task into clear, logically ordered subtasks.
3. The final output should synthesize the cited works in relation to the query paper.
4. Do not simply plan to summarize the query paper abstract.
5. Preserve citation identifiers such as @cite_0, @cite_1, etc. when discussing cited works.
6. Do NOT define milestones for individual subtasks.
7. For each subtask, provide:
   - subtask_id
   - subtask_description
8. After listing all subtasks, define final expected milestones for the overall related-work section.
9. Mandatory milestones must cover source grounding, citation faithfulness, reference coverage, synthesis quality, organization, specificity, and writing quality.
10. Optional milestones should improve nuance, flow, conciseness, or scholarly style.
11. There must be at least ONE MANDATORY milestone.
12. Provide a confidence score from 0 to 1.

Return ONLY valid JSON matching the required schema.

### EXAMPLE STRUCTURE
{
  "Subtasks": [
    {
      "subtask_id": "T1",
      "subtask_description": "Identify the main topic, problem setting, and contribution direction of the query paper from its abstract."
    },
    {
      "subtask_id": "T2",
      "subtask_description": "Analyze the cited reference abstracts and identify major themes, methods, and research lines."
    },
    {
      "subtask_id": "T3",
      "subtask_description": "Determine how the cited works relate to the query paper and group them into coherent related-work themes."
    },
    {
      "subtask_id": "T4",
      "subtask_description": "Draft a grounded scholarly related-work section using citation identifiers and avoiding unsupported claims."
    }
  ],
  "FinalExpectedMilestones": [
    {
      "milestone_id": "M1",
      "priority": "MANDATORY",
      "description": "The final related-work section must be grounded in the query abstract and cited reference abstracts."
    },
    {
      "milestone_id": "M2",
      "priority": "MANDATORY",
      "description": "The final related-work section must correctly attribute claims to cited works using citation identifiers."
    },
    {
      "milestone_id": "M3",
      "priority": "MANDATORY",
      "description": "The final related-work section must synthesize the cited works instead of listing them mechanically."
    },
    {
      "milestone_id": "M4",
      "priority": "OPTIONAL",
      "description": "The final related-work section may highlight relationships, contrasts, or gaps among the cited works."
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
{
  "agent_allocation_map": {
    "A1": ["T1"],
    "A2": ["T2", "T3"],
    "A3": ["T4"]
  },
  "agent_specs": {
  "A1": {
    "actor_type": "Query Analysis Agent",
    "skills": ["query abstract interpretation", "scientific problem identification"],
    "rationale": "This agent identifies the query paper's main topic and contribution direction."
  },
  "A2": {
    "actor_type": "Reference Analysis Agent",
    "skills": ["reference abstract analysis", "theme grouping", "citation-aware interpretation"],
    "rationale": "This agent analyzes the cited reference abstracts and groups them into related-work themes."
  },
  "A3": {
    "actor_type": "Related Work Synthesis Agent",
    "skills": ["scientific related-work writing", "source-grounded synthesis", "citation attribution"],
    "rationale": "This agent helps synthesize the query and cited works into a coherent related-work section."
  }
},
  "confidence": 0.9
}
"""
def build_executor_system_msg(actor_spec):
    return f"""
You are an Executor Agent in an autonomous multi-agent system.
- assigned subtask
- Your role as an agent is a {actor_spec['actor_type']}.
- Your expertise/skills: {actor_spec['skills']}.
Your job is to execute the subtask provided in the user message accurately and safely.

EXECUTION RULES:
1. Stay strictly within your assigned role and expertise.
2. Use provided context when relevant.
3. Keep your reasoning consistent with the provided context.
4. You need to perform the assigned subtask to the best of your ability and place the output in the "task_result" field.
5. Provide confidence about your task result in the "confidence" field (0 to 1).

Output Rules:
- Confidence must be between 0 and 1.
- Output must be valid JSON only according to the provided schema.
- Do NOT include any explanation outside JSON.

Your output must be ONLY valid JSON.

### EXAMPLE:
{{
  "subtask_id": "T1",
  "task_result": "The query paper focuses on neural methods for scientific document understanding. The cited references discuss related neural summarization, citation context modeling, and multi-document synthesis approaches. These themes can be grouped around scientific summarization and citation-aware generation.",
  "confidence": 0.9
}}
"""


synthesizer_system_msg = """
You are the Final Writer in a multi-agent scientific related-work generation system.

You will receive:
1. A query paper abstract
2. Cited reference abstracts
3. Final expected milestones
4. Executor outputs from specialized agents

Your task is to synthesize the executor outputs into one coherent scholarly related-work section.

The final output should:
- Be grounded only in the query abstract and cited reference abstracts
- Synthesize the cited works in relation to the query paper
- Use citation identifiers such as @cite_0 or @cite_3 where appropriate
- Avoid unsupported claims, invented methods, invented datasets, invented results, or incorrect citation attribution
- Sound like a scientific related-work section
- Avoid bullet points, section headers, and markdown
- Place the complete related-work section in the "task_result" field
- Provide a confidence score from 0 to 1

Output MUST be ONLY valid JSON.

### EXAMPLE:
{
  "task_result": "Prior work has explored neural approaches for scientific document understanding and citation-aware summarization. Several studies focus on modeling document structure and citation context for summarization tasks @cite_0 @cite_1, while others investigate multi-document synthesis across related scientific papers @cite_2. These lines of work provide a foundation for the query paper's focus on generating related-work text from a query abstract and cited reference abstracts.",
  "confidence": 0.9
}
"""

judge_system_msg = """
You are an expert evaluator for scientific related-work generation.

You will be given:
1. A query paper abstract
2. The cited reference abstracts
3. A generated related-work section to evaluate
4. A set of final expected milestones
5. The original gold related-work section

Your task is to judge the generated related-work section.

You must evaluate:
1. Milestone achievement
2. Source grounding
3. Citation faithfulness
4. Reference coverage
5. Synthesis quality
6. Organization
7. Specificity
8. Writing quality
9. Quality subdimensions:
   - coherence
   - readability
   - conciseness

Dimension definitions:
- source_grounding: Are the claims supported by the query abstract and reference abstracts?
- citation_faithfulness: Are cited works attributed correctly using the provided citation identifiers?
- reference_coverage: Does the related work section cover the important cited works or themes from the references?
- synthesis_quality: Does it synthesize relationships among works instead of listing papers one by one?
- organization: Is the related work section logically structured like scholarly related work?
- specificity: Does it include concrete scientific concepts rather than generic statements?
- writing_quality: Is it fluent, concise, academic, and readable?

Evaluation guidance:
- Do not require exact wording overlap with the gold related-work section.
- Use the gold related-work section as a reference for expected content, but judge semantic quality and grounding.
- Penalize unsupported claims, incorrect citation attribution, invented methods, invented results, invented datasets, or claims not supported by the provided abstracts.
- Penalize outputs that ignore most reference abstracts or discuss the query paper only.
- Missing optional details should not be penalized heavily.
- Missing mandatory milestones should reduce the score.

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
- Do not compute any aggregate milestone counts.
- Only provide milestone-level judgments.

Dimension score rules:
- Score each dimension from 0.0 to 1.0.
- 1.0 = fully satisfied.
- 0.5 = partially satisfied.
- 0.0 = not satisfied.
- Use intermediate values when appropriate.

Return ONLY valid JSON matching the required schema.
Do not include markdown or extra text outside the JSON object.

### EXAMPLE FORMAT
{
  "related_work_title": "Citation-Aware Scientific Related Work",
  "milestone_judgment": {
    "milestone_scores": [
      {
        "milestone_id": "M1",
        "priority": "MANDATORY",
        "description": "The final related-work section must be grounded in the query abstract and cited reference abstracts.",
        "achieved": true,
        "score_1_to_5": 5,
        "reason": "The generated related-work section uses only information supported by the provided abstracts."
      },
      {
        "milestone_id": "M2",
        "priority": "MANDATORY",
        "description": "The final related-work section must correctly attribute claims to cited works using citation identifiers.",
        "achieved": true,
        "score_1_to_5": 4,
        "reason": "Most claims are tied to citation identifiers, though one attribution could be more precise."
      }
    ]
  },
  "dimension_scores": {
    "source_grounding": 0.9,
    "citation_faithfulness": 0.85,
    "reference_coverage": 0.8,
    "synthesis_quality": 0.9,
    "organization": 0.9,
    "specificity": 0.85,
    "writing_quality": 0.9
  },
  "quality_analysis": {
    "coherence": 0.9,
    "readability": 0.9,
    "conciseness": 0.85
  },
  "summary": {
    "strengths": [
      "The related work section synthesizes cited work in relation to the query paper.",
      "The output is mostly grounded and uses citation identifiers."
    ],
    "weaknesses": [
      "Some cited works could be discussed with more specific attribution."
    ]
  },
  "confidence": 0.9
}
"""

# =========================================================
# HDM System Messages for Related-Work Generation
# =========================================================

claim_extraction_system_msg = """
You are hdm_claim_extractor_agent, a claim extraction agent for scientific related-work generation.

Your task is to extract atomic factual claims from the generated related-work section.

Rules:
- Extract only factual scientific claims.
- Extract claims about methods, datasets, tasks, findings, comparisons, background, and citation attribution.
- Treat a sentence that attributes a claim to a citation identifier such as @cite_0 as a citation-attribution claim.
- Split compound sentences into separate atomic claims.
- Do not extract opinions, vague commentary, or purely stylistic statements.
- Each claim must be independently checkable against the query paper abstract and cited reference abstracts.
- Preserve any citation identifiers that appear in the source sentence.
- Return ONLY valid JSON matching the schema.
Do not include markdown or extra text outside the JSON object.
"""


hdm_verification_provenance_system_msg = """
You are hdm_verification_provenance_agent for a multi-agent scientific related-work generation system.

Your task has two parts.

Part 1: Source-grounded claim verification
- Verify each extracted claim against ONLY the query paper abstract and cited reference abstracts.
- Use the original gold related-work paragraph only as auxiliary reference context, not as the sole evidence for support.
- Do not use internal knowledge, web search, or unsupported assumptions.
- Label each claim as:
  - supported: the claim is directly supported by the query abstract or cited reference abstracts.
  - unsupported: the claim is not supported by the query abstract or cited reference abstracts.
  - contradicted: the claim conflicts with the query abstract or cited reference abstracts.
- If a claim cites @cite_i, verify that the corresponding cited reference abstract supports that claim.
- Be strict about methods, datasets, tasks, findings, comparisons, numbers, named entities, and citation attribution.
- Provide clear evidence and reason for every verification label.

Part 2: Multi-agent provenance tracing
- For every unsupported or contradicted claim, trace where the claim likely originated in the multi-agent workflow.
- You will receive executor outputs, assigned agent IDs, subtask IDs, and the final synthesized related-work section.
- If the wrong claim appears in an executor output, identify that executor agent ID and corresponding subtask ID.
- If the wrong claim does not appear in executor outputs but appears in the final related-work section, assign accountability to the synthesizer.
- If the accountable stage is synthesizer, use accountable_agent_id = "not_applicable" and accountable_subtask_id = "not_applicable".
- If the origin cannot be determined, use accountable_stage = "unknown".
- Use semantic matching, not only exact string matching.

For supported claims:
- Use accountable_stage = "not_applicable".
- Use accountable_agent_id = "not_applicable".
- Use accountable_subtask_id = "not_applicable".
- Use agent_contribution = "not_applicable".
- Use trace_evidence = "Claim is supported; provenance tracing not required."

Rules:
- Do not guess beyond the provided abstracts and execution trace.
- Do not provide corrected claims unless the correction is explicitly supported by the provided evidence.
- Return ONLY valid JSON matching the schema.
Do not include markdown or extra text outside the JSON object.
"""

# =========================================================
# Prompt builders
# =========================================================
def build_initial_task(row):
    return f"""
Generate a high-quality scholarly related-work section using a multi-agent workflow.

Query Paper Abstract:
{get_row_value(row, 'abstract')}

Cited Reference Abstracts:
{format_reference_abstracts(row)}

Overall objective:
Produce one coherent scholarly related-work section. The section should synthesize the cited reference abstracts in relation to the query paper, use citation identifiers where appropriate, and avoid unsupported claims.
"""

def build_planner_user_msg(row):
    return f"""
Main Task: Generate a scholarly related-work section for a query paper using Multi-XScience data.

Query Paper Abstract:
{get_row_value(row, 'abstract')}

Cited Reference Abstracts:
{format_reference_abstracts(row)}

Produce subtasks T1, T2, etc., final milestones M1, M2, etc., and confidence about the plan.
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
3. confidence about your agent design and allocation plan
"""

def build_executor_user_msg(row, subtask, context_dict):
    return f"""
Query Paper Abstract:
{get_row_value(row, 'abstract')}

Cited Reference Abstracts:
{format_reference_abstracts(row)}

Assigned subtask:
{subtask['subtask_description']}

Subtask ID:
{subtask['subtask_id']}

Context from previously completed subtasks:
{json.dumps(context_dict, indent=2, ensure_ascii=False)}

Complete only this subtask now. Stay grounded in the query abstract and cited reference abstracts. Do not invent unsupported claims.
"""

def build_synthesizer_user_msg(row, final_expected_milestones, execution_outputs):
    simplified_outputs = {
        subtask_id: output["executor_output"]["task_result"]
        for subtask_id, output in execution_outputs.items()
    }

    return f"""
Write the final scholarly related-work section using the materials below.

Query Paper Abstract:
{get_row_value(row, 'abstract')}

Cited Reference Abstracts:
{format_reference_abstracts(row)}

Final Expected Milestones:
{json.dumps(final_expected_milestones, indent=2, ensure_ascii=False)}

Executor Outputs:
{json.dumps(simplified_outputs, indent=2, ensure_ascii=False)}

Write one coherent related-work section. The section must synthesize the cited works in relation to the query paper, use citation identifiers where appropriate, and avoid unsupported claims.

Return ONLY valid JSON with:
- task_result
- confidence
"""

def build_judge_user_msg(row, final_expected_milestones, task_result):
    reference_record = {
        "split": get_row_value(row, "split"),
        "row_id": get_row_value(row, "row_id"),
        "aid": get_row_value(row, "aid"),
        "mid": get_row_value(row, "mid"),
        "raw_ref_count": get_row_value(row, "raw_ref_count"),
        "ref_bin": get_row_value(row, "ref_bin"),
        "query_abstract": get_row_value(row, "abstract"),
        "reference_abstracts": format_reference_abstracts(row),
        "gold_related_work": get_row_value(row, "related_work")
    }

    return f"""
Evaluate the following generated related-work section.

Reference Multi-XScience Record:
{json.dumps(reference_record, indent=2, ensure_ascii=False)}

Final Expected Milestones:
{json.dumps(final_expected_milestones, indent=2, ensure_ascii=False)}

Generated Related Work:
{task_result}

Gold Related Work Section:
{get_row_value(row, 'related_work')}

Return ONLY valid JSON with:
- related_work_title
- milestone_judgment
- dimension_scores
- quality_analysis
- summary
- confidence
"""

# =========================================================
# HDM Prompt Builders for Related-Work Generation
# =========================================================

def build_claim_extraction_user_msg(task_result):
    return f"""
Generated Related-Work Section:
{task_result}

Extract atomic factual claims from this related-work section.

Return ONLY valid JSON with:
- claims
"""


def build_execution_trace_text(shared_memory, final_related_work):
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
[SYNTHESIZER FINAL RELATED-WORK OUTPUT]
{final_related_work}
"""
    )

    return "\n\n".join(blocks)


def build_verification_provenance_user_msg(row, extracted_claims, shared_memory, final_related_work):
    reference_record = {
        "split": get_row_value(row, "split"),
        "row_id": get_row_value(row, "row_id"),
        "aid": get_row_value(row, "aid"),
        "mid": get_row_value(row, "mid"),
        "raw_ref_count": get_row_value(row, "raw_ref_count"),
        "ref_bin": get_row_value(row, "ref_bin"),
        "query_abstract": get_row_value(row, "abstract"),
        "reference_abstracts": format_reference_abstracts(row),
        "gold_related_work": get_row_value(row, "related_work")
    }

    return f"""
Extracted Claims:
{json.dumps(extracted_claims, indent=2, ensure_ascii=False)}

Reference Multi-XScience Record:
{json.dumps(reference_record, indent=2, ensure_ascii=False)}

Multi-Agent Execution Trace:
{build_execution_trace_text(shared_memory, final_related_work)}

Verify every extracted claim against the query paper abstract and cited reference abstracts.
For unsupported or contradicted claims, trace the accountable executor, subtask, or synthesizer stage.

Important:
- Use the query abstract and cited reference abstracts as the primary evidence.
- Use the gold related-work paragraph only as auxiliary reference context.
- If a generated claim cites @cite_i, verify that the corresponding cited reference abstract supports that claim.
- Mark unsupported claims, invented paper details, invented methods, invented datasets, invented results, and incorrect citation attributions as unsupported or contradicted.
- For unsupported or contradicted claims, trace whether the claim came from an executor output or was introduced/distorted by the synthesizer.

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
            "timeout": timeout
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
# HDM for Multi-Agent Related-Work Generation
# =========================================================

def run_hdm_multi_agent(
    row,
    task_result,
    shared_memory,
    agent_allocation_map,
    model=MODEL_NAME,
    hdm_temperature=0.0
):
    """
    Multi-agent HDM for related-work generation:
    1. Extract claims from final synthesized related-work section.
    2. Verify claims against query abstract + cited reference abstracts.
    3. Trace unsupported/contradicted claims to executor/synthesizer.
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

    # Step 2: claim verification + provenance tracing
    verification_provenance_output = call_ollama_with_schema(
        system_msg=hdm_verification_provenance_system_msg,
        user_msg=build_verification_provenance_user_msg(
            row=row,
            extracted_claims=extracted_claims,
            shared_memory=shared_memory,
            final_related_work=task_result
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

    hdm_output = {
        "extracted_claims": extracted_claims,
        "claim_verifications": claim_verifications,
        "wrong_claims": wrong_claims,
        "total_claims": total_claims,
        "wrong_claim_count": wrong_claim_count,
        "hallucination_rate": hallucination_rate
    }

    hdm_output = filter_invalid_provenance_pairs(
        hdm_output=hdm_output,
        agent_allocation_map=agent_allocation_map
    )

    accountable_agent_counts = {}
    accountable_subtask_counts = {}

    for claim in hdm_output.get("repairable_wrong_claims", wrong_claims):
        agent_id = claim.get("accountable_agent_id", "unknown")
        subtask_id = claim.get("accountable_subtask_id", "unknown")

        accountable_agent_counts[agent_id] = accountable_agent_counts.get(agent_id, 0) + 1
        accountable_subtask_counts[subtask_id] = accountable_subtask_counts.get(subtask_id, 0) + 1

    hdm_output["accountable_agent_counts"] = accountable_agent_counts
    hdm_output["accountable_subtask_counts"] = accountable_subtask_counts

    return hdm_output


def get_subtask_order_map(subtasks):
    """
    Returns a map like {"T1": 0, "T2": 1, ...}
    based on the original planner subtask order.
    """
    return {
        subtask["subtask_id"]: idx
        for idx, subtask in enumerate(subtasks)
    }


def is_valid_agent_subtask_pair(agent_id, subtask_id, agent_allocation_map):
    """
    Checks whether HDM's claimed accountable agent/subtask pair
    is valid according to the original agent allocation map.
    """
    if agent_id not in agent_allocation_map:
        return False

    return subtask_id in agent_allocation_map.get(agent_id, [])


def filter_invalid_provenance_pairs(hdm_output, agent_allocation_map):
    """
    Filters wrong claims whose executor provenance does not match
    the agent allocation map.

    Keeps synthesizer and unknown claims as repairable because they do
    not require an executor/subtask pair.
    """
    all_wrong_claims = hdm_output.get("wrong_claims", [])

    valid_wrong_claims = []
    invalid_wrong_claims = []

    for claim in all_wrong_claims:
        stage = claim.get("accountable_stage", "unknown")
        agent_id = claim.get("accountable_agent_id", "")
        subtask_id = claim.get("accountable_subtask_id", "")

        if stage == "executor":
            if is_valid_agent_subtask_pair(agent_id, subtask_id, agent_allocation_map):
                valid_wrong_claims.append(claim)
            else:
                invalid_wrong_claims.append(claim)

        elif stage in ["synthesizer", "unknown"]:
            valid_wrong_claims.append(claim)

        else:
            valid_wrong_claims.append(claim)

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
        hdm_output.get("wrong_claims", [])
    ):
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
    Builds feedback for accountable executor agents.
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
Otherwise, keep your previous output as close as possible.
"""

    return f"""
You are rerunning one executor subtask in a multi-agent scientific related-work generation workflow.

Query Paper Abstract:
{get_row_value(row, 'abstract')}

Cited Reference Abstracts:
{format_reference_abstracts(row)}

Assigned Subtask:
{subtask['subtask_description']}

Subtask ID:
{subtask['subtask_id']}

Previous Output for This Subtask:
{json.dumps(previous_output, indent=2, ensure_ascii=False)}

Context from Updated Previously Completed Subtasks:
{json.dumps(context_dict, indent=2, ensure_ascii=False)}

HDM / Repair Feedback:
{feedback_text}

Revision Rules:
- Stay strictly within your assigned subtask.
- Use only the query abstract and cited reference abstracts as evidence.
- Do not use outside knowledge.
- Do not invent paper details, datasets, methods, results, comparisons, or citation attributions.
- Preserve valid citation identifiers such as @cite_0, @cite_1, etc.
- If the feedback identifies an unsupported or contradicted claim, remove it or rewrite it only if the provided abstracts support the correction.
- Return ONLY valid JSON with:
  - subtask_id
  - task_result
  - confidence
"""


def build_synthesizer_retrigger_user_msg(
    row,
    final_expected_milestones,
    execution_outputs,
    previous_final_related_work,
    all_wrong_claims,
    repair_summary
):
    simplified_outputs = {
        subtask_id: output["executor_output"]["task_result"]
        for subtask_id, output in execution_outputs.items()
    }

    return f"""
You are rerunning the final synthesizer in a multi-agent scientific related-work generation workflow.

Query Paper Abstract:
{get_row_value(row, 'abstract')}

Cited Reference Abstracts:
{format_reference_abstracts(row)}

Final Expected Milestones:
{json.dumps(final_expected_milestones, indent=2, ensure_ascii=False)}

Previous Final Related-Work Section:
{previous_final_related_work}

Updated Executor Outputs:
{json.dumps(simplified_outputs, indent=2, ensure_ascii=False)}

Repair Summary:
{repair_summary}

HDM Wrong Claims from Previous Final Related-Work Section:
{json.dumps(all_wrong_claims, indent=2, ensure_ascii=False)}

Synthesizer Repair Instructions:
- Do NOT restart from scratch.
- Use the updated executor outputs and the provided abstracts.
- Preserve supported parts of the previous final related-work section when possible.
- Remove or correct unsupported and contradicted claims.
- Do not use invalidated claims from old executor outputs.
- Do not introduce new unsupported paper details, datasets, methods, results, comparisons, or citation attributions.
- Preserve valid citation identifiers.
- Maintain a coherent scholarly related-work style.
- Return ONLY valid JSON with:
  - task_result
  - confidence
"""

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

    # Case 1: at least one executor is accountable
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

            feedback_claims = executor_claims_by_target.get(
                (agent_id, subtask_id),
                []
            )

            is_accountable_agent = len(feedback_claims) > 0

            if is_accountable_agent:
                accountable_feedback_targets.append({
                    "agent_id": agent_id,
                    "subtask_id": subtask_id,
                    "claim_ids": [
                        claim.get("claim_id", "")
                        for claim in feedback_claims
                    ]
                })
            else:
                downstream_retriggered_subtasks.append(subtask_id)

            context_dict = shared_memory.get_previous_subtask_outputs(subtask_id)

            previous_output = execution_outputs.get(
                subtask_id,
                {}
            ).get("executor_output", {})

            executor_retrigger_user_msg = build_executor_retrigger_user_msg(
                row=row,
                subtask=subtask,
                context_dict=context_dict,
                previous_output=previous_output,
                feedback_claims=feedback_claims,
                is_accountable_agent=is_accountable_agent
            )

            revised_executor_output = call_ollama_with_schema(
                system_msg=build_executor_system_msg(agent_spec),
                user_msg=executor_retrigger_user_msg,
                schema=executor_schema,
                model=model,
                agent_id=agent_id,
                temperature=generation_temperature
            )

            execution_outputs[subtask_id]["executor_output"] = revised_executor_output

            shared_memory.update_executor_output(
                subtask_id=subtask_id,
                task_result=revised_executor_output["task_result"],
                confidence=revised_executor_output.get("confidence")
            )

            retriggered_subtasks.append(subtask_id)

    # Case 2: only synthesizer/unknown claims exist
    else:
        earliest_subtask_id = None
        earliest_index = None

    # Always rerun synthesizer if HDM found wrong claims
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
        previous_final_related_work=synthesizer_output["task_result"],
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

    # Rerun HDM after repair
    hdm_after = run_hdm_multi_agent(
        row=row,
        task_result=revised_task_result,
        shared_memory=shared_memory,
        agent_allocation_map=agent_allocation_map,
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
def get_row_value(row, col, default=""):
    value = row.get(col, default)
    if pd.isna(value):
        return default
    return str(value)


def parse_ref_abstract(value):
    """
    Handles ref_abstract whether it is already a dict from JSONL
    or a string after loading from CSV.
    """
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        value = value.strip()

        if value == "" or value.lower() == "nan":
            return {}

        try:
            return json.loads(value)
        except Exception:
            pass

        try:
            return ast.literal_eval(value)
        except Exception:
            return {}

    return {}


def format_reference_abstracts(row):
    """
    Formats Multi-XScience reference abstracts for the prompt.
    Expected format:
        {"@cite_0": {"abstract": "...", "mid": "..."}}
    """
    ref_abstract = parse_ref_abstract(row.get("ref_abstract", {}))

    formatted_refs = []

    for cite_id, ref_info in ref_abstract.items():
        if isinstance(ref_info, dict):
            ref_mid = ref_info.get("mid", "")
            ref_abs = ref_info.get("abstract", "")
        else:
            ref_mid = ""
            ref_abs = str(ref_info)

        if str(ref_abs).strip():
            formatted_refs.append(
                f"Citation ID: {cite_id}\n"
                f"Reference MID: {ref_mid}\n"
                f"Reference Abstract: {ref_abs}"
            )

    return "\n\n".join(formatted_refs)


def load_subset_dataframe(data_path):
    if data_path.endswith(".jsonl"):
        return pd.read_json(data_path, lines=True)

    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
        if "ref_abstract" in df.columns:
            df["ref_abstract"] = df["ref_abstract"].apply(parse_ref_abstract)
        return df

    raise ValueError(f"Unsupported file format: {data_path}")

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
    run_id=1,
    use_hdm_repair=True
):
    _AUTOGEN_AGENT_CACHE.clear()

    row_start = time.perf_counter()

    shared_memory = SharedMemory(
        main_task=f"related_work_generation_row_{get_row_value(row, 'row_id')}"
    )

    # -------------------------
    # Planner
    # -------------------------
    planner_start = time.perf_counter()

    planner_output = call_ollama_with_schema(
        system_msg=planner_system_msg,
        user_msg=build_planner_user_msg(row),
        schema=planner_schema,
        model=model,
        temperature=generation_temperature
    )

    planner_elapsed = time.perf_counter() - planner_start

    subtasks = planner_output["Subtasks"]
    final_expected_milestones = planner_output["FinalExpectedMilestones"]

    shared_memory.set_planner_output(planner_output)
    shared_memory.add_subtasks_from_plan(subtasks)

    # -------------------------
    # Agent Architect
    # -------------------------
    architect_start = time.perf_counter()

    agent_architect_output = call_ollama_with_schema(
        system_msg=architect_system_msg,
        user_msg=build_architect_user_msg(row, planner_output),
        schema=architect_schema,
        model=model,
        temperature=generation_temperature
    )

    architect_elapsed = time.perf_counter() - architect_start

    agent_allocation_map = agent_architect_output["agent_allocation_map"]
    agent_specifications = agent_architect_output["agent_specs"]
    num_agents = len(agent_allocation_map)

    shared_memory.set_agent_architect_output(agent_architect_output)
    shared_memory.assign_agents(agent_allocation_map)

    execution_outputs = {}

    # -------------------------
    # Executors
    # -------------------------
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

    # -------------------------
    # Initial Synthesizer
    # -------------------------
    synthesis_start = time.perf_counter()

    synthesizer_output = call_ollama_with_schema(
        system_msg=synthesizer_system_msg,
        user_msg=build_synthesizer_user_msg(
            row,
            final_expected_milestones,
            execution_outputs
        ),
        schema=synthesizer_schema,
        model=model,
        temperature=generation_temperature
    )

    shared_memory.set_final_output(synthesizer_output)

    synthesis_elapsed = time.perf_counter() - synthesis_start

    initial_task_result = synthesizer_output["task_result"]

    # -------------------------
    # HDM before repair
    # -------------------------
    hdm_before_start = time.perf_counter()

    hdm_before = run_hdm_multi_agent(
        row=row,
        task_result=initial_task_result,
        shared_memory=shared_memory,
        agent_allocation_map=agent_allocation_map,
        model=model,
        hdm_temperature=hdm_temperature
    )

    shared_memory.memory["hdm_before_repair"] = hdm_before

    hdm_before_elapsed = time.perf_counter() - hdm_before_start

    # -------------------------
    # HDM-guided repair
    # -------------------------
    repair_start = time.perf_counter()

    if use_hdm_repair and hdm_before.get("wrong_claim_count", 0) > 0:
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
    else:
        repair_result = {
            "repair_performed": False,
            "final_synthesizer_output": synthesizer_output,
            "final_task_result": initial_task_result,
            "hdm_after": hdm_before,
            "repair_report": {
                "repair_round": 1,
                "repair_performed": False,
                "reason": "HDM found no wrong claims or repair disabled."
            }
        }

    repair_elapsed = time.perf_counter() - repair_start

    final_synthesizer_output = repair_result["final_synthesizer_output"]
    final_task_result = repair_result["final_task_result"]
    hdm_after = repair_result["hdm_after"]
    repair_report = repair_result["repair_report"]

    hdm_change = compute_hdm_change_status(
        hdm_before=hdm_before,
        hdm_after=hdm_after
    )

    # -------------------------
    # Judge final output after repair
    # -------------------------
    mandatory_total = sum(
        1 for m in final_expected_milestones
        if m.get("priority") == "MANDATORY"
    )

    optional_total = sum(
        1 for m in final_expected_milestones
        if m.get("priority") == "OPTIONAL"
    )

    judge_start = time.perf_counter()

    judge_output = call_ollama_with_schema(
        system_msg=judge_system_msg,
        user_msg=build_judge_user_msg(
            row,
            final_expected_milestones,
            final_task_result
        ),
        schema=judge_schema,
        model=model,
        temperature=evaluator_temperature
    )

    shared_memory.set_judge_output(judge_output)

    judge_elapsed = time.perf_counter() - judge_start

    total_elapsed = time.perf_counter() - row_start

    # -------------------------
    # Metric extraction
    # -------------------------
    milestone_scores = judge_output.get("milestone_judgment", {}).get("milestone_scores", [])

    total_milestones = len(milestone_scores)

    achieved_milestones_count = sum(
        1 for m in milestone_scores
        if m.get("achieved", False)
    )

    mandatory_milestones_achieved = sum(
        1 for m in milestone_scores
        if m.get("priority") == "MANDATORY" and m.get("achieved", False)
    )

    optional_milestones_achieved = sum(
        1 for m in milestone_scores
        if m.get("priority") == "OPTIONAL" and m.get("achieved", False)
    )

    achieved_milestone_ids = [
        m.get("milestone_id", "")
        for m in milestone_scores
        if m.get("achieved", False)
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
        "planner_temperature": generation_temperature,
        "architect_temperature": generation_temperature,
        "executor_temperature": generation_temperature,
        "synthesizer_temperature": generation_temperature,
        "judge_temperature": evaluator_temperature,
        "hdm_temperature": hdm_temperature,

        "split": get_row_value(row, "split"),
        "aid": get_row_value(row, "aid"),
        "mid": get_row_value(row, "mid"),
        "raw_ref_count": get_row_value(row, "raw_ref_count"),
        "ref_bin": get_row_value(row, "ref_bin"),
        "query_abstract": get_row_value(row, "abstract"),
        "gold_related_work": get_row_value(row, "related_work"),

        "planner_elapsed_sec": round(planner_elapsed, 3),
        "architect_elapsed_sec": round(architect_elapsed, 3),
        "executor_total_elapsed_sec": round(executor_total_elapsed, 3),
        "synthesis_elapsed_sec": round(synthesis_elapsed, 3),
        "hdm_before_elapsed_sec": round(hdm_before_elapsed, 3),
        "repair_elapsed_sec": round(repair_elapsed, 3),
        "judge_elapsed_sec": round(judge_elapsed, 3),
        "total_elapsed_sec": round(total_elapsed, 3),

        "subtask_count": len(subtasks),
        "num_agents": num_agents,

        "total_milestones": total_milestones,
        "achieved_milestones_count": achieved_milestones_count,
        "achieved_milestone_ids_json": json.dumps(achieved_milestone_ids, ensure_ascii=False),
        "mandatory_milestones_achieved": mandatory_milestones_achieved,
        "optional_milestones_achieved": optional_milestones_achieved,

        "milestone_achievement_rate": safe_divide(
            achieved_milestones_count,
            total_milestones
        ),
        "KPI": safe_divide(
            safe_divide(achieved_milestones_count, total_milestones),
            num_agents
        ) if num_agents > 0 else 0,
        "mandatory_milestone_achievement_rate": safe_divide(
            mandatory_milestones_achieved,
            mandatory_total
        ),
        "optional_milestone_achievement_rate": safe_divide(
            optional_milestones_achieved,
            optional_total
        ),
        "total_milestone_score_normalized": safe_divide(
            sum(m.get("score_1_to_5", 0) for m in milestone_scores),
            total_milestones * 5
        ),

        "source_grounding": dimension_scores.get("source_grounding"),
        "citation_faithfulness": dimension_scores.get("citation_faithfulness"),
        "reference_coverage": dimension_scores.get("reference_coverage"),
        "synthesis_quality": dimension_scores.get("synthesis_quality"),
        "organization": dimension_scores.get("organization"),
        "specificity": dimension_scores.get("specificity"),
        "writing_quality": dimension_scores.get("writing_quality"),
        "total_dimension_score": safe_divide(
            sum(
                dimension_scores.get(dim, 0)
                for dim in [
                    "source_grounding",
                    "citation_faithfulness",
                    "reference_coverage",
                    "synthesis_quality",
                    "organization",
                    "specificity",
                    "writing_quality"
                ]
            ),
            7
        ),

        "coherence": quality_analysis.get("coherence"),
        "readability": quality_analysis.get("readability"),
        "conciseness": quality_analysis.get("conciseness"),
        "total_quality_analysis_score": safe_divide(
            sum(
                quality_analysis.get(dim, 0)
                for dim in [
                    "coherence",
                    "readability",
                    "conciseness"
                ]
            ),
            3
        ),

        "strengths_json": json.dumps(summary.get("strengths", []), ensure_ascii=False),
        "weaknesses_json": json.dumps(summary.get("weaknesses", []), ensure_ascii=False),

        # HDM before/after metrics
        "hdm_wrong_claim_count_before": hdm_before.get("wrong_claim_count", 0),
        "hdm_wrong_claim_count_after": hdm_after.get("wrong_claim_count", 0),
        "hdm_total_claims_before": hdm_before.get("total_claims", 0),
        "hdm_total_claims_after": hdm_after.get("total_claims", 0),
        "hdm_hallucination_rate_before": hdm_before.get("hallucination_rate", None),
        "hdm_hallucination_rate_after": hdm_after.get("hallucination_rate", None),
        "hdm_hallucination_rate_delta": hdm_change.get("hallucination_rate_delta"),
        "hdm_hallucination_count_delta": hdm_change.get("hallucination_count_delta"),
        "hdm_hallucination_change_status": hdm_change.get("hallucination_change_status"),
        "hdm_hallucination_reduction_rate": hdm_change.get("hallucination_reduction_rate"),

        "hdm_invalid_provenance_claim_count_before": hdm_before.get("invalid_provenance_claim_count", 0),
        "hdm_invalid_provenance_claim_count_after": hdm_after.get("invalid_provenance_claim_count", 0),
        "hdm_accountable_agent_counts_before_json": json.dumps(
            hdm_before.get("accountable_agent_counts", {}),
            ensure_ascii=False
        ),
        "hdm_accountable_subtask_counts_before_json": json.dumps(
            hdm_before.get("accountable_subtask_counts", {}),
            ensure_ascii=False
        ),

        # Repair outputs
        "repair_performed": repair_result.get("repair_performed", False),
        "repair_report_json": json.dumps(repair_report, ensure_ascii=False),
        "earliest_retriggered_subtask": repair_report.get("earliest_retriggered_subtask", None),
        "retriggered_subtasks_json": json.dumps(
            repair_report.get("retriggered_subtasks", []),
            ensure_ascii=False
        ),
        "downstream_retriggered_subtasks_json": json.dumps(
            repair_report.get("downstream_retriggered_subtasks", []),
            ensure_ascii=False
        ),
        "accountable_stages_json": json.dumps(
            repair_report.get("accountable_stages", []),
            ensure_ascii=False
        ),

        # Full JSON logs
        "planner_output_json": json.dumps(planner_output, ensure_ascii=False),
        "agent_architect_output_json": json.dumps(agent_architect_output, ensure_ascii=False),
        "execution_outputs_json": json.dumps(execution_outputs, ensure_ascii=False),
        "final_expected_milestones_json": json.dumps(final_expected_milestones, ensure_ascii=False),
        "initial_synthesizer_output_json": json.dumps(synthesizer_output, ensure_ascii=False),
        "final_synthesizer_output_json": json.dumps(final_synthesizer_output, ensure_ascii=False),
        "judge_output_json": json.dumps(judge_output, ensure_ascii=False),
        "shared_memory_json": shared_memory.to_json(),
        "hdm_before_json": json.dumps(hdm_before, ensure_ascii=False),
        "hdm_after_json": json.dumps(hdm_after, ensure_ascii=False),

        # Text outputs
        "initial_generated_related_work": initial_task_result,
        "generated_related_work": final_task_result,
        "judge_related_work_title": judge_output.get("related_work_title", "")
    }
# =========================================================
# Whole-dataset runner
# =========================================================
def run_multi_agent_hdm_rw_on_dataset(
    csv_path,
    model=MODEL_NAME,
    generation_temperature=0.0,
    evaluator_temperature=0.0,
    run_id=1,
    max_retries=3,
    results_csv_path="MultiAgent+HDM_rw/MultiAgent+HDM_rw_gemma3/multiagent_hdm_rw_results_gemma3.csv"
):
    df = load_subset_dataframe(csv_path)
    results = []

    os.makedirs(os.path.dirname(results_csv_path) or ".", exist_ok=True)

    dataset_start = time.perf_counter()

    for idx, row in df.iterrows():
        print(f"Processing row {idx + 1}/{len(df)} | row_id={row.get('row_id', idx)}")
        print("Reference bin:", get_row_value(row, "ref_bin"))
        print("Raw ref count:", get_row_value(row, "raw_ref_count"))
        row_success = False
        last_error_msg = ""

        # attempt 0 = original try
        # attempt 1,2,3 = retries
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"Retrying row_index={idx} | retry {attempt}/{max_retries}")

                row_result = evaluate_single_row_multi_agent(
                    row,
                    model=model,
                    generation_temperature=generation_temperature,
                    evaluator_temperature=evaluator_temperature,
                    hdm_temperature=0.0,
                    run_id=run_id,
                    use_hdm_repair=True
                )

                row_result["row_index"] = idx
                row_result["retry_count"] = attempt
                row_result["run_status"] = "successful"

                results.append(row_result)
                row_success = True
                break

            except Exception as e:
                last_error_msg = f"{type(e).__name__}: {str(e)}"[:1000]
                print(f"Attempt {attempt}/{max_retries} failed for row_index={idx}: {last_error_msg}")

        # If all attempts failed, add one blank row with error message
        if not row_success:
            results.append({
            "row_index": idx,
            "row_id": row.get("row_id", ""),
            "split": get_row_value(row, "split"),
            "aid": get_row_value(row, "aid"),
            "mid": get_row_value(row, "mid"),
            "model": model,
            "run_id": run_id,
            "generation_temperature": generation_temperature,
            "evaluator_temperature": evaluator_temperature,
            "raw_ref_count": get_row_value(row, "raw_ref_count"),
            "ref_bin": get_row_value(row, "ref_bin"),
            "query_abstract": get_row_value(row, "abstract"),
            "gold_related_work": get_row_value(row, "related_work"),
            "retry_count": max_retries,
            "run_status": last_error_msg,
            "repair_performed": None,
            "hdm_wrong_claim_count_before": None,
            "hdm_wrong_claim_count_after": None,
            "hdm_hallucination_rate_before": None,
            "hdm_hallucination_rate_after": None,
            "hdm_hallucination_change_status": None,
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

    successful_df = results_df[results_df["run_status"] == "successful"]

    if len(successful_df) > 0:
        print(f"planner time: {round(successful_df['planner_elapsed_sec'].mean(), 3)} sec")
        print(f"architect time: {round(successful_df['architect_elapsed_sec'].mean(), 3)} sec")
        print(f"executor time: {round(successful_df['executor_total_elapsed_sec'].mean(), 3)} sec")
        print(f"synthesis time: {round(successful_df['synthesis_elapsed_sec'].mean(), 3)} sec")
        print(f"judge time: {round(successful_df['judge_elapsed_sec'].mean(), 3)} sec")
        print(f"total time per row: {round(successful_df['total_elapsed_sec'].mean(), 3)} sec")

    return results_df, dataset_elapsed

if __name__ == "__main__":
    CSV_PATH = "related_work_generation_dataset.jsonl"

    GENERATION_TEMPERATURES = [0.0, 0.5, 1.0]
    EVALUATOR_TEMPERATURE = 0.0
    RUNS_PER_TEMPERATURE = 3
    MAX_RETRIES = 3

    BASE_OUTPUT_DIR = "MultiAgent+HDM_rw_arr/MultiAgent+HDM_rw_arr_gemma3"

    all_results = []

    for generation_temp in GENERATION_TEMPERATURES:
        for run_id in range(1, RUNS_PER_TEMPERATURE + 1):

            gen_temp_label = temperature_label(generation_temp)
            eval_temp_label = temperature_label(EVALUATOR_TEMPERATURE)

            results_csv_path = os.path.join(
                BASE_OUTPUT_DIR,
                f"multi_agent_hdm_rw_gemma3_genT{gen_temp_label}_evalT{eval_temp_label}_run{run_id}.csv"
            )

            print("\n" + "=" * 80)
            print(
                "Running MultiAgent+HDM Related Work Generation | "
                f"model=gemma3:64k | "
                f"generation_temperature={generation_temp} | "
                f"evaluator_temperature={EVALUATOR_TEMPERATURE} | "
                f"run_id={run_id} | "
                f"max_retries={MAX_RETRIES}"
            )
            print("=" * 80)

            results_df, dataset_elapsed = run_multi_agent_hdm_rw_on_dataset(
                                        csv_path=CSV_PATH,
                                        model=MODEL_NAME,
                                        generation_temperature=generation_temp,
                                        evaluator_temperature=EVALUATOR_TEMPERATURE,
                                        run_id=run_id,
                                        max_retries=MAX_RETRIES,
                                        results_csv_path=results_csv_path
                                    )

            all_results.append(results_df)

    if not all_results:
        raise RuntimeError("No result dataframes were created. Check earlier errors before concatenating.")

    combined_results_df = (
        pd.concat(all_results, ignore_index=True)
        if all_results
        else pd.DataFrame()
    )

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    combined_results_path = os.path.join(
        BASE_OUTPUT_DIR,
        "multi_agent_hdm_rw_gemma3_all_temperatures_combined_with_rerun.csv"
    )

    combined_results_df.to_csv(combined_results_path, index=False)

    print("\nTemperature experiment finished.")
    print(f"Combined results saved to: {combined_results_path}")
    print(f"Total rows: {len(combined_results_df)}")
    print(f"Successful rows: {(combined_results_df['run_status'] == 'successful').sum()}")
    print(f"Errored rows after retries: {(combined_results_df['run_status'] != 'successful').sum()}")
    
