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
        "future_direction_title": {"type": "string"},
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
                "source_relevance": {"type": "number"},
                "novelty": {"type": "number"},
                "feasibility": {"type": "number"},
                "specificity": {"type": "number"},
                "structure": {"type": "number"},
                "metadata_grounding": {"type": "number"},
                "writing_quality": {"type": "number"}
            },
            "required": [
                "source_relevance",
                "novelty",
                "feasibility",
                "specificity",
                "structure",
                "metadata_grounding",
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
        "future_direction_title",
        "milestone_judgment",
        "dimension_scores",
        "quality_analysis",
        "summary",
        "confidence"
    ]
}

# =========================================================
# System messages (CORRECTED to match schemas)
# =========================================================
planner_system_msg = """
You are a Task Planner for a multi-agent future research direction generation system.

Your job is to take the main task and produce:
- A list of actionable subtasks
- Final expected milestones for the overall future research direction
- Confidence score for the plan (0 to 1)

The task is GrantFuture: generate a structured future research direction inspired by a funded NSF project record.

Instructions:
1. Analyze the funded project metadata and abstract.
2. Break the task into actionable, clear, and logically ordered subtasks.
3. Do not simply plan to summarize the original grant.
4. The final output should extend the original funded project by identifying a research gap or open problem, proposing a new direction, including potential objectives, possible methodology, expected contribution, and broader impact.
5. Do not skip any important subtask. Avoid unnecessary subtasks.
6. Do NOT define milestones for individual subtasks.
7. For each subtask, provide:
   - subtask_id
   - subtask_description
8. After listing all subtasks, define the final expected milestones for the overall main task.
9. Mandatory milestones must represent vital requirements for source relevance, novelty, feasibility, specificity, metadata grounding, and writing quality.
10. Optional milestones should improve depth, nuance, clarity, or usefulness.
11. There must be at least ONE MANDATORY milestone.
12. Provide a confidence score from 0 to 1.

Return ONLY valid JSON matching the required schema.

### EXAMPLE STRUCTURE
{
  "Subtasks": [
    {
      "subtask_id": "T1",
      "subtask_description": "Identify the original funded project's research area, main problem, approach, and expected contribution from the metadata and abstract."
    },
    {
      "subtask_id": "T2",
      "subtask_description": "Identify a plausible research gap or open problem that extends the funded project without merely restating it."
    },
    {
      "subtask_id": "T3",
      "subtask_description": "Propose a future research direction with potential objectives and a possible methodology."
    },
    {
      "subtask_id": "T4",
      "subtask_description": "Explain the expected contribution and broader impact of the proposed future direction."
    }
  ],
  "FinalExpectedMilestones": [
    {
      "milestone_id": "M1",
      "priority": "MANDATORY",
      "description": "The final output must clearly connect the future direction to the original funded project."
    },
    {
      "milestone_id": "M2",
      "priority": "MANDATORY",
      "description": "The final output must propose a novel future direction rather than simply summarizing the original grant."
    },
    {
      "milestone_id": "M3",
      "priority": "MANDATORY",
      "description": "The final output must include a plausible research gap, potential objectives, possible methodology, and expected contribution."
    },
    {
      "milestone_id": "M4",
      "priority": "OPTIONAL",
      "description": "The final output may include a broader impact discussion that is reasonable and grounded in the project context."
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
      "actor_type": "Source Analysis Agent",
      "skills": ["grant abstract interpretation", "metadata grounding", "source anchor extraction"],
      "rationale": "This agent analyzes the original funded project and identifies the source-grounded context."
    },
    "A2": {
      "actor_type": "Research Ideation Agent",
      "skills": ["research gap identification", "future direction generation", "methodological planning"],
      "rationale": "This agent develops a plausible future research direction that extends the original project."
    },
    "A3": {
      "actor_type": "Impact and Synthesis Agent",
      "skills": ["broader impact analysis", "research communication", "structured synthesis"],
      "rationale": "This agent helps connect the proposed direction to expected contribution, broader impact, and final structure."
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
  "task_result": "The funded project focuses on machine learning methods for scientific data analysis. A reasonable future extension is to evaluate how these methods generalize across more diverse scientific workflows while preserving the original project's focus on scalable data-driven discovery.",
  "confidence": 0.9
}}
"""


synthesizer_system_msg = """
You are the Final Writer in a multi-agent future research direction generation system.

You will receive:
1. Funded NSF project metadata
2. The original project abstract
3. Final expected milestones
4. Executor outputs from specialized agents

Your task is to synthesize the executor outputs into one coherent 400-600 word structured future research direction.

The final output should not merely summarize the original funded project. It should propose a plausible future research direction that extends the project.

The final output should include clear sections such as:
1. Future Direction Title
2. Connection to the Funded Project
3. Research Gap or Open Problem
4. Proposed Future Direction
5. Potential Objectives
6. Possible Methodology
7. Expected Contribution and Broader Impact

Writing rules:
- Stay grounded in the funded project metadata, abstract, executor outputs, and source context.
- Do not claim that the future direction is already funded.
- Do not invent completed results, publications, collaborators, or specific datasets unless clearly framed as possible future examples.
- Do not contradict the original project metadata.
- Write clearly, professionally, and concisely.
- Place the complete final future research direction in the "task_result" field.
- Provide a confidence score from 0 to 1.

Output MUST be ONLY valid JSON.

### EXAMPLE:
{
  "task_result": "Future Direction Title: Human-Centered Evaluation of Scalable Machine Learning Tools for Scientific Discovery\\n\\nConnection to the Funded Project: This future direction builds on the original NSF-funded project by extending its focus on machine learning methods for scientific data analysis toward real-world research workflows.\\n\\nResearch Gap or Open Problem: While the original project develops computational methods, a natural next step is to understand how these methods perform across diverse scientific users, datasets, and decision-making contexts.\\n\\nProposed Future Direction: A future project could investigate how researchers interact with these machine learning tools in practical scientific discovery settings.\\n\\nPotential Objectives: The work could evaluate usability, robustness, interpretability, and cross-domain generalization.\\n\\nPossible Methodology: The study could combine benchmark evaluation, user studies, and case studies across multiple scientific domains.\\n\\nExpected Contribution and Broader Impact: This direction could make machine learning tools more reliable, interpretable, and useful for broader scientific communities.",
  "confidence": 0.9
}
"""

judge_system_msg = """
You are an expert evaluator for future research direction generation from funded NSF project records.

You will be given:
1. Funded NSF project metadata
2. The original project abstract
3. A generated future research direction
4. A set of final expected milestones
5. A reference evaluation ground truth

Your task is to judge the generated future research direction.

You must evaluate:
1. Milestone achievement
2. Source relevance
3. Novelty
4. Feasibility
5. Specificity
6. Structure
7. Metadata grounding
8. Writing quality
9. Quality subdimensions:
   - coherence
   - readability
   - conciseness

Dimension definitions:
- source_relevance: Does the future direction clearly build on the original grant?
- novelty: Is the idea a meaningful extension rather than a restatement?
- feasibility: Is the proposed direction plausible given the original topic?
- specificity: Are the objectives and method concrete enough?
- structure: Does the output follow the required sections?
- metadata_grounding: Does it preserve the original grant context correctly?
- writing_quality: Is it coherent, readable, concise, and professional?

Evaluation guidance:
- Do not judge based on wording overlap.
- Judge whether the future direction is meaningfully connected to the original grant title, abstract, metadata, and ground truth reference.
- The output should propose a new future direction, not merely summarize the original funded project.
- The proposed direction should be plausible, feasible, and specific enough.
- The output should include a research gap or open problem, proposed future direction, potential objectives, possible methodology, expected contribution, and broader impact.
- Missing optional details should not be penalized heavily.
- Missing mandatory milestones or a weak connection to the original project should reduce the score.
- Unsupported claims, invented completed results, invented collaborators, invented publications, or claims that the future direction is already funded should reduce the score significantly.
- Allow reasonable creativity, but the idea must remain grounded in the source record.

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
- Score each dimension from 0.0 to 1.0
- 1.0 = fully satisfied
- 0.5 = partially satisfied
- 0.0 = not satisfied
- Use intermediate values when appropriate

Return ONLY valid JSON matching the required schema.
Do not include markdown or extra text outside the JSON object.

### EXAMPLE FORMAT
{
  "future_direction_title": "Human-Centered Evaluation of Scalable Machine Learning Tools for Scientific Discovery",
  "milestone_judgment": {
    "milestone_scores": [
      {
        "milestone_id": "M1",
        "priority": "MANDATORY",
        "description": "The final output must clearly connect the future direction to the original funded project.",
        "achieved": true,
        "score_1_to_5": 5,
        "reason": "The future direction explicitly builds on the original project topic and uses the source grant context appropriately."
      },
      {
        "milestone_id": "M2",
        "priority": "MANDATORY",
        "description": "The final output must propose a novel future direction rather than simply summarizing the original grant.",
        "achieved": true,
        "score_1_to_5": 4,
        "reason": "The output proposes a meaningful extension, although some objectives could be more specific."
      }
    ]
  },
  "dimension_scores": {
    "source_relevance": 0.95,
    "novelty": 0.85,
    "feasibility": 0.9,
    "specificity": 0.8,
    "structure": 1.0,
    "metadata_grounding": 0.95,
    "writing_quality": 0.9
  },
  "quality_analysis": {
    "coherence": 0.95,
    "readability": 0.9,
    "conciseness": 0.85
  },
  "summary": {
    "strengths": [
      "The proposed direction is clearly connected to the original funded project.",
      "The output includes a research gap, objectives, methodology, and broader impact."
    ],
    "weaknesses": [
      "The methodology could include more concrete evaluation details."
    ]
  },
  "confidence": 0.9
}
"""
# =========================================================
# Prompt builders
# =========================================================
def build_initial_task(row):
    return f"""
Generate a high-quality structured future research direction using a multi-agent workflow.

Funded Project Metadata:
- CS Field: {get_row_value(row, 'cs_field')}
- Search Keyword: {get_row_value(row, 'search_keyword')}
- Grant ID: {get_row_value(row, 'grant_id')}
- Project Title: {get_row_value(row, 'title')}
- Principal Investigator(s): {get_row_value(row, 'pi_names')}
- Institution: {get_row_value(row, 'institution')}
- Funder: {get_row_value(row, 'funder')}
- Agency: {get_row_value(row, 'agency')}
- Directorate: {get_row_value(row, 'directorate')}
- Division: {get_row_value(row, 'division')}
- Program: {get_row_value(row, 'program')}
- Award Year: {get_row_value(row, 'award_year')}
- Award Amount: {get_row_value(row, 'award_amount')}
- Project URL: {get_row_value(row, 'project_url')}

Original Project Abstract:
{get_row_value(row, 'abstract')}

Overall objective:
Produce a structured future research direction in 400-600 words. The output should extend the original project, identify a plausible research gap or open problem, propose a new direction, include potential objectives, describe a possible methodology, and explain expected contribution and broader impact.
"""

def build_planner_user_msg(row):
    return f"""
Main Task: Generate a structured future research direction inspired by the following funded NSF project record.

Funded Project Metadata:
- CS Field: {get_row_value(row, 'cs_field')}
- Search Keyword: {get_row_value(row, 'search_keyword')}
- Grant ID: {get_row_value(row, 'grant_id')}
- Project Title: {get_row_value(row, 'title')}
- Principal Investigator(s): {get_row_value(row, 'pi_names')}
- Institution: {get_row_value(row, 'institution')}
- Funder: {get_row_value(row, 'funder')}
- Program: {get_row_value(row, 'program')}
- Award Year: {get_row_value(row, 'award_year')}

Original Project Abstract:
{get_row_value(row, 'abstract')}

Produce subtasks (T1, T2...), final milestones (M1, M2...), and confidence about the plan.
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
Funded Project Metadata:
- CS Field: {get_row_value(row, 'cs_field')}
- Grant ID: {get_row_value(row, 'grant_id')}
- Project Title: {get_row_value(row, 'title')}
- Principal Investigator(s): {get_row_value(row, 'pi_names')}
- Institution: {get_row_value(row, 'institution')}
- Funder: {get_row_value(row, 'funder')}
- Program: {get_row_value(row, 'program')}
- Award Year: {get_row_value(row, 'award_year')}

Original Project Abstract:
{get_row_value(row, 'abstract')}

Assigned subtask:
{subtask['subtask_description']}

Subtask ID:
{subtask['subtask_id']}

Context from previously completed subtasks:
{json.dumps(context_dict, indent=2, ensure_ascii=False)}

Complete only this subtask now. Stay grounded in the funded project record.
"""

def build_synthesizer_user_msg(row, final_expected_milestones, execution_outputs):
    simplified_outputs = {
        subtask_id: output["executor_output"]["task_result"]
        for subtask_id, output in execution_outputs.items()
    }

    return f"""
Write the final structured future research direction using the materials below.

Funded Project Metadata:
- CS Field: {get_row_value(row, 'cs_field')}
- Grant ID: {get_row_value(row, 'grant_id')}
- Project Title: {get_row_value(row, 'title')}
- Principal Investigator(s): {get_row_value(row, 'pi_names')}
- Institution: {get_row_value(row, 'institution')}
- Funder: {get_row_value(row, 'funder')}
- Program: {get_row_value(row, 'program')}
- Award Year: {get_row_value(row, 'award_year')}

Original Project Abstract:
{get_row_value(row, 'abstract')}

Final Expected Milestones:
{json.dumps(final_expected_milestones, indent=2, ensure_ascii=False)}

Executor Outputs:
{json.dumps(simplified_outputs, indent=2, ensure_ascii=False)}

Write a 400-600 word structured future research direction with sections such as:
1. Future Direction Title
2. Connection to the Funded Project
3. Research Gap or Open Problem
4. Proposed Future Direction
5. Potential Objectives
6. Possible Methodology
7. Expected Contribution and Broader Impact

Return ONLY valid JSON with:
- task_result
- confidence
"""

def build_judge_user_msg(row, final_expected_milestones, task_result):
    reference_record = {
        "cs_field": get_row_value(row, "cs_field"),
        "search_keyword": get_row_value(row, "search_keyword"),
        "grant_id": get_row_value(row, "grant_id"),
        "title": get_row_value(row, "title"),
        "abstract": get_row_value(row, "abstract"),
        "pi_names": get_row_value(row, "pi_names"),
        "institution": get_row_value(row, "institution"),
        "funder": get_row_value(row, "funder"),
        "agency": get_row_value(row, "agency"),
        "directorate": get_row_value(row, "directorate"),
        "division": get_row_value(row, "division"),
        "program": get_row_value(row, "program"),
        "award_year": get_row_value(row, "award_year"),
        "award_amount": get_row_value(row, "award_amount"),
        "project_url": get_row_value(row, "project_url")
    }

    return f"""
Evaluate the following future research direction.

Reference Funded Project Record:
{json.dumps(reference_record, indent=2, ensure_ascii=False)}

Final Expected Milestones:
{json.dumps(final_expected_milestones, indent=2, ensure_ascii=False)}

Generated Future Research Direction:
{task_result}

Reference Evaluation Ground Truth:
{get_row_value(row, 'ground_truth_json', '{}')}

Return ONLY valid JSON with:
- future_direction_title
- milestone_judgment
- dimension_scores
- quality_analysis
- summary
- confidence
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
# Utility functions
# =========================================================
def get_row_value(row, col, default=""):
    value = row.get(col, default)
    if pd.isna(value):
        return default
    return str(value)

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
    run_id=1
):
    _AUTOGEN_AGENT_CACHE.clear()
    row_start = time.perf_counter()
    shared_memory = SharedMemory(main_task=get_row_value(row, "title"))
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

    #print("Synthesis elapsed: ", synthesis_elapsed)
    task_result = synthesizer_output["task_result"]
    mandatory_total = sum(1 for m in final_expected_milestones if m.get("priority") == "MANDATORY")
    optional_total = sum(1 for m in final_expected_milestones if m.get("priority") == "OPTIONAL")

    # Judge
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
        "planner_temperature": generation_temperature,
        "architect_temperature": generation_temperature,
        "executor_temperature": generation_temperature,
        "synthesizer_temperature": generation_temperature,
        "judge_temperature": evaluator_temperature,
        "cs_field": get_row_value(row, "cs_field"),
        "search_keyword": get_row_value(row, "search_keyword"),
        "grant_id": get_row_value(row, "grant_id"),
        "title": get_row_value(row, "title"),
        "pi_names": get_row_value(row, "pi_names"),
        "institution": get_row_value(row, "institution"),
        "funder": get_row_value(row, "funder"),
        "agency": get_row_value(row, "agency"),
        "directorate": get_row_value(row, "directorate"),
        "division": get_row_value(row, "division"),
        "program": get_row_value(row, "program"),
        "award_year": get_row_value(row, "award_year"),
        "award_amount": get_row_value(row, "award_amount"),
        "project_url": get_row_value(row, "project_url"),
        "planner_elapsed_sec": round(planner_elapsed, 3),
        "architect_elapsed_sec": round(architect_elapsed, 3),
        "executor_total_elapsed_sec": round(executor_total_elapsed, 3),
        "synthesis_elapsed_sec": round(synthesis_elapsed, 3),
        "judge_elapsed_sec": round(judge_elapsed, 3),
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

        "source_relevance": dimension_scores.get("source_relevance"),
        "novelty": dimension_scores.get("novelty"),
        "feasibility": dimension_scores.get("feasibility"),
        "specificity": dimension_scores.get("specificity"),
        "structure": dimension_scores.get("structure"),
        "metadata_grounding": dimension_scores.get("metadata_grounding"),
        "writing_quality": dimension_scores.get("writing_quality"),
        "total_dimension_score": (
            sum(dimension_scores.get(dim, 0) for dim in [
                "source_relevance",
                "novelty",
                "feasibility",
                "specificity",
                "structure",
                "metadata_grounding",
                "writing_quality"
            ]) / 7
        ),

        "coherence": quality_analysis.get("coherence"),
        "readability": quality_analysis.get("readability"),
        "conciseness": quality_analysis.get("conciseness"),
        "total_quality_analysis_score": (
        sum(quality_analysis.get(dim, 0) for dim in [
            "coherence",
            "readability",
            "conciseness"
        ]) / 3
    ),

        "strengths_json": json.dumps(summary.get("strengths", []), ensure_ascii=False),
        "weaknesses_json": json.dumps(summary.get("weaknesses", []), ensure_ascii=False),

        "planner_output_json": json.dumps(planner_output, ensure_ascii=False),
        "agent_architect_output_json": json.dumps(agent_architect_output, ensure_ascii=False),
        "execution_outputs_json": json.dumps(execution_outputs, ensure_ascii=False),
        "final_expected_milestones_json": json.dumps(final_expected_milestones, ensure_ascii=False),
        "generated_future_direction": task_result,
        "synthesizer_output_json": json.dumps(synthesizer_output, ensure_ascii=False),
        "judge_output_json": json.dumps(judge_output, ensure_ascii=False),
        "shared_memory_json": shared_memory.to_json(),
        "judge_future_direction_title": judge_output.get("future_direction_title", ""),
    }

# =========================================================
# Whole-dataset runner
# =========================================================
def run_multi_agent_baseline_on_dataset(
    csv_path,
    model=MODEL_NAME,
    generation_temperature=0.0,
    evaluator_temperature=0.0,
    run_id=1,
    max_retries=3,
    results_csv_path="Baseline2_MultiAgent+Judge/Baseline2_gemma3/multi_agent_results_gemma3.csv"
):
    df = pd.read_csv(csv_path).fillna("")
    results = []

    os.makedirs(os.path.dirname(results_csv_path) or ".", exist_ok=True)

    dataset_start = time.perf_counter()

    for idx, row in df.iterrows():
        print(f"Processing row {idx + 1}/{len(df)} | row_id={row.get('row_id', idx)}")
        print("Ground Truth JSON:")
        print(get_row_value(row, "ground_truth_json", "{}"))
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
                    run_id=run_id
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
                "model": model,
                "run_id": run_id,
                "generation_temperature": generation_temperature,
                "evaluator_temperature": evaluator_temperature,
                "planner_temperature": generation_temperature,
                "architect_temperature": generation_temperature,
                "executor_temperature": generation_temperature,
                "synthesizer_temperature": generation_temperature,
                "judge_temperature": evaluator_temperature,
                "cs_field": get_row_value(row, "cs_field"),
                "search_keyword": get_row_value(row, "search_keyword"),
                "grant_id": get_row_value(row, "grant_id"),
                "title": get_row_value(row, "title"),
                "pi_names": get_row_value(row, "pi_names"),
                "institution": get_row_value(row, "institution"),
                "funder": get_row_value(row, "funder"),
                "program": get_row_value(row, "program"),
                "award_year": get_row_value(row, "award_year"),
                "award_amount": get_row_value(row, "award_amount"),
                "project_url": get_row_value(row, "project_url"),
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
    CSV_PATH = "project_grounded_research_ideation_dataset.csv"

    GENERATION_TEMPERATURES = [0.0, 0.5, 1.0]
    EVALUATOR_TEMPERATURE = 0.0
    RUNS_PER_TEMPERATURE = 3
    MAX_RETRIES = 3

    BASE_OUTPUT_DIR = "Baseline2_MultiAgent+Judge/Baseline2_MultiAgent+Judge_gemma3"

    all_results = []

    for generation_temp in GENERATION_TEMPERATURES:
        for run_id in range(1, RUNS_PER_TEMPERATURE + 1):

            gen_temp_label = temperature_label(generation_temp)
            eval_temp_label = temperature_label(EVALUATOR_TEMPERATURE)

            results_csv_path = os.path.join(
                BASE_OUTPUT_DIR,
                f"multi_agent_grant_gemma3_genT{gen_temp_label}_evalT{eval_temp_label}_run{run_id}.csv"
            )

            print("\n" + "=" * 80)
            print(
                "Running Baseline2 Multi-Agent + Judge | "
                f"model=gemma3:64k | "
                f"generation_temperature={generation_temp} | "
                f"evaluator_temperature={EVALUATOR_TEMPERATURE} | "
                f"run_id={run_id} | "
                f"max_retries={MAX_RETRIES}"
            )
            print("=" * 80)

            results_df, dataset_elapsed = run_multi_agent_baseline_on_dataset(
                csv_path=CSV_PATH,
                model="gemma3:64k",
                generation_temperature=generation_temp,
                evaluator_temperature=EVALUATOR_TEMPERATURE,
                run_id=run_id,
                max_retries=MAX_RETRIES,
                results_csv_path=results_csv_path
            )

            all_results.append(results_df)

    combined_results_df = (
        pd.concat(all_results, ignore_index=True)
        if all_results
        else pd.DataFrame()
    )

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    combined_results_path = os.path.join(
        BASE_OUTPUT_DIR,
        "multi_agent_grant_gemma3_all_temperatures_combined_with_rerun.csv"
    )

    combined_results_df.to_csv(combined_results_path, index=False)

    print("\nTemperature experiment finished.")
    print(f"Combined results saved to: {combined_results_path}")
    print(f"Total rows: {len(combined_results_df)}")
    print(f"Successful rows: {(combined_results_df['run_status'] == 'successful').sum()}")
    print(f"Errored rows after retries: {(combined_results_df['run_status'] != 'successful').sum()}")
    
