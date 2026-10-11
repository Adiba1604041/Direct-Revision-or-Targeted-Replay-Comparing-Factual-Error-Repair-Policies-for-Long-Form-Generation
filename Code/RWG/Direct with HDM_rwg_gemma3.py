import json
import time
import traceback
import pandas as pd
import requests
from jsonschema import validate
import autogen
import re
import os
import ast

# -------------------------
# HDM Claim Extraction Schema
# -------------------------

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


# -------------------------
# HDM Claim Verification Schema
# -------------------------

claim_verification_schema = {
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
                    "reason": {"type": "string"}
                },
                "required": [
                    "claim_id",
                    "claim",
                    "label",
                    "evidence",
                    "reason"
                ]
            }
        }
    },
    "required": ["claim_verifications"]
}


# -------------------------
# Writer schema
# -------------------------
baseline1_writer_schema = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "related_work_title": {"type": "string"},
        "final_expected_milestones": {
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
                    "description": {"type": "string"}
                },
                "required": ["milestone_id", "priority", "description"]
            }
        },
        "task_result": {"type": "string"}
    },
    "required": [
        "related_work_title",
        "final_expected_milestones",
        "task_result"
    ]
}

baseline1_writer_system_msg = """
You are an expert scientific writing assistant. Your task is to generate a related-work section for a query scientific paper using the query paper abstract and the abstracts of cited reference papers.

The input contains:
1. A query paper abstract
2. Multiple cited reference abstracts
3. Citation identifiers such as @cite_0, @cite_1, etc.

Your job is to do two things:
1. Define the final expected milestones for a high-quality related-work generation task.
2. Write the related-work section.

Milestone design rules:
- Milestones must describe what the final related-work section should accomplish.
- Each milestone must be tagged as either MANDATORY or OPTIONAL.
- MANDATORY milestones are essential for correctness and adequate completion.
- OPTIONAL milestones improve completeness, depth, nuance, or style.
- There must be at least one MANDATORY milestone.
- Milestones should be specific, observable, and useful for judging the final output.
- Focus on source grounding, accurate synthesis, coverage of cited works, correct attribution, coherence, and scholarly writing style.

Writing rules:
- Generate one coherent related-work section for the query paper.
- Use only the query abstract and cited reference abstracts as evidence.
- Synthesize the cited works instead of listing them mechanically.
- Do not invent paper details, datasets, methods, results, or claims that are not supported by the provided abstracts.
- Preserve citation identifiers when discussing a cited work, for example @cite_0 or @cite_3.
- The paragraph should sound like scholarly related work, not a generic summary.
- Do not include markdown, bullet points, section headers, or explanations outside the JSON object.

Revision behavior:
- If you later receive HDM feedback, you are revising your own previous related-work section.
- Revise only the claims identified as unsupported or contradicted.
- Preserve supported content.
- Preserve valid citation identifiers such as @cite_0, @cite_1, etc.
- Do not restart from scratch.
- Do not add new unsupported paper details, datasets, methods, results, comparisons, or citation attributions.
- Do not change final_expected_milestones during revision.

Return ONLY valid JSON matching the required schema.
Use related_work_title as a short descriptive title for the generated related-work paragraph.
Use task_result for the generated related-work paragraph.
"""

# -------------------------
# Judge schema
# -------------------------
judge_schema_baseline1 = {
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
        }
    },
    "required": [
        "related_work_title",
        "milestone_judgment",
        "dimension_scores",
        "quality_analysis",
        "summary"
    ]
}

judge_system_msg_baseline1 = """
You are an expert evaluator for scientific related-work generation.

You will be given:
1. A query paper abstract
2. The cited reference abstracts
3. A generated related-work section for the query paper
4. A set of final expected milestones
5. The original gold related-work section from Multi-XScience

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
- reference_coverage: Does the paragraph cover the important cited works or themes from the references?
- synthesis_quality: Does it synthesize relationships among works instead of listing papers one by one?
- organization: Is the paragraph logically structured like scholarly related work?
- specificity: Does it include concrete scientific concepts rather than generic statements?
- writing_quality: Is it fluent, concise, academic, and readable?

Evaluation guidance:
- Do not require exact wording overlap with the gold related-work paragraph.
- Use the gold related-work paragraph as a reference for expected content, but judge semantic quality and grounding.
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
"""

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


claim_verification_system_msg = """
You are hdm_claim_verifier_agent, a source-grounded claim verification agent for scientific related-work generation.

Your task is to verify extracted claims from a generated related-work section.

You will receive:
1. Extracted claims
2. The query paper abstract
3. The cited reference abstracts with citation identifiers such as @cite_0, @cite_1, etc.
4. The original gold related-work paragraph from Multi-XScience as auxiliary reference context

Verification labels:
- supported: the claim is directly supported by the query abstract or cited reference abstracts.
- unsupported: the claim is not supported by the query abstract or cited reference abstracts.
- contradicted: the claim conflicts with the query abstract or cited reference abstracts.

Rules:
- Use the query abstract and cited reference abstracts as the primary evidence.
- Use the gold related-work paragraph only as auxiliary context, not as the sole evidence for support.
- Do not use your internal knowledge to verify claims.
- Be strict about methods, datasets, tasks, findings, comparisons, numbers, named entities, and citation attribution.
- If a claim is attached to a citation identifier, verify that the cited reference abstract actually supports that claim.
- Do not reward a claim just because it sounds plausible.
- Provide clear evidence when a claim is supported.
- For unsupported or contradicted claims, explain why the claim is not supported or why it conflicts with the provided evidence.
- Return ONLY valid JSON matching the schema.
Do not include markdown or extra text outside the JSON object.
"""


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
    Supports official nested format:
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
# -------------------------
# Prompt builders
# -------------------------
def build_writer_user_msg(row):
    reference_abstracts_text = format_reference_abstracts(row)

    return f"""
Generate a scholarly related-work paragraph for the following query paper.

Query Paper Abstract:
{get_row_value(row, 'abstract')}

Cited Reference Abstracts:
{reference_abstracts_text}

Task:
Generate a related-work section that synthesizes the cited reference abstracts in relation to the query paper. The section should be grounded in the provided abstracts, use citation identifiers where appropriate, and avoid unsupported claims.

Return ONLY valid JSON with:
- related_work_title: a short descriptive title
- final_expected_milestones
- task_result: the generated related-work section
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
Evaluate the following generated related-work paragraph.

Reference Multi-XScience Record:
{json.dumps(reference_record, indent=2, ensure_ascii=False)}

Final Expected Milestones:
{json.dumps(final_expected_milestones, indent=2, ensure_ascii=False)}

Generated Related Work:
{task_result}

Gold Related Work Paragraph:
{get_row_value(row, 'related_work')}

Return ONLY valid JSON with:
- related_work_title
- milestone_judgment
- dimension_scores
- quality_analysis
- summary
"""

def build_claim_extraction_user_msg(task_result):
    return f"""
Generated Related-Work Section:
{task_result}

Extract atomic factual claims from this related-work section.

Return ONLY valid JSON with:
- claims
"""


def build_claim_verification_user_msg(row, extracted_claims):
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

Verify each extracted claim against the query paper abstract and cited reference abstracts.

Important:
- Use the query abstract and cited reference abstracts as the primary evidence.
- Use the gold related-work paragraph only as auxiliary reference context.
- If a generated claim cites @cite_i, verify that the corresponding cited reference abstract supports that claim.
- Mark unsupported claims, invented paper details, invented methods, invented datasets, invented results, and incorrect citation attributions as unsupported or contradicted.

Return ONLY valid JSON with:
- claim_verifications
"""


def build_hdm_feedback_msg(wrong_claims):
    if not wrong_claims:
        return "There is no unsupported or contradicted claim in your related-work section. Great job! No revision needed."

    feedback_lines = [
        "HDM found unsupported or contradicted claims in your previous related-work section.",
        "Revise only the parts related to the following wrong claims.",
        ""
    ]

    for i, claim in enumerate(wrong_claims, start=1):
        feedback_lines.append(f"{i}. Claim: {claim['claim']}")
        feedback_lines.append(f"   Label: {claim['label']}")
        feedback_lines.append(f"   Reason: {claim['reason']}")
        feedback_lines.append("")

    feedback_lines.append(
        "Do not add new unsupported scientific claims, paper details, datasets, methods, results, comparisons, or citation attributions. Preserve all supported content from the original related-work section."
    )

    return "\n".join(feedback_lines)


def build_writer_revision_user_msg(row, previous_writer_output, hdm_feedback_msg):
    reference_abstracts_text = format_reference_abstracts(row)

    return f"""
You are writer_agent. You previously wrote the following related-work section.

Previous Writer Output:
{json.dumps(previous_writer_output, indent=2, ensure_ascii=False)}

HDM Feedback:
{hdm_feedback_msg}

Query Paper Abstract:
{get_row_value(row, 'abstract')}

Cited Reference Abstracts:
{reference_abstracts_text}

Gold Related Work Paragraph:
{get_row_value(row, 'related_work')}

Revise the related-work section using the HDM feedback.

Important:
- Revise only the unsupported or contradicted claims identified by HDM.
- For each unsupported or contradicted claim, use the HDM feedback and the provided abstracts to decide whether the claim can be safely rewritten.
- If the provided abstracts support a corrected version, rewrite the claim so that it becomes supported.
- If the provided abstracts do not support a safe correction, remove the claim.
- Preserve supported content.
- Preserve valid citation identifiers.
- Do not restart from scratch.
- Do not change final_expected_milestones. Keep them exactly as in the previous writer output.
- Do not introduce new unsupported paper details, methods, datasets, results, comparisons, or citation attributions.
- Maintain a scholarly related-work style.

Return ONLY valid JSON with:
- related_work_title
- final_expected_milestones
- task_result
"""

def safe_divide(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator

# -------------------------
# AutoGen helpers
# -------------------------

def extract_json_from_text(text):
    """
    AutoGen/local models may return markdown fences or thinking text.
    This tries to extract the JSON object safely.
    """
    text = text.strip()

    # Remove Qwen/DeepSeek-style thinking if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Remove markdown fences if present
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


def build_llm_config(model="gemma3:64k", timeout=900, temperature=0.0):
    """
    AutoGen config for Ollama OpenAI-compatible endpoint.
    Temperature controls generation stochasticity.
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


def create_row_agents(
    model="gemma3:64k",
    timeout=900,
    writer_temperature=0.0,
    judge_temperature=0.0
):
    """
    Creates separate AutoGen agents for one dataset row.
    This prevents conversation history from one row leaking into another row.

    The same writer_agent is used for:
    1. initial related-work generation
    2. revision after HDM feedback

    Writer temperature is varied.
    HDM and judge temperatures stay fixed/deterministic.
    """

    writer_llm_config = build_llm_config(
        model=model,
        timeout=timeout,
        temperature=writer_temperature
    )

    hdm_llm_config = build_llm_config(
        model=model,
        timeout=timeout,
        temperature=0.0
    )

    judge_llm_config = build_llm_config(
        model=model,
        timeout=timeout,
        temperature=judge_temperature
    )

    writer_agent = autogen.AssistantAgent(
        name="writer_agent",
        system_message=baseline1_writer_system_msg,
        llm_config=writer_llm_config
    )

    claim_extractor_agent = autogen.AssistantAgent(
        name="hdm_claim_extractor_agent",
        system_message=claim_extraction_system_msg,
        llm_config=hdm_llm_config
    )

    claim_verifier_agent = autogen.AssistantAgent(
        name="hdm_claim_verifier_agent",
        system_message=claim_verification_system_msg,
        llm_config=hdm_llm_config
    )

    judge_agent = autogen.AssistantAgent(
        name="judge_agent",
        system_message=judge_system_msg_baseline1,
        llm_config=judge_llm_config
    )

    proxies = {
        "writer": autogen.UserProxyAgent(
            name="writer_proxy",
            human_input_mode="NEVER",
            code_execution_config=False
        ),
        "claim_extractor": autogen.UserProxyAgent(
            name="claim_extractor_proxy",
            human_input_mode="NEVER",
            code_execution_config=False
        ),
        "claim_verifier": autogen.UserProxyAgent(
            name="claim_verifier_proxy",
            human_input_mode="NEVER",
            code_execution_config=False
        ),
        "judge": autogen.UserProxyAgent(
            name="judge_proxy",
            human_input_mode="NEVER",
            code_execution_config=False
        )
    }

    agents = {
        "writer": writer_agent,
        "claim_extractor": claim_extractor_agent,
        "claim_verifier": claim_verifier_agent,
        "judge": judge_agent
    }

    return agents, proxies


def get_autogen_response(agent, proxy, chat_result=None):
    """
    Gets the most recent assistant message.
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


def call_autogen_agent_with_schema(
    agent,
    proxy,
    user_msg,
    schema,
    clear_history=True,
    max_retries=1
):
    """
    Sends one task to one AutoGen agent and validates the returned JSON.

    clear_history=True:
        use this for independent calls like claim extraction, claim verification, and judge.

    clear_history=False:
        use this when sending HDM feedback back to the same writer_agent.
    """
    grounded_user_msg = (
        user_msg
        + "\n\nThe response must conform to this JSON schema exactly:\n"
        + json.dumps(schema, indent=2, ensure_ascii=False)
        + "\n\nReturn ONLY the JSON object. Do not include markdown, explanation, or reasoning."
    )

    last_error = None

    for attempt in range(max_retries + 1):
        if attempt == 0:
            message = grounded_user_msg
        else:
            message = (
                grounded_user_msg
                + "\n\nYour previous response failed JSON validation/parsing."
                + f"\nError: {last_error}"
                + "\nReturn the corrected JSON object only."
            )

        try:
            try:
                chat_result = proxy.initiate_chat(
                    agent,
                    message=message,
                    max_turns=1,
                    silent=True,
                    clear_history=clear_history if attempt == 0 else False
                )
            except TypeError:
                # For older AutoGen versions that may not support clear_history
                chat_result = proxy.initiate_chat(
                    agent,
                    message=message,
                    max_turns=1,
                    silent=True
                )

            content = get_autogen_response(agent, proxy, chat_result)
            parsed = extract_json_from_text(content)
            validate(instance=parsed, schema=schema)
            return parsed

        except Exception as e:
            last_error = str(e)

            if attempt >= max_retries:
                raise

    raise RuntimeError("AutoGen call failed unexpectedly.")


# -------------------------
# HDM
# -------------------------

def run_hdm(row, task_result, agents, proxies):
    # Step 1: claim extraction
    extraction_output = call_autogen_agent_with_schema(
        agent=agents["claim_extractor"],
        proxy=proxies["claim_extractor"],
        user_msg=build_claim_extraction_user_msg(task_result),
        schema=claim_extraction_schema,
        clear_history=True
    )

    extracted_claims = extraction_output["claims"]

    print(
        f"Extracted claims for row_id={row.get('row_id', '')}:\n"
        f"{json.dumps(extracted_claims, indent=2, ensure_ascii=False)}\n"
    )

    # Step 2: claim verification
    verification_output = call_autogen_agent_with_schema(
        agent=agents["claim_verifier"],
        proxy=proxies["claim_verifier"],
        user_msg=build_claim_verification_user_msg(row, extracted_claims),
        schema=claim_verification_schema,
        clear_history=True
    )

    claim_verifications = verification_output["claim_verifications"]

    print(
        f"Claim verifications for row_id={row.get('row_id', '')}:\n"
        f"{json.dumps(claim_verifications, indent=2, ensure_ascii=False)}\n"
    )

    wrong_claims = [
        c for c in claim_verifications
        if c["label"] in ["unsupported", "contradicted"]
    ]

    total_claims = len(claim_verifications)
    wrong_claim_count = len(wrong_claims)

    hallucination_rate = safe_divide(wrong_claim_count, total_claims)

    return {
        "extracted_claims": extracted_claims,
        "claim_verifications": claim_verifications,
        "wrong_claims": wrong_claims,
        "total_claims": total_claims,
        "wrong_claim_count": wrong_claim_count,
        "hallucination_rate": hallucination_rate
    }


# -------------------------
# Single-row evaluation
# -------------------------

def evaluate_single_row(
    row,
    model="gemma3:64k",
    writer_temperature=0.0,
    judge_temperature=0.0,
    run_id=1,
    use_hdm_revision=True
):
    row_start = time.perf_counter()

    # Create separate AutoGen agents for this row
    agents, proxies = create_row_agents(
        model=model,
        writer_temperature=writer_temperature,
        judge_temperature=judge_temperature
    )

    # -------------------------
    # Initial writer
    # -------------------------
    writer_user_msg = build_writer_user_msg(row)

    writer_start = time.perf_counter()

    baseline1_writer_output = call_autogen_agent_with_schema(
        agent=agents["writer"],
        proxy=proxies["writer"],
        user_msg=writer_user_msg,
        schema=baseline1_writer_schema,
        clear_history=True
    )

    writer_elapsed = time.perf_counter() - writer_start

    initial_task_result = baseline1_writer_output["task_result"]

    print(
        f"Initial generated related work for row_id={row.get('row_id', '')}:\n"
        f"{initial_task_result}\n"
    )

    # -------------------------
    # HDM before revision
    # -------------------------
    hdm_before_start = time.perf_counter()

    hdm_before = run_hdm(
        row=row,
        task_result=initial_task_result,
        agents=agents,
        proxies=proxies
    )

    hdm_before_elapsed = time.perf_counter() - hdm_before_start

    hdm_feedback_msg = build_hdm_feedback_msg(hdm_before["wrong_claims"])

    # Default: use original writer output
    final_writer_output = baseline1_writer_output
    final_task_result = initial_task_result
    hdm_after = hdm_before
    revision_elapsed = 0.0
    hdm_after_elapsed = 0.0

    # -------------------------
    # If wrong claims exist, send feedback to same writer_agent and revise once
    # -------------------------
    if use_hdm_revision and hdm_before["wrong_claim_count"] > 0:
        print(
            f"HDM found {hdm_before['wrong_claim_count']} wrong claim(s). "
            f"Revising related-work output..."
        )

        revision_start = time.perf_counter()

        revision_user_msg = build_writer_revision_user_msg(
            row=row,
            previous_writer_output=baseline1_writer_output,
            hdm_feedback_msg=hdm_feedback_msg
        )

        revised_writer_output = call_autogen_agent_with_schema(
            agent=agents["writer"],
            proxy=proxies["writer"],
            user_msg=revision_user_msg,
            schema=baseline1_writer_schema,
            clear_history=False
        )

        revision_elapsed = time.perf_counter() - revision_start

        final_writer_output = revised_writer_output
        final_task_result = revised_writer_output["task_result"]

        print(
            f"Revised generated related work for row_id={row.get('row_id', '')}:\n"
            f"{final_task_result}\n"
        )

        # Recompute HDM after revision
        hdm_after_start = time.perf_counter()

        hdm_after = run_hdm(
            row=row,
            task_result=final_task_result,
            agents=agents,
            proxies=proxies
        )

        hdm_after_elapsed = time.perf_counter() - hdm_after_start

    # -------------------------
    # Judge final output
    # -------------------------
    final_expected_milestones = final_writer_output["final_expected_milestones"]

    mandatory_total = sum(
        1 for m in final_expected_milestones
        if m.get("priority") == "MANDATORY"
    )

    optional_total = sum(
        1 for m in final_expected_milestones
        if m.get("priority") == "OPTIONAL"
    )

    judge_user_msg = build_judge_user_msg(
        row,
        final_expected_milestones,
        final_task_result
    )

    judge_start = time.perf_counter()

    judge_output = call_autogen_agent_with_schema(
        agent=agents["judge"],
        proxy=proxies["judge"],
        user_msg=judge_user_msg,
        schema=judge_schema_baseline1,
        clear_history=True
    )

    judge_elapsed = time.perf_counter() - judge_start

    total_elapsed = time.perf_counter() - row_start

    # -------------------------
    # Metric extraction
    # -------------------------
    milestone_judgment = judge_output.get("milestone_judgment", {})
    milestone_scores = milestone_judgment.get("milestone_scores", [])

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

    hdm_before_rate = hdm_before.get("hallucination_rate", None)
    hdm_after_rate = hdm_after.get("hallucination_rate", None)

    hdm_before_wrong_count = hdm_before.get("wrong_claim_count", 0)
    hdm_after_wrong_count = hdm_after.get("wrong_claim_count", 0)

    hallucination_rate_delta = None
    hallucination_count_delta = None
    hallucination_reduction_rate = None
    hallucination_change_status = "unknown"

    if hdm_before_rate is not None and hdm_after_rate is not None:
        hallucination_rate_delta = hdm_after_rate - hdm_before_rate
        hallucination_count_delta = hdm_after_wrong_count - hdm_before_wrong_count

        if hdm_after_rate < hdm_before_rate:
            hallucination_change_status = "reduced"
        elif hdm_after_rate > hdm_before_rate:
            hallucination_change_status = "increased"
        else:
            hallucination_change_status = "unchanged"

        if hdm_before_rate != 0:
            hallucination_reduction_rate = (
                hdm_before_rate - hdm_after_rate
            ) / hdm_before_rate
        else:
            hallucination_reduction_rate = 0.0

    return {
        "row_id": row.get("row_id", ""),
        "split": get_row_value(row, "split"),
        "aid": get_row_value(row, "aid"),
        "mid": get_row_value(row, "mid"),
        "model": model,
        "run_id": run_id,
        "writer_temperature": writer_temperature,
        "judge_temperature": judge_temperature,
        "raw_ref_count": get_row_value(row, "raw_ref_count"),
        "ref_bin": get_row_value(row, "ref_bin"),

        "query_abstract": get_row_value(row, "abstract"),
        "gold_related_work": get_row_value(row, "related_work"),

        "writer_elapsed_sec": round(writer_elapsed, 3),
        "hdm_before_elapsed_sec": round(hdm_before_elapsed, 3),
        "revision_elapsed_sec": round(revision_elapsed, 3),
        "hdm_after_elapsed_sec": round(hdm_after_elapsed, 3),
        "judge_elapsed_sec": round(judge_elapsed, 3),
        "total_elapsed_sec": round(total_elapsed, 3),

        "total_milestones": total_milestones,
        "achieved_milestones_count": achieved_milestones_count,
        "achieved_milestone_ids_json": json.dumps(achieved_milestone_ids, ensure_ascii=False),
        "mandatory_milestones_achieved": mandatory_milestones_achieved,
        "optional_milestones_achieved": optional_milestones_achieved,

        "milestone_achievement_rate": safe_divide(
            achieved_milestones_count,
            total_milestones
        ),
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

        "source_grounding": dimension_scores.get("source_grounding", None),
        "citation_faithfulness": dimension_scores.get("citation_faithfulness", None),
        "reference_coverage": dimension_scores.get("reference_coverage", None),
        "synthesis_quality": dimension_scores.get("synthesis_quality", None),
        "organization": dimension_scores.get("organization", None),
        "specificity": dimension_scores.get("specificity", None),
        "writing_quality": dimension_scores.get("writing_quality", None),
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

        "coherence": quality_analysis.get("coherence", None),
        "readability": quality_analysis.get("readability", None),
        "conciseness": quality_analysis.get("conciseness", None),
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

        "writer_related_work_title": final_writer_output.get("related_work_title", ""),
        "judge_related_work_title": judge_output.get("related_work_title", ""),
        "strengths_json": json.dumps(summary.get("strengths", []), ensure_ascii=False),
        "weaknesses_json": json.dumps(summary.get("weaknesses", []), ensure_ascii=False),

        # HDM outputs
        "hdm_wrong_claim_count_before": hdm_before["wrong_claim_count"],
        "hdm_wrong_claim_count_after": hdm_after["wrong_claim_count"],
        "hdm_total_claims_before": hdm_before["total_claims"],
        "hdm_total_claims_after": hdm_after["total_claims"],
        "hdm_hallucination_rate_before": hdm_before["hallucination_rate"],
        "hdm_hallucination_rate_after": hdm_after["hallucination_rate"],
        "hdm_hallucination_rate_delta": hallucination_rate_delta,
        "hdm_hallucination_count_delta": hallucination_count_delta,
        "hdm_hallucination_change_status": hallucination_change_status,
        "hdm_hallucination_reduction_rate": hallucination_reduction_rate,
        "hdm_feedback_msg": hdm_feedback_msg,
        "hdm_before_json": json.dumps(hdm_before, ensure_ascii=False),
        "hdm_after_json": json.dumps(hdm_after, ensure_ascii=False),

        "final_expected_milestones_json": json.dumps(final_expected_milestones, ensure_ascii=False),
        "initial_generated_related_work": initial_task_result,
        "generated_related_work": final_task_result,
        "initial_writer_output_json": json.dumps(baseline1_writer_output, ensure_ascii=False),
        "writer_output_json": json.dumps(final_writer_output, ensure_ascii=False),
        "judge_output_json": json.dumps(judge_output, ensure_ascii=False)
    }

def load_subset_dataframe(data_path):
    if data_path.endswith(".jsonl"):
        return pd.read_json(data_path, lines=True)

    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)

        # Convert ref_abstract string back to dict if needed
        if "ref_abstract" in df.columns:
            df["ref_abstract"] = df["ref_abstract"].apply(parse_ref_abstract)

        return df

    raise ValueError(f"Unsupported file format: {data_path}")


# -------------------------
# Whole-dataset runner
# -------------------------
def run_singleagent_hdm_rw_on_dataset(
    csv_path,
    model="gemma3:64k",
    writer_temperature=0.0,
    judge_temperature=0.0,
    run_id=1,
    max_retries=3,
    results_csv_path="SingleAgent+HDM_rw/SingleAgent+HDM_rw_gemma3/singleagent_hdm_rw_results_gemma3.csv"
):
    df = load_subset_dataframe(csv_path)
    results = []

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

                row_result = evaluate_single_row(
                    row,
                    model=model,
                    writer_temperature=writer_temperature,
                    judge_temperature=judge_temperature,
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
                "writer_temperature": writer_temperature,
                "judge_temperature": judge_temperature,
                "raw_ref_count": get_row_value(row, "raw_ref_count"),
                "ref_bin": get_row_value(row, "ref_bin"),
                "query_abstract": get_row_value(row, "abstract"),
                "gold_related_work": get_row_value(row, "related_work"),
                "retry_count": max_retries,
                "run_status": last_error_msg
            })

    dataset_elapsed = time.perf_counter() - dataset_start

    results_df = pd.DataFrame(results)

    os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)
    results_df.to_csv(results_csv_path, index=False)

    print(f"\nFinished dataset run.")
    print(f"Total dataset elapsed time: {round(dataset_elapsed, 3)} seconds")
    print(f"Total rows saved: {len(results_df)}")
    print(f"Successful rows: {(results_df['run_status'] == 'successful').sum()}")
    print(f"Errored rows after retries: {(results_df['run_status'] != 'successful').sum()}")
    print(f"Results saved to: {results_csv_path}")

    if len(results_df) > 0 and "writer_elapsed_sec" in results_df.columns:
        successful_df = results_df[results_df["run_status"] == "successful"]

        if len(successful_df) > 0:
            print(f"Average writer time: {round(successful_df['writer_elapsed_sec'].mean(), 3)} sec")
            print(f"Average HDM before time: {round(successful_df['hdm_before_elapsed_sec'].mean(), 3)} sec")
            print(f"Average revision time: {round(successful_df['revision_elapsed_sec'].mean(), 3)} sec")
            print(f"Average HDM after time: {round(successful_df['hdm_after_elapsed_sec'].mean(), 3)} sec")
            print(f"Average judge time: {round(successful_df['judge_elapsed_sec'].mean(), 3)} sec")
            print(f"Average total time per row: {round(successful_df['total_elapsed_sec'].mean(), 3)} sec")
    return results_df, dataset_elapsed

# -------------------------
# Run on the whole dataset
# -------------------------

CSV_PATH = "related_work_generation_dataset.jsonl"

TEMPERATURES = [0.0, 0.5, 1.0]
RUNS_PER_TEMPERATURE = 3
MAX_RETRIES = 3

BASE_OUTPUT_DIR = "SingleAgent+HDM_rw_arr/SingleAgent+HDM_rw_arr_gemma3"

all_results = []

for temp in TEMPERATURES:
    for run_id in range(1, RUNS_PER_TEMPERATURE + 1):

        temp_label = str(temp).replace(".", "p")

        results_csv_path = os.path.join(
            BASE_OUTPUT_DIR,
            f"singleagent_hdm_rw_gemma3_writerT{temp_label}_judgeT0_run{run_id}.csv"
        )

        print("\n" + "=" * 80)
        print(
            f"Running SingleAgent+HDM Related Work Generation | model=gemma3:64k | "
            f"writer_temperature={temp} | judge_temperature=0.0 | "
            f"run_id={run_id} | max_retries={MAX_RETRIES}"
        )
        print("=" * 80)

        results_df, dataset_elapsed = run_singleagent_hdm_rw_on_dataset(
            csv_path=CSV_PATH,
            model="gemma3:64k",
            writer_temperature=temp,
            judge_temperature=0.0,
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
    "singleagent_hdm_rw_gemma3_all_temperatures_combined_with_rerun.csv"
)

combined_results_df.to_csv(combined_results_path, index=False)

print("\nTemperature experiment finished.")
print(f"Combined results saved to: {combined_results_path}")
print(f"Total rows: {len(combined_results_df)}")
print(f"Successful rows: {(combined_results_df['run_status'] == 'successful').sum()}")
print(f"Errored rows after retries: {(combined_results_df['run_status'] != 'successful').sum()}")