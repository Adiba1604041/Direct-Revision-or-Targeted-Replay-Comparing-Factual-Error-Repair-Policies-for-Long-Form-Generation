import json
import time
import traceback
import pandas as pd
import requests
import re
import os
from jsonschema import validate
import autogen


# -------------------------
# Fetch the source articles from the row and combine them into a single text block for context.
# -------------------------

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
        "event_title": {"type": "string"},
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
        "event_title",
        "final_expected_milestones",
        "task_result"
    ]
}


baseline1_writer_system_msg = """
You are writer_agent, an expert in public-event narrative synthesis for digital collections.

Your task is to transform structured event metadata and a set of related
source documents into a readable, public-facing event narrative of 400-600
words. The narrative should help a user understand an event-centered
collection without requiring the user to read every source document.

Your job is to do two things:
1. Define the final expected milestones for a high-quality public-event
   narrative.
2. Generate the public-event narrative.

Milestone design rules:
- Milestones must describe what the final public-event narrative should
  accomplish.
- Each milestone must be tagged as either MANDATORY or OPTIONAL.
- MANDATORY milestones are essential for factual correctness and adequate
  completion.
- OPTIONAL milestones improve completeness, context, nuance, accessibility,
  or synthesis quality.
- There must be at least one MANDATORY milestone.
- Milestones should be specific, observable, and useful for judging the final
  output.
- Do not create vague milestones such as "write well" or "be good."
- Focus on event fidelity, important entities, key information, cross-source
  synthesis, narrative structure, public accessibility, and careful claim
  handling.

Grounding and synthesis rules:
- Treat the supplied source documents as the only evidence for factual claims.
- Every factual claim in the narrative must be supported by the supplied
  metadata or source documents.
- Synthesize information across the source documents rather than summarizing
  only one document.
- Do not invent unsupported facts, statistics, quotations, dates, entities,
  motivations, or outcomes.
- If the source documents disagree, describe the disagreement cautiously
  without selecting an unsupported version as certain.
- If important information is missing or uncertain, do not fill the gap with
  speculation.

Narrative rules:
- Write a coherent public-facing narrative based on the event metadata and
  related source documents.
- Write within 400-600 words.
- Provide sufficient context for a reader who has not read the source
  documents.
- The output should be readable, factual, accessible, and neutral.
- The output is not an opinion piece, newspaper column, or journalistic report.

Revision behavior:
- If you later receive HDM feedback, you are revising your own previous output.
- Revise only the parts related to wrong claims.
- Preserve supported content.
- Do not restart from scratch.
- Do not add new unsupported facts.
- Do not change final_expected_milestones during revision.

Return ONLY valid JSON matching the required schema.
Do not include markdown or extra text outside the JSON object.
"""

# -------------------------
# Judge schema
# -------------------------

judge_schema_baseline1 = {
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
        }
    },
    "required": [
        "event_title",
        "milestone_judgment",
        "dimension_scores",
        "quality_analysis",
        "summary"
    ]
}


judge_system_msg_baseline1 = """
You are judge_agent, an expert evaluator of public-event narratives generated
from event-centered digital collections.

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


claim_verification_system_msg = """
You are hdm_claim_verifier_agent, a source-grounded claim verification agent.

Your task is to verify extracted claims from a generated public-event narrative.

You will receive:
1. Extracted claims
2. Constraint-based ground truth
3. Related source documents

Verification labels:
- supported: the claim is supported by the given evidence.
- unsupported: the claim is not supported by the given evidence.
- contradicted: the claim conflicts with the given evidence.

Rules:
- Use the provided source documents and constraint-based ground truth as the only evidence for verification.
- Do not use your internal knowledge to verify claims.
- Be strict about numbers, dates, entities, causal claims, and quotes.
- Do not reward a claim just because it sounds plausible.
- Provide a clear reason for each verification label.
- For unsupported or contradicted claims, the reason should not contain the corrected claim.
- The reason should only explain why the claim is not supported or contradicted.
- Provide evidence for each verification label from the provided sources when available.
- Return ONLY valid JSON matching the schema.
Do not include markdown or extra text outside the JSON object.
"""


# -------------------------
# Prompt builders
# -------------------------

def build_writer_user_msg(row):
    source_documents = build_source_articles_text(row)

    return f"""
Generate a public-event narrative for the following event-centered digital
collection.

Event Metadata:
- Year: {row['year']}
- Month: {row['month']}
- Date: {row['date']}
- Event: {row['event']}

Related Source Documents:
{source_documents}

Generate a readable, factual, and neutral narrative that synthesizes the
supplied source documents for a public audience.

Return ONLY valid JSON with:
- event_title
- final_expected_milestones
- task_result (the task_result should be the full text of the generated
  public-event narrative)
"""


def build_claim_extraction_user_msg(task_result):
    return f"""
Generated Public-Event Narrative:
{task_result}

Extract atomic factual claims from this narrative.

Return ONLY valid JSON with:
- claims
"""


def build_claim_verification_user_msg(row, extracted_claims):
    return f"""
Extracted Claims:
{json.dumps(extracted_claims, indent=2, ensure_ascii=False)}

Constraint-Based Ground Truth:
{row['ground_truth_json']}

Related Source Documents:
{build_source_articles_text(row)}

Verify each claim.

Return ONLY valid JSON with:
- claim_verifications
"""


def build_hdm_feedback_msg(wrong_claims):
    if not wrong_claims:
        return "There is no unsupported or contradicted claim in your public-event narrative. Great job! No revision needed."

    feedback_lines = [
        "HDM found unsupported or contradicted claims in your previous public-event narrative.",
        "Revise only the parts related to the following wrong claims.",
        ""
    ]

    for i, claim in enumerate(wrong_claims, start=1):
        feedback_lines.append(f"{i}. Claim: {claim['claim']}")
        feedback_lines.append(f"   Label: {claim['label']}")
        feedback_lines.append(f"   Reason: {claim['reason']}")
        feedback_lines.append("")

    feedback_lines.append(
        "Do not add new unsupported information. Preserve all supported content from the original narrative."
    )

    return "\n".join(feedback_lines)


def build_writer_revision_user_msg(row, previous_writer_output, hdm_feedback_msg):
    source_documents = build_source_articles_text(row)

    return f"""
You are writer_agent. You previously wrote the following public-event narrative.

Previous Writer Output:
{json.dumps(previous_writer_output, indent=2, ensure_ascii=False)}

HDM Feedback:
{hdm_feedback_msg}

Event Metadata:
- Year: {row['year']}
- Month: {row['month']}
- Date: {row['date']}
- Event: {row['event']}

Related Source Documents:
{source_documents}

Revise the public-event narrative using the HDM feedback.

Important:
- Revise only the parts related to the wrong claims.
- For each unsupported or contradicted claim, use the HDM feedback reason and the related source documents to decide how the claim should be repaired.
- If the reason and source documents provide enough guidance to correct the claim safely, rewrite the claim so that it becomes supported.
- If the reason and source documents do not provide enough guidance to safely correct the claim, remove the claim.
- Preserve supported content.
- Do not restart from scratch.
- Do not change final_expected_milestones. Keep them exactly as in the previous writer output.
- Do not introduce new unsupported claims.
- Maintain a neutral, public-facing narrative style.

Return ONLY valid JSON with:
- event_title
- final_expected_milestones (keep the same as previous writer output)
- task_result
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

    # Remove Qwen-style thinking if present
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
    1. initial public-event narrative generation
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
        use this for independent calls like claim extraction, claim verification, judge.

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
    print(f"Extracted claims for row_id={row.get('row_id', '')}:\n{json.dumps(extracted_claims, indent=2, ensure_ascii=False)}\n")

    # Step 2: claim verification
    verification_output = call_autogen_agent_with_schema(
        agent=agents["claim_verifier"],
        proxy=proxies["claim_verifier"],
        user_msg=build_claim_verification_user_msg(row, extracted_claims),
        schema=claim_verification_schema,
        clear_history=True
    )

    claim_verifications = verification_output["claim_verifications"]
    print(f"Claim verifications for row_id={row.get('row_id', '')}:\n{json.dumps(claim_verifications, indent=2, ensure_ascii=False)}\n")
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
    # Writer
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

    print(f"Initial writer output for row_id={row.get('row_id', '')}:\n{initial_task_result}\n")

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
        print(f"HDM found {hdm_before['wrong_claim_count']} wrong claim(s). Revising writer output...")

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

        print(f"Revised writer output for row_id={row.get('row_id', '')}:\n{final_task_result}\n")

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
    # Existing metric extraction
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
        "model": model,
        "run_id": run_id,
        "writer_temperature": writer_temperature,
        "judge_temperature": judge_temperature,
        "year": row.get("year", ""),
        "month": row.get("month", ""),
        "date": row.get("date", ""),
        "event": row.get("event", ""),

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

        "core_event_fidelity": dimension_scores.get("core_event_fidelity", None),
        "required_entity_coverage": dimension_scores.get("required_entity_coverage", None),
        "key_information_coverage": dimension_scores.get("key_information_coverage", None),
        "cross_source_synthesis": dimension_scores.get("cross_source_synthesis", None),
        "structural_coverage": dimension_scores.get("structural_coverage", None),
        "statistics_and_claims_accuracy": dimension_scores.get("statistics_and_claims_accuracy", None),
        "narrative_quality": dimension_scores.get("narrative_quality", None),
        "total_dimension_score": safe_divide(
            sum(
                dimension_scores.get(dim, 0)
                for dim in [
                    "core_event_fidelity",
                    "required_entity_coverage",
                    "key_information_coverage",
                    "cross_source_synthesis",
                    "structural_coverage",
                    "statistics_and_claims_accuracy",
                    "narrative_quality"
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
        "initial_generated_event_narrative": initial_task_result,
        "generated_event_narrative": final_task_result,
        "writer_output_json": json.dumps(final_writer_output, ensure_ascii=False),
        "judge_output_json": json.dumps(judge_output, ensure_ascii=False)
    }


# -------------------------
# Whole-dataset runner
# -------------------------

def run_singleagent_hdm_on_dataset(
    csv_path,
    model="gemma3:64k",
    writer_temperature=0.0,
    judge_temperature=0.0,
    run_id=1,
    max_retries=3,
    results_csv_path="SingleAgent+HDM_pns/SingleAgent+HDM_pns_gemma3/singleagent_hdm_pns_results_gemma3.csv"
):
    df = pd.read_csv(csv_path)
    results = []

    dataset_start = time.perf_counter()

    for idx, row in df.iterrows():
        print("=" * 80)
        print(f"Processing row {idx + 1}/{len(df)} | row_id={row.get('row_id', idx)}")
        print("=" * 80)

        row_success = False
        last_error_msg = ""

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

        if not row_success:
            results.append({
                "row_index": idx,
                "row_id": row.get("row_id", ""),
                "model": model,
                "run_id": run_id,
                "writer_temperature": writer_temperature,
                "judge_temperature": judge_temperature,
                "year": row.get("year", ""),
                "month": row.get("month", ""),
                "date": row.get("date", ""),
                "event": row.get("event", ""),
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
# Run
# -------------------------

CSV_PATH = "public_narrative_synthesis_dataset.csv"

TEMPERATURES = [0.0, 0.5, 1.0]
RUNS_PER_TEMPERATURE = 3
MAX_RETRIES = 3

BASE_OUTPUT_DIR = "SingleAgent+HDM_pns/SingleAgent+HDM_pns_gemma3"

all_results = []

for temp in TEMPERATURES:
    for run_id in range(1, RUNS_PER_TEMPERATURE + 1):

        temp_label = str(temp).replace(".", "p")

        results_csv_path = os.path.join(
            BASE_OUTPUT_DIR,
            f"singleagent_hdm_pns_gemma3_writerT{temp_label}_judgeT0_run{run_id}.csv"
        )

        print("\n" + "=" * 80)
        print(
            f"Running SingleAgent+HDM PNS | model=gemma3:64k | "
            f"writer_temperature={temp} | judge_temperature=0.0 | "
            f"run_id={run_id} | max_retries={MAX_RETRIES}"
        )
        print("=" * 80)

        results_df, dataset_elapsed = run_singleagent_hdm_on_dataset(
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
    "singleagent_hdm_pns_gemma3_all_temperatures_combined_with_rerun.csv"
)

combined_results_df.to_csv(combined_results_path, index=False)

print("\nTemperature experiment finished.")
print(f"Combined results saved to: {combined_results_path}")
print(f"Total rows: {len(combined_results_df)}")
print(f"Successful rows: {(combined_results_df['run_status'] == 'successful').sum()}")
print(f"Errored rows after retries: {(combined_results_df['run_status'] != 'successful').sum()}")