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

def safe_divide(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator

# -------------------------
# Ollama call with schema
# -------------------------
# -------------------------
# AutoGen agent setup
# -------------------------

_AUTOGEN_AGENT_CACHE = {}


def extract_json_from_text(text):
    """
    AutoGen/local models may sometimes return markdown fences or extra text.
    This extracts the JSON object safely.
    """
    text = text.strip()

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
                "timeout": timeout
            }
        ],
        "temperature": temperature,
        "cache_seed": None
    }


def get_autogen_agents(
    model="gemma3:64k",
    timeout=900,
    writer_temperature=0.0,
    judge_temperature=0.0
):
    """
    Creates separate AutoGen objects for writer and judge.

    Important:
    - writer_temperature is varied in the experiment.
    - judge_temperature should stay fixed at 0.0 for consistent evaluation.
    """

    writer_temp_label = str(writer_temperature).replace(".", "p")
    judge_temp_label = str(judge_temperature).replace(".", "p")

    cache_key = f"{model}_{timeout}_writerT{writer_temp_label}_judgeT{judge_temp_label}"

    if cache_key in _AUTOGEN_AGENT_CACHE:
        return _AUTOGEN_AGENT_CACHE[cache_key]

    writer_llm_config = build_llm_config(
        model=model,
        timeout=timeout,
        temperature=writer_temperature
    )

    judge_llm_config = build_llm_config(
        model=model,
        timeout=timeout,
        temperature=judge_temperature
    )

    writer_agent = autogen.AssistantAgent(
    name=f"writer_agent_T{writer_temp_label}",
    system_message=baseline1_writer_system_msg,
    llm_config=writer_llm_config
)

    judge_agent = autogen.AssistantAgent(
        name=f"judge_agent_T{judge_temp_label}",
        system_message=judge_system_msg_baseline1,
        llm_config=judge_llm_config
    )

    writer_proxy = autogen.UserProxyAgent(
        name="writer_proxy",
        human_input_mode="NEVER",
        code_execution_config=False
    )

    judge_proxy = autogen.UserProxyAgent(
        name="judge_proxy",
        human_input_mode="NEVER",
        code_execution_config=False
    )

    agents = {
        "writer": writer_agent,
        "judge": judge_agent
    }

    proxies = {
        "writer": writer_proxy,
        "judge": judge_proxy
    }

    _AUTOGEN_AGENT_CACHE[cache_key] = (agents, proxies)

    return agents, proxies


def choose_agent_key(system_msg):
    """
    Chooses which AutoGen agent object to use based on the system message.
    """
    if system_msg == baseline1_writer_system_msg:
        return "writer"

    if system_msg == judge_system_msg_baseline1:
        return "judge"

    raise ValueError("Unknown system message. Could not choose AutoGen agent.")


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


# -------------------------
# AutoGen call with schema
# -------------------------

def call_ollama_with_schema(
    system_msg,
    user_msg,
    schema,
    model="gemma3:64k",
    timeout=900,
    writer_temperature=0.0,
    judge_temperature=0.0
):
    """
    Same function name as before, so the rest of your code does not need to change.

    Now it uses separate AutoGen agent objects:
    - writer_agent for writing
    - judge_agent for judging
    """

    grounded_user_msg = (
        user_msg
        + "\n\nThe response must conform to this JSON schema exactly:\n"
        + json.dumps(schema, indent=2, ensure_ascii=False)
        + "\n\nReturn ONLY the JSON object. Do not include markdown or explanation."
    )

    agents, proxies = get_autogen_agents(
    model=model,
    timeout=timeout,
    writer_temperature=writer_temperature,
    judge_temperature=judge_temperature
)
    agent_key = choose_agent_key(system_msg)

    agent = agents[agent_key]
    proxy = proxies[agent_key]

    try:
        chat_result = proxy.initiate_chat(
            agent,
            message=grounded_user_msg,
            max_turns=1,
            silent=True,
            clear_history=True
        )
    except Exception as e:
        raise RuntimeError(
            f"AutoGen chat failed for agent '{agent_key}'. "
            f"Original error: {type(e).__name__}: {e}"
        )

    content = get_autogen_response(agent, proxy, chat_result)
    parsed = extract_json_from_text(content)

    validate(instance=parsed, schema=schema)

    return parsed

# -------------------------
# Single-row evaluation
# -------------------------
def evaluate_single_row(
    row,
    model="gemma3:64k",
    writer_temperature=0.0,
    judge_temperature=0.0,
    run_id=1
):
    row_start = time.perf_counter()

    # Writer
    writer_user_msg = build_writer_user_msg(row)
    writer_start = time.perf_counter()
    baseline1_writer_output = call_ollama_with_schema(
    system_msg=baseline1_writer_system_msg,
    user_msg=writer_user_msg,
    schema=baseline1_writer_schema,
    model=model,
    writer_temperature=writer_temperature,
    judge_temperature=judge_temperature
)
    writer_elapsed = time.perf_counter() - writer_start

    final_expected_milestones = baseline1_writer_output["final_expected_milestones"]
    task_result = baseline1_writer_output["task_result"]
    mandatory_total = sum(
    1 for m in final_expected_milestones
    if m.get("priority") == "MANDATORY"
    )
    optional_total = sum(
    1 for m in final_expected_milestones
    if m.get("priority") == "OPTIONAL"
    )
    print(f"Generated related work for row_id={row.get('row_id', '')}:\n{task_result}\n")
    # Judge
    judge_user_msg = build_judge_user_msg(row, final_expected_milestones, task_result)
    judge_start = time.perf_counter()
    judge_output = call_ollama_with_schema(
    system_msg=judge_system_msg_baseline1,
    user_msg=judge_user_msg,
    schema=judge_schema_baseline1,
    model=model,
    writer_temperature=writer_temperature,
    judge_temperature=judge_temperature
)
    judge_elapsed = time.perf_counter() - judge_start

    total_elapsed = time.perf_counter() - row_start

    milestone_judgment = judge_output.get("milestone_judgment", {})
    milestone_scores = milestone_judgment.get("milestone_scores", [])

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
        "judge_elapsed_sec": round(judge_elapsed, 3),
        "total_elapsed_sec": round(total_elapsed, 3),

        "total_milestones": total_milestones,
        "achieved_milestones_count": achieved_milestones_count,
        "achieved_milestone_ids_json": json.dumps(achieved_milestone_ids, ensure_ascii=False),
        "mandatory_milestones_achieved": mandatory_milestones_achieved,
        "optional_milestones_achieved": optional_milestones_achieved,
        "milestone_achievement_rate": safe_divide(achieved_milestones_count, total_milestones),
        "mandatory_milestone_achievement_rate": safe_divide(mandatory_milestones_achieved, mandatory_total),
        "optional_milestone_achievement_rate": safe_divide(optional_milestones_achieved, optional_total),
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

        "total_dimension_score": (
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
            )
        ) / 7,

        "coherence": quality_analysis.get("coherence", None),
        "readability": quality_analysis.get("readability", None),
        "conciseness": quality_analysis.get("conciseness", None),
        "total_quality_analysis_score": (
            sum(
                quality_analysis.get(dim, 0)
                for dim in ["coherence", "readability", "conciseness"]
            )
        ) / 3,

        "writer_related_work_title": baseline1_writer_output.get("related_work_title", ""),
        "judge_related_work_title": judge_output.get("related_work_title", ""),
        "strengths_json": json.dumps(summary.get("strengths", []), ensure_ascii=False),
        "weaknesses_json": json.dumps(summary.get("weaknesses", []), ensure_ascii=False),

        "final_expected_milestones_json": json.dumps(final_expected_milestones, ensure_ascii=False),
        "generated_related_work": task_result,
        "writer_output_json": json.dumps(baseline1_writer_output, ensure_ascii=False),
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
def run_baseline1_on_dataset(
    csv_path,
    model="gemma3:64k",
    writer_temperature=0.0,
    judge_temperature=0.0,
    run_id=1,
    max_retries=3,
    results_csv_path="Baseline1_SingleAgent+Judge/Baseline1_gemma3/baseline1_results_gemma3.csv"
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

BASE_OUTPUT_DIR = "Baseline1_SingleAgent+Judge/Baseline1_rw_gemma3latest"

all_results = []

for temp in TEMPERATURES:
    for run_id in range(1, RUNS_PER_TEMPERATURE + 1):

        temp_label = str(temp).replace(".", "p")

        results_csv_path = os.path.join(
            BASE_OUTPUT_DIR,
            f"baseline1_rw_gemma3_writerT{temp_label}_judgeT0_run{run_id}.csv"
        )

        print("\n" + "=" * 80)
        print(
            f"Running Baseline1 Related Work Generation | model=gemma3:64k | "
            f"writer_temperature={temp} | judge_temperature=0.0 | "
            f"run_id={run_id} | max_retries={MAX_RETRIES}"
        )
        print("=" * 80)

        results_df, dataset_elapsed = run_baseline1_on_dataset(
            csv_path=CSV_PATH,
            model="gemma3:64k",
            writer_temperature=temp,
            judge_temperature=0.0,
            run_id=run_id,
            max_retries=MAX_RETRIES,
            results_csv_path=results_csv_path
        )

        all_results.append(results_df)

combined_results_df = pd.concat(all_results, ignore_index=True)

combined_results_path = os.path.join(
    BASE_OUTPUT_DIR,
    "baseline1_rw_gemma3_all_temperatures_combined_with_rerun.csv"
)

combined_results_df.to_csv(combined_results_path, index=False)

print("\nTemperature experiment finished.")
print(f"Combined results saved to: {combined_results_path}")
print(f"Total rows: {len(combined_results_df)}")
print(f"Successful rows: {(combined_results_df['run_status'] == 'successful').sum()}")
print(f"Errored rows after retries: {(combined_results_df['run_status'] != 'successful').sum()}")