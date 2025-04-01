def initialize_logging():
    """
    Initializes logging so that only WARNING and ERROR messages are output.

    This function configures logging with a specific format and level.
    """
    import logging

    logging.basicConfig(
        level=logging.WARNING,  # Only warnings and errors will be logged.
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def extract_first_markdown_table(text):
    """
    Extracts the first markdown table found in the provided text.
    """
    lines = text.splitlines()
    table_lines = []
    in_table = False
    for line in lines:
        if line.strip().startswith("|"):
            table_lines.append(line)
            in_table = True
        elif in_table:
            break
          
    return "\n".join(table_lines)


def parse_markdown_table_block(table_text):
    """
    Parses a markdown table and returns a dictionary mapping headers to data values.
    Raises an error if more than one data row is found.
    """
    import re

    lines = table_text.strip().splitlines()
    if len(lines) < 3:
        raise ValueError(
            "Markdown table does not have enough lines (header, separator, data)."
        )

    header_line = lines[0]
    # Collect all data rows (ignore separator rows)
    data_rows = [
        line for line in lines[1:] if not re.match(r"^\s*\|[\s\-|]+\|\s*$", line)
    ]

    if not data_rows:
        raise ValueError("No data row found in markdown table.")
    if len(data_rows) > 1:
        raise ValueError(
            "More than one data row found in markdown table. Expected exactly one row."
        )

    # Only one data row is allowed.
    data_line = data_rows[0]

    headers = [h.strip() for h in header_line.strip("|").split("|")]
    data = [d.strip() for d in data_line.strip("|").split("|")]

    if len(headers) != len(data):
        raise ValueError("Header and data column count mismatch in markdown table.")

    return dict(zip(headers, data))


async def evaluate_llm_model_async(
    prompt, 
    llm_name, 
    llm_instance, 
    concept_id, 
    concept_name
    ):
    """
    Asynchronously calls an LLM model with retry logic and parses the output.

    This function:
      - Retries the LLM call up to 5 times.
      - Uses asyncio.to_thread for blocking calls.
      - Extracts and parses a markdown table from the output.

    Returns a dictionary with the evaluation results.
    """
    import asyncio
    import time
    import logging

    max_retries = 5
    processing_error = ""
    parse_error = ""
    parse_success = False
    relevance = ""
    reasoning = ""
    commentary = ""
    looked_up_ancestor = ""
    raw_llm_output = ""

    logging.info(
        f"LLM call with model {llm_name} for concept: {concept_id} and concept name: {concept_name}"
    )

    for attempt in range(1, max_retries + 1):
        logging.info(
            f"Attempt {attempt} for concept_id {concept_id} with model {llm_name}"
        )
        try:
            start_time = time.time()
            output_obj = await asyncio.to_thread(llm_instance.invoke, prompt)
            raw_llm_output = (
                output_obj.content
                if hasattr(output_obj, "content")
                else str(output_obj)
            )
            elapsed = time.time() - start_time
            logging.info(
                f"LLM call succeeded on attempt {attempt} for concept_id {concept_id} with model {llm_name} in {elapsed:.2f} sec"
            )
        except Exception as e:
            processing_error = str(e)
            logging.error(
                f"Attempt {attempt} for concept_id {concept_id} with {llm_name} processing error: {processing_error}"
            )
            continue

        try:
            table_text = extract_first_markdown_table(raw_llm_output)
            parsed = parse_markdown_table_block(table_text)
            relevance = parsed.get("relevance_score", "")
            reasoning = parsed.get("reasoning", "")
            commentary = parsed.get("commentary", "")
            looked_up_ancestor = parsed.get("looked_up_ancestor", "")
            parse_success = True
        except Exception as pe:
            parse_error = str(pe)
            logging.error(
                f"Attempt {attempt} for concept_id {concept_id} with {llm_name} parse error: {parse_error}"
            )
            continue

        if parse_success:
            break

    return {
        "concept_id": concept_id,
        "concept_name": concept_name,
        "llm_model": llm_name,
        "relevance_score": relevance,
        "reasoning": reasoning,
        "commentary": commentary,
        "looked_up_ancestor": looked_up_ancestor,
        "raw_llm_output": raw_llm_output,
        "raw_llm_input": prompt,
        "processing_error": processing_error,
        "parse_error": parse_error,
        "parse_success": parse_success,
    }


async def run_llm_evaluations_async(
    final_df, 
    llm_dict, 
    system_prompt, 
    clinical_prompt
    ):
    """
    Launches asynchronous LLM calls for each row in a DataFrame with incremental progress updates.

    For each row in the DataFrame, this function:
      - Constructs a full prompt by combining the system prompt, clinical prompt, and the row's text.
      - Creates an asynchronous task for each LLM in llm_dict.
      - As tasks complete, prints a status line including progress and active thread count.

    Returns a DataFrame with the results of each LLM call.
    """
    import pandas as pd
    import asyncio
    import threading
    from datetime import datetime

    if final_df.empty:
        print("No records to process.")
        return pd.DataFrame()

    tasks = []
    for idx, row in final_df.iterrows():
        concept_id = row.get("concept_id")
        concept_name = row.get("concept_name", "")
        text_val = row.get("text", "")
        prompt = f"{system_prompt}\n\n{clinical_prompt}\n\n{text_val}"
        for llm_name, llm_instance in llm_dict.items():
            tasks.append(
                evaluate_llm_model_async(prompt, llm_name, llm_instance, concept_id, concept_name)
            )

    total_tasks = len(tasks)
    completed_tasks = 0
    progress_bar_length = 20
    results = []

    print(f"Processing concept ids using LLM model: {llm_name}")

    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
        completed_tasks += 1
        percent_complete = (completed_tasks / total_tasks) * 100
        num_hashes = int(percent_complete / (100 / progress_bar_length))
        progress_bar = (
            "[" + "#" * num_hashes + " " * (progress_bar_length - num_hashes) + "]"
        )
        active_threads = threading.active_count()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{current_time} [STATUS] {progress_bar} {percent_complete:.0f}% | Processed: {completed_tasks}/{total_tasks} | Parallel Threads: {active_threads}"
        )
    return pd.DataFrame(results)


def validate_and_prepare_concept_dataframe(
    data_frame,
    concept_set_id,
    type_filter=("recommended", "resolved_standard", "resolved_source"),
    ):
    """
    Validates and prepares the input DataFrame for LLM processing.

    This function checks that the required columns exist, filters the DataFrame to include only rows with the
    specified concept_set_id and types, and constructs a text column by concatenating concept_name, synonyms, and ancestors.

    Returns a DataFrame with 'concept_id' and a combined 'text' column.
    """
    import logging
    import pandas as pd

    required_columns = {
        "concept_id",
        "concept_name",
        "synonyms",
        "ancestors",
        "type",
        "concept_set_id",
    }
    missing_columns = required_columns - set(data_frame.columns)
    if missing_columns:
        raise ValueError(f"data_frame is missing required columns: {missing_columns}")

    logging.info(
        f"Filtering DataFrame for concept_set_id = {concept_set_id} and type in {type_filter}"
    )
    filtered_df = (
        data_frame[
            (data_frame["concept_set_id"] == concept_set_id)
            & (data_frame["type"].isin(type_filter))
        ][["concept_id", "concept_name", "synonyms", "ancestors"]]
        .copy()
        .drop_duplicates()
    )

    filtered_df["text"] = (
        'concept_name: "'
        + filtered_df["concept_name"].fillna("").astype(str)
        + '"; '
        + 'concept_synonym: "'
        + filtered_df["synonyms"].fillna(filtered_df["concept_name"]).astype(str)
        + '"; '
        + 'concept_ancestor: "'
        + filtered_df["ancestors"].fillna(filtered_df["concept_name"]).astype(str)
        + '"'
    )

    # Rename columns for clarity in the resulting DataFrame.
    filtered_df = filtered_df.rename(
        columns={"synonyms": "concept_synonym", "ancestors": "concept_ancestor"}
    )
    final_df = filtered_df[["concept_id", "concept_name", "text"]].copy()
    
    return final_df


def merge_llm_results_with_concepts(
    processed_df, 
    data_frame
    ):
    """
    Merges LLM evaluation results with the original concept details.

    If the 'concept_name' column is missing from the original data, returns processed_df unchanged.
    Otherwise, resets the index, attempts to attach concept names from the original data, and reorders columns.
    If the row counts mismatch, a warning is issued.
    """
    import pandas as pd
    import logging

    if "concept_name" not in data_frame.columns:
        logging.warning(
            "'concept_name' column missing from data_frame; skipping concept details attachment."
        )
        return processed_df

    processed_df = processed_df.reset_index(drop=True)
    original_concepts = (
        data_frame[["concept_id", "concept_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if len(original_concepts) != len(processed_df):
        logging.warning(
            "Row count mismatch between processed results and original data; concept details attachment may be unreliable."
        )
    else:
        processed_df["concept_name"] = original_concepts["concept_name"]

    cols = list(processed_df.columns)
    if "concept_name" in cols:
        cols.remove("concept_name")
        cols.insert(1, "concept_name")
    processed_df = processed_df[cols]

    return processed_df


def write_results_to_delta_table(
    merged_df, 
    table_name, 
    spark_lock
    ):
    """
    Writes the merged DataFrame to a Delta table using Spark.

    This function writes the DataFrame to a specified Delta table and then reads the table back into a Pandas DataFrame.
    """
    import logging

    try:
        spark.createDataFrame(merged_df).write.format("delta").option(
            "overwriteSchema", "true"
        ).mode("overwrite").saveAsTable(table_name)
        merged_df = spark.sql(f"SELECT * FROM {table_name}").toPandas()
    except Exception as e:
        logging.error(f"Error writing to Delta table: {e}")
        
    return merged_df


def process_concept_reviews_async(
    data_frame,
    concept_set_id,
    table_name,
    llm_dict,
    system_prompt,
    clinical_prompt,
    type_filter=("recommended", "resolved_standard", "resolved_source"),
    ):
    """
    Orchestrates asynchronous processing of a concept set review.

    This function validates and prepares the input DataFrame, launches asynchronous LLM evaluations,
    merges the LLM results with the original metadata, and writes the final merged DataFrame to a Delta table.
    """
    import threading
    import logging

    try:
        import nest_asyncio

        nest_asyncio.apply()
    except ImportError:
        logging.warning(
            "nest_asyncio not installed. Nested event loops may cause errors."
        )

    initialize_logging()
    spark_lock = threading.Lock()

    final_df = validate_and_prepare_concept_dataframe(
        data_frame, concept_set_id, type_filter
    )
    if final_df.empty:
        print(f"No records to process for concept_set_id: {concept_set_id}.")
        return final_df

    logging.info("Starting asynchronous LLM processing...")
    import asyncio

    loop = asyncio.get_event_loop()
    processed_df = loop.run_until_complete(
        run_llm_evaluations_async(final_df, llm_dict, system_prompt, clinical_prompt)
    )

    merged_df = merge_llm_results_with_concepts(processed_df, data_frame)
    merged_df = write_results_to_delta_table(merged_df, table_name, spark_lock)

    logging.info("Completed asynchronous processing of concept set reviews.")
    
    return merged_df


async def process_conditions_async(
    conditions,
    table_name,
    spark,
    process_concept_reviews_async,
    data_frame,
    llm_dict,
    system_prompt,
    type_filter=("recommended", "resolved_standard", "resolved_source")
    ):
    """
    Processes a list of clinical conditions incrementally by leveraging asynchronous review processing.

    For each condition, this function:
      - Loads existing data from a Delta table.
      - Checks if the concept set for the condition has already been processed.
      - If not, calls process_concept_reviews_async to perform asynchronous LLM evaluations.
      - Adds columns for the condition and concept_set_id.
      - Appends the new results to the cumulative DataFrame and writes the combined results back to the Delta table.
    """
    from datetime import datetime
    import threading

    existing_df = load_existing_delta_data(table_name, spark)
    processed_concept_ids = get_processed_concept_set_ids(existing_df)

    for idx, condition in enumerate(conditions):
        active_threads = threading.active_count()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        concept_set_id = condition.get("concept_set_id")
        cond_name = condition.get("name", "N/A")
        print(
            f"{current_time} [HEADER] Processing concept_set_id: {concept_set_id}, "
            f"Condition: {cond_name} using {active_threads} parallel thread(s)."
        )

        existing_df = await process_condition_async(
            condition=condition,
            existing_df=existing_df,
            processed_concept_ids=processed_concept_ids,
            process_concept_reviews_async=process_concept_reviews_async,
            data_frame=data_frame,
            llm_dict=llm_dict,
            system_prompt=system_prompt,
            type_filter=type_filter,
            spark=spark,
            table_name=table_name,
        )
        
    return existing_df


def load_existing_delta_data(
    table_name, 
    spark=None
    ):
    """
    Loads existing data from the specified Delta table.

    This function attempts to read the Delta table using Spark. If the table exists,
    it converts the table to a Pandas DataFrame; otherwise, an empty DataFrame is returned.
    """
    import pandas as pd

    try:
        if spark.catalog.tableExists(table_name):
            existing_df = spark.table(table_name).toPandas()
            print("Existing table found. Loaded current data.")
        else:
            existing_df = pd.DataFrame()
            print("No existing table found. Starting fresh.")
    except Exception as e:
        existing_df = pd.DataFrame()
        print(f"Error loading existing data: {e}")
    return existing_df


def get_processed_concept_set_ids(existing_df):
    """
    Retrieves a set of processed concept_set_ids from the existing DataFrame.
    """
    if not existing_df.empty and "concept_set_id" in existing_df.columns:
        return set(existing_df["concept_set_id"].unique())
    return set()


async def process_condition_async(
    condition,
    existing_df,
    processed_concept_ids,
    process_concept_reviews_async,
    data_frame,
    llm_dict,
    system_prompt,
    type_filter,
    table_name,
    spark=None
    ):
    """
    Processes a single clinical condition by checking if it has already been processed and, if not,
    running asynchronous LLM evaluations and updating the Delta table.
    """
    import pandas as pd

    cond_name = condition.get("name", "Unknown")
    concept_set_id = condition.get("concept_set_id")

    if concept_set_id in processed_concept_ids:
        print(
            f"Skipping {cond_name} (concept_set_id: {concept_set_id}): already processed."
        )
        return existing_df

    result_df = process_concept_reviews_async(
        data_frame=data_frame,
        concept_set_id=concept_set_id,
        table_name=table_name,
        llm_dict=llm_dict,
        system_prompt=system_prompt,
        clinical_prompt=condition.get("clinical_prompt", ""),
        type_filter=type_filter,
    )

    result_df = add_condition_columns_and_reorder(result_df, cond_name, concept_set_id)

    if existing_df.empty:
        cumulative_df = result_df.copy()
    else:
        cumulative_df = pd.concat([existing_df, result_df], ignore_index=True)

    cumulative_df = update_delta_table_with_results(cumulative_df, table_name, spark)
    processed_concept_ids.add(concept_set_id)
    print(
        f"Processed and saved output for {cond_name} (concept_set_id: {concept_set_id})."
    )
    
    return cumulative_df


def add_condition_columns_and_reorder(
    result_df, 
    cond_name, 
    concept_set_id
    ):
    """
    Adds condition-specific columns to the result DataFrame and reorders the columns.

    This function adds 'condition' and 'concept_set_id' columns and moves them to the beginning
    of the DataFrame for clarity.
    """
    result_df["condition"] = cond_name
    result_df["concept_set_id"] = concept_set_id
    cols = result_df.columns.tolist()
    for col in ["condition", "concept_set_id"]:
        if col in cols:
            cols.remove(col)
    new_order = ["condition", "concept_set_id"] + cols
    
    return result_df[new_order]


def update_delta_table_with_results(
    cumulative_df, 
    table_name, 
    spark=None
    ):
    """
    Writes the cumulative DataFrame to a Delta table and reads it back for verification.

    This function writes the DataFrame to the specified Delta table using Spark, then reads the table
    back into a Pandas DataFrame.
    """
    import logging

    try:
        spark.createDataFrame(cumulative_df).write.format("delta").option(
            "overwriteSchema", "true"
        ).mode("overwrite").saveAsTable(table_name)
        cumulative_df = spark.sql(f"SELECT * FROM {table_name}").toPandas()
    except Exception as e:
        logging.error(f"Error writing to Delta table: {e}")
        
    return cumulative_df


def run_process_conditions(
    conditions,
    table_name,
    process_concept_reviews_async,
    data_frame,
    llm_dict,
    system_prompt,
    type_filter=("resolved_standard", "resolved_source"),
    spark=None
    ):
    """
    Orchestrates the complete asynchronous processing of clinical conditions.

    This function runs the asynchronous process_conditions_async function. It uses asyncio.run
    if no event loop is running; otherwise, it retrieves the current event loop and runs until completion.
    """
    try:
        import nest_asyncio

        nest_asyncio.apply()
    except ImportError:
        pass

    import asyncio

    try:
        return asyncio.run(
            process_conditions_async(
                conditions=conditions,
                table_name=table_name,
                spark=spark,
                process_concept_reviews_async=process_concept_reviews_async,
                data_frame=data_frame,
                llm_dict=llm_dict,
                system_prompt=system_prompt,
                type_filter=type_filter,
            )
        )
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            process_conditions_async(
                conditions=conditions,
                table_name=table_name,
                spark=spark,
                process_concept_reviews_async=process_concept_reviews_async,
                data_frame=data_frame,
                llm_dict=llm_dict,
                system_prompt=system_prompt,
                type_filter=type_filter,
            )
        )


def getLLMmodel(
    dial_key,
    temperature=0.0, 
    azure_endpoint="https://ai-proxy.lab.epam.com", 
    api_version="2024-08-01-preview",
    llm_model=None
    ):

    from langchain_openai import AzureChatOpenAI
    
    if llm_model is None:
         llm_dict = {
              "claude_sonnet": AzureChatOpenAI(
                  api_key=dial_key,
                  api_version=api_version,
                  azure_endpoint=azure_endpoint,
                  model="anthropic.claude-v3-sonnet",
                  temperature=temperature,
              ),
              "gpt-4o-full": AzureChatOpenAI(
                  api_key=dial_key,
                  api_version=api_version,
                  azure_endpoint=azure_endpoint,
                  model="gpt-4o",
                  temperature=temperature,
              )
                  }
    elif llm_model == "claude":
           llm_dict = {
                "claude_sonnet": AzureChatOpenAI(
                    api_key=dial_key,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint,
                    model="anthropic.claude-v3-sonnet",
                    temperature=temperature,
                )
                      }
    elif llm_model == "gpt-4o-full":
           llm_dict =  {
                "gpt-4o-full": AzureChatOpenAI(
                    api_key=dial_key,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint,
                    model="gpt-4o",
                    temperature=temperature,
                )
                       }
    else:
      print("Wrong LLM model selected.")
  
    return llm_dict
