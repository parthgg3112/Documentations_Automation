# utils_openai_summary.py

import openai
import os


def get_openai_summary(sql_content):
    prompt = (
        "You are a data expert. Summarize the following DBT model SQL logic in concise sentences:\n\n"
        f"{sql_content}\n\nSummary:"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=100
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"(OpenAI Summary Failed: {e})"
