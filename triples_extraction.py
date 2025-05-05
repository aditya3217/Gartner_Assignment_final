import openai
import os
import re
from typing import List, Tuple
openai.api_key = os.environ['OPENAI_API_KEY']
def extract_triples(text: str) -> List[Tuple[str, str, str]]:
    prompt = f"""
    Extract semantic triples from the following text in the format (subject, predicate, object):

    Text:
    \"\"\"
    {text}
    \"\"\"

    Triples:
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )

    result = response.choices[0].message['content']
    triples = []

    pattern = r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)"

    matches = re.findall(pattern, result)

    for match in matches:
        subject, predicate, obj = match
        triples.append((subject.strip(), predicate.strip(), obj.strip()))

    return triples
