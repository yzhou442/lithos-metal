"""The prompt set shared by the speculative-decoding benches (``spec_lmbench``, ``spec_policy_ab``, ``spec_copy_sim``).

``load_prompts(source)`` returns ``(PROMPTS, SUITES, LONG, doc_prompt)``:

* ``PROMPTS``: name -> (chat messages, generated tokens) for the short prompts (code, chat, math, web, tool) and the
  copy-heavy ``agent`` prompts (``edit``: rewrite a Python module the prompt contains; ``json``: edit a JSON recipe);
* ``LONG``: name -> (characters, question, generated tokens) for the long-context prompts built by ``doc_prompt`` from
  the repo's design and hardware docs;
* ``SUITES``: suite name -> prompt names.

``source`` is a lithos-metal checkout whose files supply the document and ``agent`` prompt text. The text (and so
the token ids) depends on that checkout's files, so comparisons across commits should pass one fixed checkout to
every run (for example the commit both sides branch from).
"""

from __future__ import annotations

import json
from pathlib import Path

SYS_WEB = ('You are a coding expert. Produce a complete, self-contained single HTML file with inline CSS and '
           'JavaScript. Output only the code.')
PROMPTS = {
    'code': ([{'role': 'user', 'content': 'Write a Python function that parses an ISO-8601 timestamp without '
              'using the datetime module. Include a docstring, input validation and unit tests.'}], 256),
    'chat': ([{'role': 'user', 'content': 'Explain to a curious high-school student how attention works in a '
              'transformer language model, with an everyday analogy.'}], 256),
    'math': ([{'role': 'user', 'content': 'A train leaves at 9:40 travelling 84 km/h; a second train leaves the '
              'same station at 10:05 at 102 km/h on a parallel track. When and where does the second catch up? '
              'Solve step by step.'}], 256),
    'web': ([{'role': 'system', 'content': SYS_WEB},
             {'role': 'user', 'content': 'Build a pricing page with three tiers, a monthly/yearly toggle and a '
              'feature comparison table.'}], 384),
    'tool': ([{'role': 'user', 'content': 'Here is a shell session:\n$ ls src\nmain.rs lib.rs parser.rs\n$ cargo test\n'
              'error[E0308]: mismatched types\n --> src/parser.rs:42:17\n   |\n42 |     let n: u32 = tok.len();\n'
              '   |            ---   ^^^^^^^^^ expected `u32`, found `usize`\n\nExplain the error and give the fixed '
              'line plus a short justification.'}], 192),
}
SUITES = {
    'agent': ['edit', 'json'],
    'quick': ['code', 'chat', 'web'],
    'full': ['code', 'chat', 'math', 'web', 'tool', 'doc4k'],
    'long': ['doc4k', 'doc16k'],
    'prefill': ['doc4k', 'doc16k', 'doc28k'],
}
LONG = {'doc4k': (14000, 'Summarize the key design decisions above in 8 bullet points.', 192),
        'doc16k': (58000, 'List the five most important measured hardware facts above and why each matters.', 160),
        'doc28k': (100000, 'Give a one-paragraph summary.', 32)}


def load_prompts(source):
    """The prompt set with its document and ``agent`` text read from the checkout ``source``."""
    root = Path(source).expanduser().resolve()
    doc = root / 'docs' / 'design' / 'design.md'
    doc2 = root / 'docs' / 'research' / 'apple-gpu-probes.md'

    def doc_prompt(n_chars, ask, max_new):
        text = (doc.read_text() + '\n\n' + doc2.read_text()) * 4
        return ([{'role': 'user', 'content': text[:n_chars] + '\n\n' + ask}], max_new)

    setup = (root / 'monolith' / 'serving' / 'setup.py').read_text()
    recipe = json.dumps(json.loads((root / 'monolith' / 'backends' / 'metal' / 'm5_max_40c' / 'recipes' / 'dspark'
                                    / 'selected-nvfp4-endpoints.json').read_text())['128']['target'], indent=1)
    prompts = dict(PROMPTS)
    prompts['edit'] = ([{'role': 'user', 'content': 'Here is a Python module:\n```python\n' + setup + '```\nAdd a one-line '
                         'docstring to every function and method that lacks one, change nothing else, and output the '
                         'complete updated file.'}], 640)
    prompts['json'] = ([{'role': 'user', 'content': 'Here is a JSON config:\n```json\n' + recipe + '\n```\nChange every '
                         '"workers" value of 160 to 192 and every "sgs" value of 4 to 8. Output the complete updated JSON '
                         'only.'}], 512)
    return prompts, dict(SUITES), dict(LONG), doc_prompt
