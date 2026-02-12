# PM Angel vs Devil (Streamlit)

Quick demo app that takes a PM idea and runs it through two parallel modes: Angel (hype) and Devil (roast).

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your_key_here"
```

3. Run the app:

```bash
streamlit run app.py
```

## Notes
- The model can be edited in the sidebar. Default is `gpt-5.1`.
- The devil mode is intentionally critical but should stay playful and non-personal.
