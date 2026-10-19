"""Record2Vec step 2: frozen-LLM summaries of the serialised first-W-hour record (local vLLM, OpenAI-compatible API).

Prompt styles
  narrative   free-text clinical summary (3-6 sentences)
  structured  per-organ-system assessment (respiratory / coagulation / liver / cardiovascular / CNS / renal / metabolic),
              trajectory, and an overall impression -- same information, fixed schema
  notes       narrative prompt + the last <=3 clinical notes available in the window (UTDE pickles) -> unified text (condition D)
Temperature 0, one call per stay; results cached per line in results/record2vec/summ_W{W}_{style}.jsonl (resumable).
Usage: VLLM_BASE_URL=http://localhost:8011/v1 python scripts/17_record2vec_summarize.py --window 24 --style narrative structured
"""
import argparse
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'record2vec'
UTDE = ROOT.parent / 'MultimodalMIMIC' / 'Data' / 'sofa28'
PLACEHOLDER = 'No clinical note available in this window.'

SYSTEM = ('You are an experienced intensivist. You summarise early ICU monitoring data for a colleague. '
          'Use only the data given; do not invent values, diagnoses or outcomes. Do not state a mortality probability.')
NARRATIVE = ('Below are the measurements from the first {W} hours of an ICU stay. Write a concise clinical summary (3-6 sentences) '
             'describing the physiological state, organ dysfunction, and whether the trajectory is stable, improving or worsening. '
             'Mention the most abnormal values with their numbers.\n\n{rec}')
STRUCTURED = ('Below are the measurements from the first {W} hours of an ICU stay. Fill in the following schema, one short line each '
              '(write "not assessed" when the data are absent):\n'
              'Respiratory (SpO2, FiO2, PaO2, respiratory rate, oxygen need):\nCoagulation (platelets):\nLiver (bilirubin):\n'
              'Cardiovascular (blood pressure, heart rate, vasopressor use):\nCentral nervous system (GCS):\n'
              'Renal (creatinine, urine output):\nMetabolic (lactate, base excess, temperature):\n'
              'Trajectory over the window (stable / improving / worsening, with the key numbers):\n'
              'Overall impression (one sentence, severity of illness):\n\n{rec}')
NOTES = ('Below are the measurements from the first {W} hours of an ICU stay, followed by the clinical notes written in that window. '
         'Write a concise clinical summary (4-8 sentences): reason for admission and relevant history (from the notes), current '
         'physiological state and organ dysfunction (from the measurements and notes), and whether the trajectory is stable, improving '
         'or worsening. Mention the most abnormal values with their numbers.\n\n{rec}\n\nCLINICAL NOTES:\n{notes}')
PROMPTS = {'narrative': NARRATIVE, 'structured': STRUCTURED, 'notes': NOTES}


def load_notes(W):
    out = {}
    for split in ('train', 'val', 'test'):
        for d in pickle.load(open(UTDE / f'W{W}' / f'{split}p2x_data.pkl', 'rb')):
            notes = [t for t in d['text_data'] if t != PLACEHOLDER][-3:]
            out[d['name']] = '\n\n---\n\n'.join(t[:4000] for t in notes) if notes else PLACEHOLDER   # ~1k tokens per note
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, required=True)
    ap.add_argument('--style', nargs='+', default=['narrative', 'structured'])
    ap.add_argument('--model', default='qwen25-7b')
    ap.add_argument('--workers', type=int, default=120)
    ap.add_argument('--max-tokens', type=int, default=600)
    args = ap.parse_args()
    client = OpenAI(base_url=os.environ.get('VLLM_BASE_URL', 'http://localhost:8011/v1'), api_key='EMPTY')
    ser = pd.read_parquet(OUT / f'serialized_W{args.window}.parquet')
    notes = load_notes(args.window) if 'notes' in args.style else None

    def one(sid, prompt):
        from openai import BadRequestError
        for cut in (22000, 15000, 10000, 7000):            # numeric records tokenise densely; shrink until it fits the 8k context
            try:
                r = client.chat.completions.create(model=args.model, temperature=0, max_tokens=args.max_tokens, seed=0,
                                                   messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt[:cut]}])
                break
            except BadRequestError as e:
                if 'maximum context length' not in str(e) or cut == 7000:
                    raise
        return {'ICUSTAY_ID': int(sid), 'summary': r.choices[0].message.content.strip(), 'finish': r.choices[0].finish_reason,
                'prompt_tokens': r.usage.prompt_tokens, 'completion_tokens': r.usage.completion_tokens}

    for style in args.style:
        path = OUT / f'summ_W{args.window}_{style}.jsonl'
        done = set()
        if path.exists():                      # keep only complete lines (a killed run may leave a truncated tail)
            lines = [l for l in open(path) if l.endswith('\n')]
            open(path, 'w').writelines(lines); done = {json.loads(l)['ICUSTAY_ID'] for l in lines}
        todo = [(sid, PROMPTS[style].format(W=args.window, rec=row.text, notes=notes[sid] if notes else ''))
                for sid, row in ser.iterrows() if sid not in done]
        print(f'W={args.window} {style}: {len(done)} cached, {len(todo)} to do', flush=True)
        with open(path, 'a') as f, ThreadPoolExecutor(args.workers) as ex:
            for k, res in enumerate(ex.map(lambda a: one(*a), todo)):
                f.write(json.dumps(res) + '\n')
                if k % 500 == 0:
                    f.flush(); print(f'  {k}/{len(todo)}', flush=True)
        print(f'done {style}', flush=True)


if __name__ == '__main__':
    main()
