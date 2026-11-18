"""Check documentation spelling, local links and coverage of all conditions."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 methods=json.loads((ROOT/'docs/methods.json').read_text());sources=json.loads((ROOT/'docs/sources.json').read_text());families=json.loads((ROOT/'docs/families.json').read_text())
 cards=list((ROOT/'results/methods').glob('*.md'));assert len(cards)==len(methods)==28
 for d in methods.values():assert d['family'] in families
 for f in families.values():assert all(k in sources for k in f['sources'])
 required=['## Идея и реализация','## Метрики и сравнение','## Сравнение со статьями','## Вывод','## Каждый запуск']
 for f in cards:
  text=f.read_text();assert all(h in text for h in required),f
 for f in ROOT.rglob('*.md'):
  if '.git' in f.parts:continue
  text=f.read_text();assert '\u0451' not in text and '\u0401' not in text,f
  if f.name!='report_template.md':assert '{{TABLE_' not in text,f
  if f == ROOT/'REPORT.md' or f.parent in (ROOT/'docs',ROOT/'results/methods'):
   assert not re.search(r'\b(?:[Мм]ы|нас|нам|[Нн]аш(?:а|е|и|его|ей|ем|их|им|ими|у)?)\b',text),f
  for target in re.findall(r'\]\(([^)]+)\)',text):
   if target.startswith(('https:','http:','#')):continue
   base = ROOT if f.name == 'report_template.md' else f.parent
   assert (base/target.split('#')[0]).exists(),(f,target)
 print('28 method cards, source mapping, spelling and local links: OK')
if __name__=='__main__':main()
