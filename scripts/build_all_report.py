"""Build readable Russian reports from verified metrics and editorial metadata."""
import argparse,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def clean(s):return '\n'.join(line.rstrip() for line in s.replace('ё','е').replace('Ё','Е').split('\n'))
def table(df):
 return '| '+' | '.join(df.columns)+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join(str(v).replace('|','/') for v in row)+' |' for row in df.itertuples(index=False,name=None))+'\n'
def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=ROOT/'results');p.add_argument('--output',type=Path,default=ROOT/'REPORT.md');a=p.parse_args()
 methods=json.loads((ROOT/'docs/methods.json').read_text());families=json.loads((ROOT/'docs/families.json').read_text());sources=json.loads((ROOT/'docs/sources.json').read_text())
 s=pd.read_csv(a.results/'analysis/all_summary.csv');runs=pd.read_csv(a.results/'analysis/all_runs.csv');assert set(s.name)==set(methods)
 def link(part,name):return f'[{methods[name]["title"]}](results/methods/{part}_{name}.md)'
 def report_table(part,names,base,second=None):
  z=s[s.part==part].set_index('name').loc[names];acc=s[(s.part==part)&(s.name==base)].accuracy_pct.iloc[0]
  tab=pd.DataFrame({'Как выбирали данные':[link(part,n) for n in names],'Верных ответов, %':[f'{v:.2f}' for v in z.accuracy_pct],'Разброс, п.п.':[f'{v:.2f}' for v in z.sd_pp], 'Разница, п.п.':['—' if n==base else f'{z.loc[n,"accuracy_pct"]-acc:+.2f}' for n in names]})
  if second:
   tab=tab.rename(columns={'Разница, п.п.':'К обычной случайной, п.п.'});acc2=s[(s.part==part)&(s.name==second)].accuracy_pct.iloc[0]
   tab['К набору без нестабильных, п.п.']=['—' if n==second else f'{z.loc[n,"accuracy_pct"]-acc2:+.2f}' for n in names]
  return table(tab)
 replacements={
 '{{TABLE_PART1}}':'Сравнение со случайным отбором первой серии.\n\n'+report_table('part1',['random','heldout_forgetting','dedup_exact','lexical_diversity','rare_words'],'random'),
 '{{TABLE_SKILLS}}':'Сравнение со случайным отбором из тех же доступных строк.\n\n'+report_table('part2',['skill_random','negation_keyword','contrast_keyword','negation_embedding','contrast_embedding','negation_gradient','contrast_gradient','negation_randomref_gradient','contrast_randomref_gradient'],'skill_random'),
 '{{TABLE_PAIRS}}':'Сравнение с обычным случайным набором парной серии.\n\n'+report_table('part2',['pair_random','pair_matched_random','pair_pairs','pair_broken_pairs'],'pair_random'),
 '{{TABLE_FORGET_DOSES}}':report_table('part3',['forget_random','forget_matched_zero','forget_dose05','forget_dose10','forget_all'],'forget_random','forget_matched_zero'),
 '{{TABLE_REPAIRS}}':'Сравнение со случайным набором того же состава.\n\n'+report_table('part3',['skill_matched_random_v2','grad_raw_mix25','grad_nobias_mix25','grad_randomref_nobias_mix25','neg_embedding_mix25'],'skill_matched_random_v2')}
 cards=a.results/'methods';cards.mkdir(exist_ok=True)
 for row in s.itertuples():
  info=methods[row.name];family=families[info['family']];base=methods[row.baseline]['title']
  g=runs[(runs.part==row.part)&(runs.name==row.name)].sort_values('seed')
  performance=pd.DataFrame({'Показатель':['Верных ответов (accuracy)','F1, среднее двух классов','F1 положительного класса','Верных на фразах с отрицанием, 164 строки','Верных на фразах с противопоставлением, 121 строка'],
   'Среднее, %':[f'{v:.2f}' for v in [row.accuracy_pct,row.macro_f1_pct,row.positive_f1_pct,row.negation_pct,row.contrast_pct]]})
  tab=g[['seed','accuracy','macro_f1','negation_accuracy','contrast_accuracy']].copy()
  for c in tab.columns[1:]:tab[c]=tab[c].map(lambda x:f'{100*x:.2f}')
  tab=tab.rename(columns={'seed':'Номер запуска','accuracy':'Верных ответов, %','macro_f1':'F1 двух классов, %','negation_accuracy':'На отрицаниях, %','contrast_accuracy':'На противопоставлениях, %'})
  comparison=f'Сравнение: **{base.lower()}**. Разница средней точности: **{row.delta_pp:+.2f} п.п.**.' if row.baseline!=row.name else 'Это контрольный вариант: с ним сравниваются другие способы отбора в той же группе.'
  if row.part=='part3' and row.name.startswith('forget_') and row.name!='forget_random':
   random_acc=s[(s.part=='part3')&(s.name=='forget_random')].accuracy_pct.iloc[0]
   comparison+=f' Относительно обычной случайной выборки: **{row.accuracy_pct-random_acc:+.2f} п.п.** Это другое сравнение.'
  refs='\n'.join(f'- [{sources[k]["short"]}]({sources[k]["url"]}).' for k in family['sources']) or 'Это контрольное условие для сравнения методов.'
  card=f'''# {info['title']}

**Результат: {row.accuracy_pct:.2f}% верных ответов, разброс {row.sd_pp:.2f} п.п.**

{info['meaning']}

## Идея и реализация

{info['why']}

{info['how']}

## Метрики и сравнение

{comparison}

{table(performance)}
## Сравнение со статьями

{family['paper']}

{family['comparison']}

{refs}

Полные названия и проверенные места статей — в [списке источников](../../docs/SOURCES.md).

## Вывод

{info['next']}

Вывод основан на трех запусках. При одновременном изменении нескольких условий оценивается их совместный эффект.

## Каждый запуск

{table(tab)}
Разброс в начале карточки — стандартное отклонение. Различных наборов текстов: {row.distinct_subsets}. Если набор один, между запусками менялось только обучение. Группы отрицаний и противопоставлений выделены по словам, без независимой проверки смысла.

## Где проверить числа

Имя в коде: `{row.name}`. Серия: `{row.part}`. Полная таблица всех метрик, включая F1 положительного класса и внутреннюю проверку, — [all_runs.csv](../analysis/all_runs.csv).

'''
  for run in g.itertuples():card+=f'- Запуск {run.seed}: [метрики](../raw/{row.part}/runs/{run.run_id}/metrics.json), [ответы модели](../raw/{row.part}/runs/{run.run_id}/predictions.csv), [параметры](../raw/{row.part}/runs/{run.run_id}/identity.json).\n'
  (cards/f'{row.part}_{row.name}.md').write_text(clean(card))
 index=['## Карточки методов']
 for part,title in [('part1','Простые способы'),('part2','Языковые группы и пары'),('part3','Проверка исправлений')]:
  index+=['### '+title,'\n'+'\n'.join('- '+link(part,n) for n in s[s.part==part].name if not (part=='part3' and n.startswith('forget_')))+'\n']
 replacements['{{METHOD_INDEX}}']='\n\n'.join(index)
 examples=pd.read_csv(a.results/'analysis/selection_examples.csv')
 erows=[]
 for name in ['random','lexical_diversity','rare_words']:
  for label in [0,1]:
   row=examples[(examples.method==name)&(examples.label==label)].sort_values('pool_index').iloc[0]
   erows.append({'Метод':methods[name]['title'],'Фраза SST-2':row.text.strip(),'Метка':'положительная' if label else 'отрицательная','Слов':int(row.words)})
 replacements['{{SELECTION_EXAMPLES}}']=table(pd.DataFrame(erows))
 epochs=pd.read_csv(a.results/'analysis/training_epochs.csv')
 trainrows=[]
 for part,name in [('part1','random'),('part1','heldout_forgetting'),('part1','rare_words'),('part2','skill_random'),('part2','negation_gradient'),('part2','contrast_gradient')]:
  g=epochs[(epochs.part==part)&(epochs.name==name)]
  trainrows.append({'Метод':link(part,name),'Val после 1 эпохи, %':f'{g.val_epoch1_pct.mean():.2f}','Val после 3 эпох, %':f'{g.val_epoch3_pct.mean():.2f}','Test выбранных весов, %':f'{g.test_accuracy_pct.mean():.2f}'})
 replacements['{{TRAINING_TABLE}}']=table(pd.DataFrame(trainrows))
 report=(ROOT/'docs/report_template.md').read_text()
 for old,new in replacements.items():report=report.replace(old,new)
 assert '{{' not in report
 a.output.write_text(clean(report).rstrip() + '\n')
 src=['# Статьи и связь с моими опытами\n','Проверено по arXiv 18 ноября 2026. Для трех ключевых сравнений прочитан полный текст; для остальных — аннотация. Ниже указан объем проверки. Числа относятся к условиям авторов. Ссылки ведут на первоисточники.\n']
 for key,source in sources.items():
  src.append(f'## {source["short"]}\n\n[{source["title"]}]({source["url"]})\n\n**Что проверено:** {source["checked"]}.\n\n**Результат статьи.** {source["result"]}\n\n**Связь с моим опытом.** {source["difference"]}\n')
 src+=['## Модель представлений\n\n[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) из Sentence Transformers. Ссылка на Sentence-BERT объясняет общий подход к представлениям предложений.\n',
 '## Собственные правила отбора\n\nСлучайные контроли, средняя редкость слов, доля уникальных слов, конкретные доли добавок и мои ограничения состава придуманы для этого проекта. \n']
 (ROOT/'docs/SOURCES.md').write_text(clean('\n'.join(src)))
 print('Updated report, 28 method cards and source notes')
if __name__=='__main__':main()
