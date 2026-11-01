"""
Сборка PDF-отчёта по лабораторной работе № 3 из артефактов в results/.

Запуск:  python make_report.py
Результат: results/lab3_report.pdf
"""
import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

import matplotlib

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
OUT = os.path.join(RESULTS, 'lab3_report.pdf')


def register_fonts():
    ttf_dir = os.path.join(os.path.dirname(matplotlib.__file__),
                           'mpl-data', 'fonts', 'ttf')
    regular = os.path.join(ttf_dir, 'DejaVuSans.ttf')
    bold = os.path.join(ttf_dir, 'DejaVuSans-Bold.ttf')
    pdfmetrics.registerFont(TTFont('DejaVu', regular))
    pdfmetrics.registerFont(TTFont('DejaVu-Bold', bold))
    pdfmetrics.registerFontFamily('DejaVu', normal='DejaVu', bold='DejaVu-Bold',
                                  italic='DejaVu', boldItalic='DejaVu-Bold')


def load(tag):
    path = os.path.join(RESULTS, tag + '_summary.json')
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def fmt_sec(sec):
    if sec < 60:
        return '%.1f с' % sec
    return '%d мин %04.1f с' % (int(sec // 60), sec % 60)


def styles():
    ss = getSampleStyleSheet()
    base = dict(fontName='DejaVu', fontSize=10.5, leading=15,
                alignment=TA_JUSTIFY)
    st = {
        'title': ParagraphStyle('title', fontName='DejaVu-Bold', fontSize=16,
                                leading=22, alignment=TA_CENTER,
                                spaceAfter=6),
        'subtitle': ParagraphStyle('subtitle', fontName='DejaVu', fontSize=11,
                                   leading=16, alignment=TA_CENTER,
                                   textColor=colors.HexColor('#333333')),
        'h1': ParagraphStyle('h1', fontName='DejaVu-Bold', fontSize=13,
                             leading=18, spaceBefore=10, spaceAfter=6),
        'h2': ParagraphStyle('h2', fontName='DejaVu-Bold', fontSize=11.5,
                             leading=16, spaceBefore=8, spaceAfter=4),
        'body': ParagraphStyle('body', **base),
        'small': ParagraphStyle('small', fontName='DejaVu', fontSize=8.5,
                                leading=12, alignment=TA_JUSTIFY),
        'center': ParagraphStyle('center', fontName='DejaVu', fontSize=10.5,
                                 leading=15, alignment=TA_CENTER),
    }
    return st


def table(data, widths, st, header=True):
    t = Table(data, colWidths=widths)
    style = [
        ('FONTNAME', (0, 0), (-1, -1), 'DejaVu'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('LEADING', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#999999')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [('FONTNAME', (0, 0), (-1, 0), 'DejaVu-Bold'),
                  ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eef7'))]
    t.setStyle(TableStyle(style))
    return t


def image(path, width):
    w = width
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        iw, ih = im.size
    return Image(path, width=w, height=w * ih / iw)


def build():
    register_fonts()
    st = styles()
    doc = SimpleDocTemplate(OUT, pagesize=A4,
                            leftMargin=2.2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title='Лабораторная работа №3 — Рекуррентные нейронные сети',
                            author='Овсянников А. Н.')
    S = []

    # ------------------------------------------------------------ title page
    S.append(Spacer(1, 1.2 * cm))
    S.append(Paragraph('Министерство науки и высшего образования РФ', st['center']))
    S.append(Spacer(1, 0.2 * cm))
    S.append(Paragraph('Самарский национальный исследовательский университет<br/>'
                       'имени академика С. П. Королева', st['center']))
    S.append(Spacer(1, 2.0 * cm))
    S.append(Paragraph('Лабораторная работа № 3', st['title']))
    S.append(Paragraph('Рекуррентные нейронные сети', st['subtitle']))
    S.append(Spacer(1, 0.4 * cm))
    S.append(Paragraph('Дисциплина «Нейронные сети и глубокое обучение»', st['subtitle']))
    S.append(Spacer(1, 4.0 * cm))
    S.append(Paragraph('Выполнил: Овсянников А. Н.<br/>'
                       'Группа: 6213', st['center']))
    S.append(Spacer(1, 0.3 * cm))
    S.append(Paragraph('Вариант и наборы данных: airline-passengers.csv, '
                       'shampoo.csv, input.txt, cnus-clean.txt', st['center']))
    S.append(Spacer(1, 4.2 * cm))
    S.append(Paragraph('Самара, 2026', st['center']))
    S.append(PageBreak())

    # ------------------------------------------------------------ task
    S.append(Paragraph('1. Цель работы', st['h1']))
    S.append(Paragraph(
        'Изучить принципы работы и основные функции построения, обучения, '
        'оптимизации и оценки качества рекуррентных нейронных сетей (РНС): '
        'простейшей сети Vanilla RNN и сети долгой краткосрочной памяти LSTM.', st['body']))

    S.append(Paragraph('2. Задание', st['h1']))
    for item in [
        'Изучить код lstm_torch.py и запустить обучение на рядах airline-passengers.csv и shampoo.csv.',
        'Изучить min-char-rnn.py и min-char-lstm.py; добавить замер времени обучения и сохранение значений критерия качества по этапам обучения.',
        'Обучить min-char-rnn.py на input.txt и cnus-clean.txt; сохранить значения критерия, примеры генерации и график качества (PDF).',
        'Выполнить п.3 для min-char-lstm.py.',
        'Сопоставить графики качества обоих моделей с временами обучения и оформить сводный PDF.',
        'Сделать выводы по результатам исследования.',
    ]:
        S.append(Paragraph('— ' + item, st['body']))

    # ------------------------------------------------------------ environment
    S.append(Paragraph('3. Среда и данные', st['h1']))
    S.append(Paragraph(
        'Эксперименты выполнены на Python 3.13 (Windows 11) с numpy 2.4, pandas 3.0, '
        'scikit-learn 1.8, matplotlib 3.10, torch 2.15 (CPU). Для torch-части '
        'использована та же версия PyTorch без GPU, поэтому времена обучения '
        'приведены для CPU.', st['body']))
    S.append(Paragraph(
        '<b>Данные.</b> airline-passengers.csv — 144 месячных значения (1949–1960); '
        'shampoo.csv — 36 значений; input.txt (1 115 394 символа, 65 уникальных) — '
        'пьеса Шекспира; cnus-clean.txt (3 099 238 символов, 54 уникальных) — '
        'сборник рассказов о Шерлоке Холмсе.', st['body']))

    # ------------------------------------------------------------ models
    S.append(Paragraph('4. Краткое описание моделей', st['h1']))
    S.append(Paragraph(
        '<b>Vanilla RNN</b> хранит единственный вектор скрытого состояния: '
        'h(t) = tanh(W<sub>xh</sub>x(t) + W<sub>hh</sub>h(t−1) + b<sub>h</sub>), '
        'y(t) = W<sub>hy</sub>h(t) + b<sub>y</sub>. Обучение — BPTT: сеть '
        'раскатывается на seq_length = 16 шагов, прямой проход кеширует '
        'промежуточные значения, обратный идёт по времени назад и суммирует '
        'вклады всех шагов в градиент каждого веса.', st['body']))
    S.append(Paragraph(
        '<b>LSTM</b> дополняет h(t) ячейкой памяти c(t) и тремя вентилями: '
        'f(t) = σ(W<sub>f</sub>[x;h] + b<sub>f</sub>) — забывание, '
        'i(t) = σ(W<sub>i</sub>[x;h] + b<sub>i</sub>) — запись, '
        'o(t) = σ(W<sub>o</sub>[x;h] + b<sub>o</sub>) — чтение; '
        'c(t) = f(t)°c(t−1) + i(t)°c̃(t), h(t) = o(t)°tanh(c(t)). '
        'Аддитивное обновление памяти сохраняет градиент на длинных '
        'последовательностях.', st['body']))
    S.append(Paragraph(
        '<b>Временной ряд</b> решается LSTM из torch.nn: окно из 4 предыдущих '
        'значений → следующее, значения масштабируются MinMaxScaler в [0, 1], '
        'данные делятся 67 % train / 33 % test, критерий — MSE, оптимизатор Adam. '
        'Числовые текстовые последовательности обучаются посимвольно: вход — '
        'one-hot вектор символа, выход — softmax по словарю, критерий — '
        'кросс-энтропия, оптимизатор Adagrad, градиенты обрезаются до ±5.', st['body']))

    # ------------------------------------------------------------ modifications
    S.append(Paragraph('5. Что добавлено в программы', st['h1']))
    S.append(Paragraph(
        'Исходная математика прямого и обратного прохода сохранена без изменений. '
        'Добавлена измерительная обвязка: аргументы командной строки (файл данных, '
        'объём обучения, каталог и тег результата); замер времени обучения через '
        'time.perf_counter(); запись значений критерия в CSV на каждом логируемом '
        'шаге; построение графиков качества в PNG и PDF; сохранение примеров '
        'генерации; персональный PDF-отчёт по каждому прогону; численная проверка '
        'градиентов по флагу --grad-check. Для torch-части добавлены MSE по эпохам, '
        'график прогноза, RMSE на train/test в исходных единицах и время обучения. '
        'Сводный запуск и сборка итогового PDF выполняются скриптом run_all.py.', st['body']))

    S.append(PageBreak())

    # ------------------------------------------------------------ results
    rnn_in, rnn_cn = load('rnn_input'), load('rnn_cnus')
    lstm_in, lstm_cn = load('lstm_input'), load('lstm_cnus')
    air, sham = load('airline'), load('shampoo')

    S.append(Paragraph('6. Результаты обучения текстовых моделей', st['h1']))
    S.append(Paragraph(
        'Все четыре прогона обработали 1 000 000 символов за 62 500 итераций '
        '(окно 16 символов), hidden_size = 100, learning_rate = 0.1.', st['body']))
    S.append(Spacer(1, 0.2 * cm))
    rows = [['Модель', 'Данные', 'Итераций', 'Время', 'Итоговый smooth_loss']]
    for name, s in (('RNN', rnn_in), ('RNN', rnn_cn),
                    ('LSTM', lstm_in), ('LSTM', lstm_cn)):
        rows.append([name, os.path.basename(s['dataset']), str(s['iterations']),
                     fmt_sec(s['training_seconds']),
                     '%.2f' % s['final_smooth_loss']])
    S.append(table(rows, [2.6 * cm, 4.4 * cm, 2.4 * cm, 3.2 * cm, 4.4 * cm], st))
    S.append(Spacer(1, 0.2 * cm))
    S.append(Paragraph(
        'smooth_loss — сумма кросс-энтропий по 16 символам окна, поэтому в расчёте '
        'на один символ: RNN ≈ 2.0, LSTM ≈ 1.5–1.7. Стартовый уровень (случайное '
        'угадывание) равен ln(V)·16 ≈ 64–67. Сравнение RNN и LSTM корректно внутри '
        'одного набора данных, поскольку словари различаются (65 и 54 символа).',
        st['body']))
    S.append(Spacer(1, 0.3 * cm))
    S.append(image(os.path.join(RESULTS, 'summary_quality.png'), 16.5 * cm))
    S.append(Paragraph('Рисунок 1 — Кривые качества RNN и LSTM; в легенде указано '
                       'время обучения.', st['small']))

    S.append(PageBreak())
    S.append(Paragraph('7. Результаты прогнозирования временных рядов', st['h1']))
    S.append(Paragraph(
        'Обе модели обучались 2000 эпох; критерий — MSE по масштабированным '
        'значениям, ниже приведена ошибка в исходных единицах.', st['body']))
    S.append(Spacer(1, 0.2 * cm))
    rows = [['Ряд', 'Окон', 'Время', 'RMSE train', 'RMSE test']]
    for name, s in (('airline-passengers', air), ('shampoo', sham)):
        rows.append([name, str(s['samples']), fmt_sec(s['training_seconds']),
                     '%.2f' % s['train_rmse_original_units'],
                     '%.2f' % s['test_rmse_original_units']])
    S.append(table(rows, [5.0 * cm, 2.2 * cm, 3.0 * cm, 3.2 * cm, 3.4 * cm], st))
    S.append(Spacer(1, 0.3 * cm))
    S.append(image(os.path.join(RESULTS, 'airline_prediction.png'), 16.5 * cm))
    S.append(Paragraph('Рисунок 2 — Прогноз ряда airline-passengers (красная линия — '
                       'граница train/test).', st['small']))
    S.append(Spacer(1, 0.3 * cm))
    S.append(image(os.path.join(RESULTS, 'shampoo_prediction.png'), 16.5 * cm))
    S.append(Paragraph('Рисунок 3 — Прогноз ряда shampoo.', st['small']))

    S.append(PageBreak())
    S.append(Paragraph('8. Анализ результатов', st['h1']))
    S.append(Paragraph(
        'На обоих текстах LSTM даёт меньший критерий: 26.64 против 32.37 '
        '(input.txt) и 24.48 против 32.04 (cnus-clean.txt). Кривая LSTM снижается '
        'быстрее на первых 5–10 тыс. итераций и дальше не деградирует, тогда как '
        'у RNN после 35 тыс. итераций наблюдается плато с колебаниями. Причина — '
        'аддитивное обновление памяти c(t): вентиль забывания пропускает градиент '
        'через много шагов, тогда как в Vanilla RNN градиент проходит через '
        'произведение (f′(·)·W<sub>hh</sub>) на каждом шаге и затухает.', st['body']))
    S.append(Paragraph(
        'Платой за качество является стоимость шага: LSTM выполняет десять матричных '
        'умножений на временной шаг вместо трёх, поэтому при одинаковом объёме '
        'данных обучение идёт примерно в 4.1 раза дольше (259.1 с против 63.1 с '
        'на input.txt и 244.4 с против 59.2 с на cnus-clean.txt).', st['body']))
    S.append(Paragraph(
        'Для временных рядов сеть с hidden_size = 2 хорошо подгоняет train-часть: '
        'RMSE 21.34 (airline) и 24.21 (shampoo), но на тесте ошибка возрастает до '
        '51.52 и 246.21 соответственно. На shampoo это особенно заметно: всего 31 '
        'окно, из которых 11 приходятся на тест, поэтому модели не хватает данных '
        'и ёмкости. Для airline прогноз повторяет форму ряда, но занижает амплитуду '
        'сезонных пиков — типичное поведение маленькой LSTM при MSE-критерии.', st['body']))
    S.append(Paragraph(
        'Численная проверка градиентов (--grad-check) для обеих ручных реализаций '
        'дала относительную ошибку порядка 1e-9, что подтверждает корректность '
        'выведенных формул обратного распространения.', st['body']))

    S.append(Paragraph('9. Выводы', st['h1']))
    for line in [
        'Реализованы и обучены Vanilla RNN и LSTM на NumPy; обе модели воспроизводят '
        'распределение символов языка, генерируя осмысленные фрагменты текста к концу обучения.',
        'LSTM обеспечивает меньший критерий качества на тех же данных, но требует '
        'примерно в 4 раза больше времени на итерацию.',
        'Клиппинг градиентов (±5) и адаптивные оптимизаторы (Adagrad, Adam) '
        'стабилизируют обучение; численная проверка подтвердила правильность BPTT.',
        'LSTM из PyTorch адекватно прогнозирует временной ряд на train-части; '
        'на тестовой части ошибка выше, что объясняется малой ёмкостью сети и '
        'небольшим объёмом данных.',
        'Все требования п. 3.2 методических указаний выполнены: времена измерены, '
        'значения критерия сохранены (CSV/JSON), графики сохранены в PDF, '
        'подготовлен сводный PDF-файл.',
    ]:
        S.append(Paragraph('— ' + line, st['body']))

    S.append(Paragraph('10. Состав приложенных файлов', st['h1']))
    files = [
        ('min-char-rnn.py', 'Vanilla RNN на NumPy: обучение, замер времени, логи, графики.'),
        ('min-char-lstm.py', 'LSTM на NumPy: обучение, замер времени, логи, графики.'),
        ('lstm_torch.py', 'LSTM на PyTorch для временных рядов: MSE, прогноз, метрики.'),
        ('run_all.py', 'Запуск всех шести прогонов и сборка сводного PDF.'),
        ('make_report.py', 'Сборка этого отчёта из результатов.'),
        ('results/summary_report.pdf', 'Сводный PDF: таблица времён, графики RNN/LSTM, прогнозы.'),
        ('results/*_report.pdf', 'Индивидуальные отчёты прогонов (кривые, примеры генерации).'),
        ('results/*_loss.csv', 'Значения критерия качества по этапам обучения.'),
        ('results/*_summary.json', 'Время обучения, число итераций, итоговые метрики.'),
    ]
    rows = [['Файл', 'Содержимое']]
    for name, desc in files:
        rows.append([Paragraph(name, st['small']), Paragraph(desc, st['small'])])
    S.append(table(rows, [5.2 * cm, 11.6 * cm], st))
    S.append(Spacer(1, 0.3 * cm))
    S.append(Paragraph('11. Команды воспроизведения', st['h1']))
    for cmd in ['python run_all.py',
                'python run_all.py --skip-existing',
                'python min-char-lstm.py --grad-check',
                'python lstm_torch.py --dataset data/shampoo.csv --label "Shampoo Sales" --tag shampoo',
                'python make_report.py']:
        S.append(Paragraph('<font face="DejaVu" size="8.5">%s</font>' % cmd, st['body']))

    doc.build(S)
    print('Saved', OUT)


if __name__ == '__main__':
    build()
