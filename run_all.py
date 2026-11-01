"""
Запуск всех экспериментов лабораторной работы № 3 и сборка сводного отчёта.

Что делает:
  1. min-char-rnn.py   на data/input.txt и data/cnus-clean.txt;
  2. min-char-lstm.py  на data/input.txt и data/cnus-clean.txt;
  3. lstm_torch.py     на data/airline-passengers.csv и data/shampoo.csv;
  4. строит results/summary_report.pdf: таблица времён обучения, кривые
     качества RNN vs LSTM для обоих текстов, прогнозы для обоих рядов.

Каждый запуск складывает в results/:
  <tag>_loss.csv / .png / .pdf, <tag>_samples.txt, <tag>_summary.json,
  <tag>_report.pdf, а stdout пишется в results/logs/<tag>.log.

Usage:
    python run_all.py
    python run_all.py --max-data 200000 --epochs 500
    python run_all.py --no-char            # только временные ряды
    python run_all.py --show               # показать итоговый график
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import textwrap
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


CHAR_RUNS = [
    ('min-char-rnn.py', 'rnn_input', ['--file', os.path.join('data', 'input.txt')]),
    ('min-char-rnn.py', 'rnn_cnus', ['--file', os.path.join('data', 'cnus-clean.txt')]),
    ('min-char-lstm.py', 'lstm_input', ['--file', os.path.join('data', 'input.txt')]),
    ('min-char-lstm.py', 'lstm_cnus', ['--file', os.path.join('data', 'cnus-clean.txt')]),
]

TORCH_RUNS = [
    ('lstm_torch.py', 'airline',
     ['--dataset', os.path.join('data', 'airline-passengers.csv'),
      '--label', 'Airline Passengers']),
    ('lstm_torch.py', 'shampoo',
     ['--dataset', os.path.join('data', 'shampoo.csv'),
      '--label', 'Shampoo Sales']),
]


def run_one(script, tag, args, out_dir, log_dir, extra, skip_existing=False):
    summary_path = os.path.join(out_dir, tag + '_summary.json')
    if skip_existing and os.path.exists(summary_path):
        print('>>> %s: already done, skipping' % tag, flush=True)
        return None

    cmd = [sys.executable, script, '--tag', tag, '--out-dir', out_dir] + args + extra
    print('>>> ' + ' '.join(cmd), flush=True)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    elapsed = time.perf_counter() - t0
    log_path = os.path.join(log_dir, tag + '.log')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(result.stdout)
    if result.returncode != 0:
        raise RuntimeError('%s failed (exit %d), see %s'
                           % (tag, result.returncode, log_path))
    print('    done in %.1f s -> %s' % (elapsed, log_path), flush=True)
    return elapsed


def read_char_csv(path):
    iters, losses = [], []
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            iters.append(int(row['iteration']))
            losses.append(float(row['smooth_loss']))
    return iters, losses


def read_torch_csv(path):
    epochs, losses = [], []
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            epochs.append(int(row['epoch']))
            losses.append(float(row['train_mse']))
    return epochs, losses


def load_summary(out_dir, tag):
    with open(os.path.join(out_dir, tag + '_summary.json'), encoding='utf-8') as f:
        return json.load(f)


def fmt_seconds(sec):
    if sec is None:
        return '-'
    if sec < 60:
        return '%.1f s' % sec
    return '%d:%04.1f' % (int(sec // 60), sec % 60)


def build_summary(out_dir, char_tags, torch_tags, show=False):
    summaries = {}
    if char_tags:
        for tag in char_tags:
            summaries[tag] = load_summary(out_dir, tag)

    pdf_path = os.path.join(out_dir, 'summary_report.pdf')

    with PdfPages(pdf_path) as pdf:
        # ---------------- Page 1: table of runs ----------------
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(0.04, 0.95, 'Лабораторная работа № 3 — сводные результаты',
                 fontsize=16, fontweight='bold', va='top')
        fig.text(0.04, 0.91,
                 'Рекуррентные сети: Vanilla RNN / LSTM (NumPy) и LSTM (PyTorch)',
                 fontsize=10, va='top')

        y = 0.84
        header = '%-30s %-18s %10s %10s %12s' % (
            'run', 'dataset', 'iter', 'seconds', 'final loss')
        fig.text(0.04, y, header, fontsize=9.5, family='monospace',
                 fontweight='bold', va='top')
        y -= 0.03
        for tag, s in summaries.items():
            line = '%-30s %-18s %10s %10s %12s' % (
                tag,
                os.path.basename(s['dataset'])[:18],
                s.get('iterations', '-'),
                fmt_seconds(s.get('training_seconds')),
                '%.3f' % s['final_smooth_loss'])
            fig.text(0.04, y, line, fontsize=9.5, family='monospace', va='top')
            y -= 0.027

        if torch_tags:
            y -= 0.02
            fig.text(0.04, y, 'PyTorch LSTM runs:', fontsize=11,
                     fontweight='bold', va='top')
            y -= 0.032
            for tag in torch_tags:
                s = load_summary(out_dir, tag)
                lines = [
                    '%s (%s): epochs=%d, train time=%s' % (
                        tag, os.path.basename(s['dataset']),
                        s['epochs'], fmt_seconds(s['training_seconds'])),
                    '    RMSE (original units): train=%.3f, test=%.3f' % (
                        s['train_rmse_original_units'], s['test_rmse_original_units']),
                ]
                for line in lines:
                    fig.text(0.04, y, line, fontsize=9.5, family='monospace',
                             va='top')
                    y -= 0.027

        pdf.savefig(fig)
        plt.close(fig)

        # ---------------- Page 2: RNN vs LSTM loss curves ----------------
        if char_tags:
            datasets = [('input.txt', 'rnn_input', 'lstm_input'),
                        ('cnus-clean.txt', 'rnn_cnus', 'lstm_cnus')]
            datasets = [d for d in datasets if d[1] in summaries and d[2] in summaries]
            if datasets:
                fig, axes = plt.subplots(1, len(datasets), figsize=(12, 4.6),
                                         squeeze=False)
                for ax, (name, rnn_tag, lstm_tag) in zip(axes[0], datasets):
                    for tag, style in ((rnn_tag, 'tab:blue'), (lstm_tag, 'tab:red')):
                        iters, losses = read_char_csv(
                            os.path.join(out_dir, tag + '_loss.csv'))
                        s = summaries[tag]
                        ax.plot(iters, losses, style, label='%s (%.1f s)' % (
                            'RNN' if 'rnn' in tag else 'LSTM',
                            s['training_seconds']))
                    ax.set_title('Quality on %s' % name)
                    ax.set_xlabel('iteration')
                    ax.set_ylabel('smoothed loss')
                    ax.legend()
                    ax.grid(alpha=0.3)
                fig.suptitle('Vanilla RNN vs LSTM — training quality and time')
                fig.tight_layout()
                pdf.savefig(fig)
                fig.savefig(os.path.join(out_dir, 'summary_quality.png'), dpi=150)
                plt.close(fig)

        # ---------------- Page 3: time-series predictions ----------------
        if torch_tags:
            for tag in torch_tags:
                s = load_summary(out_dir, tag)
                loss_png = os.path.join(out_dir, tag + '_loss.png')
                pred_png = os.path.join(out_dir, tag + '_prediction.png')
                fig = plt.figure(figsize=(11, 8))
                fig.text(0.03, 0.97,
                         'Временной ряд: %s (обучение %s)' % (
                             s.get('dataset_label', tag),
                             fmt_seconds(s['training_seconds'])),
                         fontsize=12, fontweight='bold', va='top')
                ax1 = fig.add_axes([0.06, 0.55, 0.90, 0.32])
                ax1.imshow(plt.imread(loss_png))
                ax1.axis('off')
                ax1.set_title('MSE по эпохам', fontsize=9)
                ax2 = fig.add_axes([0.06, 0.08, 0.90, 0.42])
                ax2.imshow(plt.imread(pred_png))
                ax2.axis('off')
                pdf.savefig(fig)
                plt.close(fig)

        # ---------------- Page 4: conclusions ----------------
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(0.04, 0.95, 'Выводы', fontsize=15, fontweight='bold', va='top')
        lines = []
        if char_tags:
            for tag in char_tags:
                s = summaries[tag]
                lines.append(
                    '%-12s %-12s: %5.1f s, %d ит., итоговый loss %.3f' % (
                        'RNN' if 'rnn' in tag else 'LSTM',
                        os.path.basename(s['dataset']),
                        s['training_seconds'], s['iterations'],
                        s['final_smooth_loss']))
            if all(t in summaries for t in ('rnn_input', 'lstm_input')):
                r, l = summaries['rnn_input'], summaries['lstm_input']
                lines.append('')
                lines.append('input.txt: LSTM медленнее RNN в %.1f раза (%.1f s vs %.1f s).'
                             % (l['training_seconds'] / r['training_seconds'],
                                l['training_seconds'], r['training_seconds']))
            if all(t in summaries for t in ('rnn_cnus', 'lstm_cnus')):
                r, l = summaries['rnn_cnus'], summaries['lstm_cnus']
                lines.append('cnus-clean.txt: LSTM медленнее RNN в %.1f раза (%.1f s vs %.1f s).'
                             % (l['training_seconds'] / r['training_seconds'],
                                l['training_seconds'], r['training_seconds']))
            lines.append('')
            lines.append('По кривым видно, что LSTM снижает loss быстрее и глубже: '
                         'аддитивное обновление памяти c(t) сохраняет градиент '
                         'на длинных последовательностях, тогда как RNN страдает '
                         'от его затухания.')
        if torch_tags:
            lines.append('')
            for tag in torch_tags:
                s = load_summary(out_dir, tag)
                lines.append('%s: RMSE train %.2f, test %.2f (исходные единицы), '
                             'обучение %s.' % (
                                 s.get('dataset_label', tag),
                                 s['train_rmse_original_units'],
                                 s['test_rmse_original_units'],
                                 fmt_seconds(s['training_seconds'])))
            lines.append('')
            lines.append('PyTorch-LSTM уверенно воспроизводит ряд на train-части; '
                         'на test-части ошибка выше, что ожидаемо для такой '
                         'маленькой сети (hidden_size=2).')
        y = 0.88
        for line in lines:
            wrapped = textwrap.fill(line, 112) if line else ''
            fig.text(0.04, y, wrapped, fontsize=9.5, family='monospace', va='top')
            y -= 0.035 * (wrapped.count('\n') + 1) + 0.008
        pdf.savefig(fig)
        plt.close(fig)

    print('Saved summary: %s' % pdf_path)
    if show:
        plt.show()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='Run all Lab 3 experiments')
    ap.add_argument('--out-dir', default='results')
    ap.add_argument('--max-data', type=int, default=1000000)
    ap.add_argument('--epochs', type=int, default=2000)
    ap.add_argument('--no-char', action='store_true')
    ap.add_argument('--no-torch', action='store_true')
    ap.add_argument('--skip-existing', action='store_true',
                    help='skip runs whose <tag>_summary.json already exists')
    ap.add_argument('--show', action='store_true')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = args.out_dir
    log_dir = os.path.join(out_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    char_tags, torch_tags = [], []
    t_all = time.perf_counter()

    if not args.no_char:
        extra = ['--max-data', str(args.max_data)]
        for script, tag, sargs in CHAR_RUNS:
            run_one(script, tag, sargs, out_dir, log_dir, extra,
                    skip_existing=args.skip_existing)
            char_tags.append(tag)

    if not args.no_torch:
        extra = ['--epochs', str(args.epochs)]
        for script, tag, sargs in TORCH_RUNS:
            run_one(script, tag, sargs, out_dir, log_dir, extra,
                    skip_existing=args.skip_existing)
            torch_tags.append(tag)

    print('\nAll runs finished in %.1f s' % (time.perf_counter() - t_all))
    build_summary(out_dir, char_tags, torch_tags, show=args.show)
    return 0


if __name__ == '__main__':
    sys.exit(main())
