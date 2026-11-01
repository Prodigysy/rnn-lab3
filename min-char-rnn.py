"""
Minimal character-level Vanilla RNN language model.

Structure follows Andrej Karpathy's min-char-rnn gist:
    https://gist.github.com/karpathy/d4de56e743c9e7a3b0b3
The forward/backward equations are unchanged; see the comments below.

Lab 3 additions (see методичка, п. 3.2):
  * total training time measurement;
  * quality metric (loss) logged to CSV during training;
  * loss-vs-iteration plot saved to PNG and PDF;
  * generated samples saved to a text file and to a per-run PDF;
  * command-line arguments, so the same script can be run on each dataset
    (data/input.txt, data/cnus-clean.txt).

Usage:
    python min-char-rnn.py                       # default: data/cnus-clean.txt
    python min-char-rnn.py --file data/input.txt --tag rnn_input
    python min-char-rnn.py --max-data 200000     # shorter run
    python min-char-rnn.py --grad-check          # numerical gradient check only
"""
from __future__ import print_function

import argparse
import csv
import json
import os
import sys
import textwrap
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless: we save figures instead of showing them
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ----------------------------------------------------------------------------
# Defaults (the original script's hyperparameters)
# ----------------------------------------------------------------------------
DEFAULT_FILE = os.path.join('data', 'cnus-clean.txt')
DEFAULT_HIDDEN = 100        # size of hidden layer of neurons
DEFAULT_SEQ_LENGTH = 16     # number of steps to unroll the RNN for
DEFAULT_LR = 1e-1
DEFAULT_MAX_DATA = 1000000  # stop when processed this much data
DEFAULT_SAMPLE_EVERY = 1000
DEFAULT_LOG_EVERY = 200


# ----------------------------------------------------------------------------
# Data and model parameters
# ----------------------------------------------------------------------------
def load_corpus(filename):
    """Read the training text and build the character vocabulary."""
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        data = f.read()

    chars = sorted(set(data))
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    return data, chars, char_to_ix, ix_to_char


def init_params(hidden_size, vocab_size, rng):
    """Model parameters -- shared among all time steps.

    Weights are initialized randomly; biases are initialized to 0.
    Inputs are characters one-hot encoded in a vocab-sized vector.
    Dimensions: H = hidden_size, V = vocab_size
    """
    return {
        'Wxh': rng.randn(hidden_size, vocab_size) * 0.01,   # input to hidden
        'Whh': rng.randn(hidden_size, hidden_size) * 0.01,  # hidden to hidden
        'Why': rng.randn(vocab_size, hidden_size) * 0.01,   # hidden to output
        'bh': np.zeros((hidden_size, 1)),                   # hidden bias
        'by': np.zeros((vocab_size, 1)),                    # output bias
    }


# ----------------------------------------------------------------------------
# Forward + backward pass (BPTT)
# ----------------------------------------------------------------------------
def lossFun(inputs, targets, hprev, p, vocab_size):
    """Runs forward and backward passes through the RNN.

    inputs, targets: Lists of integers. For some i, inputs[i] is the input
                     character (encoded as an index into the ix_to_char map)
                     and targets[i] is the corresponding next character in the
                     training data (similarly encoded).
    hprev: Hx1 array of initial hidden state
    p: dict with the model parameters (Wxh, Whh, Why, bh, by)
    returns: loss, gradients on model parameters, and last hidden state
    """
    Wxh, Whh, Why, bh, by = p['Wxh'], p['Whh'], p['Why'], p['bh'], p['by']

    # Caches that keep values computed in the forward pass at each time step,
    # to be reused in the backward pass.
    xs, hs, ys, ps = {}, {}, {}, {}

    # Initial incoming state.
    hs[-1] = np.copy(hprev)
    loss = 0

    # Forward pass
    for t in range(len(inputs)):
        # Input at time step t is xs[t]. Prepare a one-hot encoded vector of
        # shape (V, 1). inputs[t] is the index where the 1 goes.
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1

        # Compute h[t] from h[t-1] and x[t]
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)

        # Compute ps[t] - softmax probabilities for output.
        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))

        # Cross-entropy loss. targets[t] is 1-hot encoded implicitly, so the
        # sum over the vocabulary collapses to -log(p of the right character).
        loss += -np.log(ps[t][targets[t], 0])

    # Backward pass: compute gradients going backwards.
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)

    # Incoming gradient of h is 0 for truncated BPTT.
    dhnext = np.zeros_like(hs[0])

    # The backwards pass iterates over the input sequence backwards.
    for t in reversed(range(len(inputs))):
        # Backprop through the gradients of loss and softmax.
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1

        # Gradients for the Why and by parameters.
        dWhy += np.dot(dy, hs[t].T)
        dby += dy

        # Backprop to h; add the incoming gradient from the next cell.
        dh = np.dot(Why.T, dy) + dhnext

        # Backprop through tanh.
        dhraw = (1 - hs[t] * hs[t]) * dh

        # Gradients for the dWxh, Whh, bh parameters.
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)

        # Backprop the gradient to the incoming h.
        dhnext = np.dot(Whh.T, dhraw)

    grads = {'Wxh': dWxh, 'Whh': dWhh, 'Why': dWhy, 'bh': dbh, 'by': dby}

    # Gradient clipping to the range [-5, 5].
    for dparam in grads.values():
        np.clip(dparam, -5, 5, out=dparam)

    return loss, grads, hs[len(inputs)-1]


def sample(h, seed_ix, n, p, vocab_size):
    """Sample a sequence of integers from the model.

    Runs the RNN in forward mode for n steps; seed_ix is the seed letter for
    the first time step, and h is the memory state.
    """
    Wxh, Whh, Why, bh, by = p['Wxh'], p['Whh'], p['Why'], p['bh'], p['by']

    # Create a one-hot vector to represent the input.
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []

    for t in range(n):
        # Run the forward pass only.
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p_out = np.exp(y) / np.sum(np.exp(y))

        # Sample from the distribution produced by softmax.
        ix = np.random.choice(range(vocab_size), p=p_out.ravel())

        # Prepare input for the next cell.
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes


# ----------------------------------------------------------------------------
# Numerical gradient check (optional, --grad-check)
# ----------------------------------------------------------------------------
from random import uniform


def gradCheck(inputs, targets, hprev, p, vocab_size, num_checks=10, delta=1e-5):
    _, grads, _ = lossFun(inputs, targets, hprev, p, vocab_size)
    names = ['Wxh', 'Whh', 'Why', 'bh', 'by']
    for name in names:
        param, dparam = p[name], grads[name]
        s0, s1 = dparam.shape, param.shape
        assert s0 == s1, 'Error dims dont match: %s and %s.' % (s0, s1)
        print(name)
        for i in range(num_checks):
            ri = int(uniform(0, param.size))
            # evaluate cost at [x + delta] and [x - delta]
            old_val = param.flat[ri]
            param.flat[ri] = old_val + delta
            cg0, _, _ = lossFun(inputs, targets, hprev, p, vocab_size)
            param.flat[ri] = old_val - delta
            cg1, _, _ = lossFun(inputs, targets, hprev, p, vocab_size)
            param.flat[ri] = old_val  # reset old value for this parameter
            # fetch both numerical and analytic gradient
            grad_analytic = dparam.flat[ri]
            grad_numerical = (cg0 - cg1) / (2 * delta)
            rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
            print('%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
            # rel_error should be on order of 1e-7 or less


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train(p, data, char_to_ix, ix_to_char, hidden_size, seq_length, learning_rate,
          max_data, sample_every, log_every, sample_len=200, log_file=None,
          sample_file=None):
    """Sweeps through the corpus, training with truncated BPTT + Adagrad.

    Returns (history, samples, elapsed_seconds); `history` is a list of tuples
    (iteration, chars_seen, window_loss, smooth_loss).
    """
    vocab_size = len(char_to_ix)

    # n is the iteration counter; p_ptr is the input sequence pointer, at the
    # beginning of each step it points at the sequence used for training.
    n, p_ptr = 0, 0

    # Memory variables for Adagrad.
    mem = {name: np.zeros_like(val) for name, val in p.items()}
    smooth_loss = -np.log(1.0 / vocab_size) * seq_length

    history = []
    samples = []
    csv_writer = csv.writer(log_file) if log_file is not None else None

    t0 = time.perf_counter()

    while p_ptr < max_data:
        # Prepare inputs (we're sweeping from left to right in steps seq_length
        # long)
        if p_ptr + seq_length + 1 >= len(data) or n == 0:
            hprev = np.zeros((hidden_size, 1))  # reset RNN memory
            p_ptr = 0  # go from start of data

        # In each step we unroll the RNN for seq_length cells, and present it
        # with seq_length inputs and targets to learn.
        inputs = [char_to_ix[ch] for ch in data[p_ptr:p_ptr + seq_length]]
        targets = [char_to_ix[ch] for ch in data[p_ptr + 1:p_ptr + seq_length + 1]]

        # Sample from the model now and then.
        if n % sample_every == 0:
            sample_ix = sample(hprev, inputs[0], sample_len, p, vocab_size)
            print('---- \n Starting symbol >>> ', ix_to_char[inputs[0]])
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print('Generated text after symbol \n %s \n ----' % (txt,))
            samples.append((n, ix_to_char[inputs[0]], txt))
            if sample_file is not None:
                sample_file.write('==== iteration %d (chars seen %d), seed %r\n%s\n\n'
                                  % (n, p_ptr, ix_to_char[inputs[0]], txt))
                sample_file.flush()

        # Forward seq_length characters through the net and fetch gradient.
        loss, grads, hprev = lossFun(inputs, targets, hprev, p, vocab_size)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        if n % log_every == 0:
            history.append((n, p_ptr, loss, smooth_loss))
            print('iter %d (p=%d), loss: %f' % (n, p_ptr, smooth_loss))
            if csv_writer is not None:
                csv_writer.writerow([n, p_ptr, loss, smooth_loss])
                if log_file is not None:
                    log_file.flush()

        # Perform parameter update with Adagrad
        for name in p:
            dparam, mem_param = grads[name], mem[name]
            mem_param += dparam * dparam
            p[name] += -learning_rate * dparam / np.sqrt(mem_param + 1e-8)

        p_ptr += seq_length
        n += 1

    elapsed = time.perf_counter() - t0
    return history, samples, elapsed, n


# ----------------------------------------------------------------------------
# Saving results
# ----------------------------------------------------------------------------
def save_plots(history, path_png, path_pdf, title):
    iters = [row[0] for row in history]
    smooth = [row[3] for row in history]
    raw = [row[2] for row in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(iters, smooth, color='tab:blue')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('smoothed loss')
    ax1.set_title('Smoothed training loss')
    ax1.grid(alpha=0.3)

    ax2.plot(iters, raw, color='tab:orange', alpha=0.8)
    ax2.set_xlabel('iteration')
    ax2.set_ylabel('window loss')
    ax2.set_title('Loss on a single window')
    ax2.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    fig.savefig(path_pdf)
    plt.close(fig)


def save_report_pdf(path_pdf, history, samples, summary, title):
    """Per-run PDF: loss curve page + summary/samples page."""
    with PdfPages(path_pdf) as pdf:
        # Page 1: loss curve.
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot([r[0] for r in history], [r[3] for r in history],
                label='smoothed loss', color='tab:blue')
        ax.plot([r[0] for r in history], [r[2] for r in history],
                label='window loss', color='tab:orange', alpha=0.5)
        ax.set_xlabel('iteration')
        ax.set_ylabel('loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: run summary and generated samples.
        fig = plt.figure(figsize=(10, 10))
        fig.text(0.02, 0.98, title, fontsize=13, fontweight='bold', va='top')

        lines = ['%s: %s' % (k, v) for k, v in summary.items()]
        y = 0.93
        for line in lines:
            fig.text(0.02, y, line[:140], fontsize=9, family='monospace', va='top')
            y -= 0.025

        y -= 0.02
        fig.text(0.02, y, 'Generated samples:', fontsize=11, fontweight='bold', va='top')
        y -= 0.035
        picks = samples[:2] + samples[max(0, len(samples) // 2 - 1):len(samples) // 2 + 1] + samples[-1:]
        seen = set()
        for it, seed, txt in picks:
            if it in seen:
                continue
            seen.add(it)
            block = 'iteration %d (seed %r):\n%s\n' % (it, seed, txt.replace('\n', ' ')[:180])
            wrapped = textwrap.fill(block, 110)
            fig.text(0.02, y, wrapped, fontsize=7.5, family='monospace', va='top')
            y -= 0.045 + 0.011 * (wrapped.count('\n') + 1)
            if y < 0.05:
                break

        pdf.savefig(fig)
        plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='min-char-rnn (Lab 3)')
    ap.add_argument('--file', default=DEFAULT_FILE, help='training text file')
    ap.add_argument('--tag', default=None, help='output file prefix')
    ap.add_argument('--out-dir', default='results', help='output directory')
    ap.add_argument('--hidden', type=int, default=DEFAULT_HIDDEN)
    ap.add_argument('--seq-length', type=int, default=DEFAULT_SEQ_LENGTH)
    ap.add_argument('--lr', type=float, default=DEFAULT_LR)
    ap.add_argument('--max-data', type=int, default=DEFAULT_MAX_DATA)
    ap.add_argument('--sample-every', type=int, default=DEFAULT_SAMPLE_EVERY)
    ap.add_argument('--log-every', type=int, default=DEFAULT_LOG_EVERY)
    ap.add_argument('--sample-len', type=int, default=200)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--grad-check', action='store_true',
                    help='run numerical gradient check and exit')
    ap.add_argument('--show', action='store_true',
                    help='show the figures after training (interactive)')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    np.random.seed(args.seed)

    data, chars, char_to_ix, ix_to_char = load_corpus(args.file)
    data_size, vocab_size = len(data), len(chars)
    print('data has %d characters, %d unique.' % (data_size, vocab_size))
    print('char_to_ix', char_to_ix)
    print('ix_to_char', ix_to_char)

    p = init_params(args.hidden, vocab_size, np.random)

    if args.grad_check:
        inputs = [char_to_ix[ch] for ch in data[:args.seq_length]]
        targets = [char_to_ix[ch] for ch in data[1:args.seq_length + 1]]
        hprev = np.zeros((args.hidden, 1))
        print('Running numerical gradient check (relative error should be ~1e-7):')
        gradCheck(inputs, targets, hprev, p, vocab_size)
        return 0

    tag = args.tag or os.path.splitext(os.path.basename(args.file))[0] + '_rnn'
    os.makedirs(args.out_dir, exist_ok=True)

    csv_path = os.path.join(args.out_dir, tag + '_loss.csv')
    png_path = os.path.join(args.out_dir, tag + '_loss.png')
    pdf_path = os.path.join(args.out_dir, tag + '_loss.pdf')
    report_path = os.path.join(args.out_dir, tag + '_report.pdf')
    samples_path = os.path.join(args.out_dir, tag + '_samples.txt')
    summary_path = os.path.join(args.out_dir, tag + '_summary.json')

    t_start = time.perf_counter()

    with open(csv_path, 'w', newline='', encoding='utf-8') as log_file, \
            open(samples_path, 'w', encoding='utf-8') as sample_file:
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(['iteration', 'chars_seen', 'window_loss', 'smooth_loss'])
        history, samples, elapsed, n_iters = train(
            p, data, char_to_ix, ix_to_char,
            hidden_size=args.hidden,
            seq_length=args.seq_length,
            learning_rate=args.lr,
            max_data=args.max_data,
            sample_every=args.sample_every,
            log_every=args.log_every,
            sample_len=args.sample_len,
            log_file=log_file,
            sample_file=sample_file,
        )

    total = time.perf_counter() - t_start
    final_smooth = history[-1][3] if history else float('nan')

    summary = {
        'model': 'Vanilla RNN (min-char-rnn)',
        'dataset': args.file,
        'dataset_chars': data_size,
        'vocab_size': vocab_size,
        'hidden_size': args.hidden,
        'seq_length': args.seq_length,
        'learning_rate': args.lr,
        'max_data': args.max_data,
        'iterations': n_iters,
        'chars_processed': n_iters * args.seq_length,
        'training_seconds': round(elapsed, 3),
        'total_seconds': round(total, 3),
        'iterations_per_second': round(n_iters / elapsed, 3) if elapsed else None,
        'final_smooth_loss': round(float(final_smooth), 4),
        'initial_smooth_loss': round(-np.log(1.0 / vocab_size) * args.seq_length, 4),
        'log_every': args.log_every,
        'sample_every': args.sample_every,
    }

    title = 'Vanilla RNN — %s' % os.path.basename(args.file)
    save_plots(history, png_path, pdf_path, title)
    save_report_pdf(report_path, history, samples, summary, title)

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('\nTraining time: %.2f s (%.2f it/s, %.0f chars/s)'
          % (elapsed, summary['iterations_per_second'],
             summary['chars_processed'] / elapsed if elapsed else 0))
    print('Final smoothed loss: %.4f' % final_smooth)
    print('Saved: %s' % ', '.join([csv_path, png_path, pdf_path, report_path,
                                   samples_path, summary_path]))

    if args.show:
        plt.show()
    return 0


if __name__ == '__main__':
    sys.exit(main())
