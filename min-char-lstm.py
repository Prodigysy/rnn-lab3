# Minimal character-based language model learning with an LSTM architecture.
#
# Overall code structure based on Andrej Karpathy's min-char-rnn model:
#    https://gist.github.com/karpathy/d4de56e743c9e7a3b0b3
#
# But the architecture is modified to be LSTM rather than vanilla RNN.
# The companion blog post is:
#   https://eli.thegreenplace.net/2018/minimal-character-based-lstm-implementation/
#
# Tested with Python 3.6
#
# Eli Bendersky [https://eli.thegreenplace.net]
# BSD License per original (@karpathy)
#
# Lab 3 additions (see методичка, п. 3.2):
#   * total training time measurement;
#   * quality metric (loss) logged to CSV during training;
#   * loss-vs-iteration plot saved to PNG and PDF;
#   * generated samples saved to a text file and to a per-run PDF;
#   * command-line arguments, so the same script can be run on each dataset.
#
# Usage:
#     python min-char-lstm.py                    # default: data/cnus-clean.txt
#     python min-char-lstm.py --file data/input.txt --tag lstm_input
#     python min-char-lstm.py --max-data 200000  # shorter run
#     python min-char-lstm.py --grad-check       # numerical gradient check only
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
DEFAULT_HIDDEN = 100        # size of hidden state vectors; applies to h and c
DEFAULT_SEQ_LENGTH = 16     # number of steps to unroll the LSTM for
DEFAULT_LR = 0.1
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

    Inputs are characters one-hot encoded in a vocab-sized vector, so every
    gate receives the concatenation [x; h], whose height is H + V.
    Dimensions: H = hidden_size, V = vocab_size, HV = hidden_size + vocab_size
    """
    H, V = hidden_size, vocab_size
    HV = H + V
    return {
        'Wf': rng.randn(H, HV) * 0.01,
        'bf': np.zeros((H, 1)),
        'Wi': rng.randn(H, HV) * 0.01,
        'bi': np.zeros((H, 1)),
        'Wcc': rng.randn(H, HV) * 0.01,
        'bcc': np.zeros((H, 1)),
        'Wo': rng.randn(H, HV) * 0.01,
        'bo': np.zeros((H, 1)),
        'Wy': rng.randn(V, H) * 0.01,
        'by': np.zeros((V, 1)),
    }


def sigmoid(z):
    """Computes sigmoid function.

    z: array of float values.
    Returns array with values in the range [0, 1].
    """
    # (see the original implementation for the numerically stable variant)
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))


# ----------------------------------------------------------------------------
# Forward + backward pass (BPTT)
# ----------------------------------------------------------------------------
def lossFun(inputs, targets, hprev, cprev, p, V):
    """Runs forward and backward passes through the LSTM.

    inputs, targets: Lists of integers. For some i, inputs[i] is the input
                     character (encoded as an index into the ix_to_char map)
                     and targets[i] is the corresponding next character in the
                     training data (similarly encoded).
    hprev, cprev: Hx1 arrays of initial hidden and cell state.
    p: dict with the model parameters.
    returns: loss, dict of gradients, last hidden state, last cell state.
    """
    Wf, bf = p['Wf'], p['bf']
    Wi, bi = p['Wi'], p['bi']
    Wcc, bcc = p['Wcc'], p['bcc']
    Wo, bo = p['Wo'], p['bo']
    Wy, by = p['Wy'], p['by']

    H = Wf.shape[0]

    # Caches to be reused in the backward pass.
    xs, hs, cs, xhs = {}, {}, {}, {}
    fgs, igs, ogs = {}, {}, {}
    ccs, ys, ps = {}, {}, {}

    # Initial states.
    hs[-1] = np.copy(hprev)
    cs[-1] = np.copy(cprev)
    loss = 0

    # Forward pass
    for t in range(len(inputs)):
        # Input at time step t is xs[t]. Prepare a one-hot encoded vector of
        # shape (V, 1). inputs[t] is the index where the 1 goes.
        xs[t] = np.zeros((V, 1))
        xs[t][inputs[t]] = 1

        # hprev and xs[t] are column vectors; stack them together into a
        # "taller" column vector - first the elements of x, then h.
        xhs[t] = np.vstack((xs[t], hs[t-1]))

        # Gates f, i and o.
        fgs[t] = sigmoid(np.dot(Wf, xhs[t]) + bf)
        igs[t] = sigmoid(np.dot(Wi, xhs[t]) + bi)
        ogs[t] = sigmoid(np.dot(Wo, xhs[t]) + bo)

        # Candidate cc.
        ccs[t] = np.tanh(np.dot(Wcc, xhs[t]) + bcc)

        # This step's h and c.
        cs[t] = fgs[t] * cs[t-1] + igs[t] * ccs[t]
        hs[t] = np.tanh(cs[t]) * ogs[t]

        # Softmax for output.
        ys[t] = np.dot(Wy, hs[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))

        # Cross-entropy loss.
        loss += -np.log(ps[t][targets[t], 0])

    # Initialize gradients of all weights/biases to 0.
    grads = {name: np.zeros_like(val) for name, val in p.items()}

    # Incoming gradients for h and c; for truncated BPTT we assume they are 0.
    dhnext = np.zeros_like(hs[0])
    dcnext = np.zeros_like(cs[0])

    # The backwards pass iterates over the input sequence backwards.
    for t in reversed(range(len(inputs))):
        # Backprop through the gradients of loss and softmax.
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1

        # Gradients for the Wy and by parameters.
        grads['Wy'] += np.dot(dy, hs[t].T)
        grads['by'] += dy

        # Backprop through the fully-connected layer (Wy, by) to h. Also add
        # the incoming gradient for h from the next cell.
        dh = np.dot(Wy.T, dy) + dhnext

        # Backprop through multiplication with output gate; here "dtanh" means
        # the gradient at the output of tanh.
        dctanh = ogs[t] * dh
        # Backprop through the tanh function; since cs[t] branches in two
        # directions we add dcnext too.
        dc = dctanh * (1 - np.tanh(cs[t]) ** 2) + dcnext

        # Backprop through multiplication with the tanh; here "dhogs" means
        # the gradient at the output of the sigmoid of the output gate. Then
        # backprop through the sigmoid itself (ogs[t] is the sigmoid output).
        dhogs = dh * np.tanh(cs[t])
        dho = dhogs * ogs[t] * (1 - ogs[t])

        # Gradients for the output gate parameters.
        grads['Wo'] += np.dot(dho, xhs[t].T)
        grads['bo'] += dho

        # Backprop dho to the xh input.
        dxh_from_o = np.dot(Wo.T, dho)

        # Backprop through the forget gate: sigmoid and elementwise mul.
        dhf = cs[t-1] * dc * fgs[t] * (1 - fgs[t])
        grads['Wf'] += np.dot(dhf, xhs[t].T)
        grads['bf'] += dhf
        dxh_from_f = np.dot(Wf.T, dhf)

        # Backprop through the input gate: sigmoid and elementwise mul.
        dhi = ccs[t] * dc * igs[t] * (1 - igs[t])
        grads['Wi'] += np.dot(dhi, xhs[t].T)
        grads['bi'] += dhi
        dxh_from_i = np.dot(Wi.T, dhi)

        dhcc = igs[t] * dc * (1 - ccs[t] ** 2)
        grads['Wcc'] += np.dot(dhcc, xhs[t].T)
        grads['bcc'] += dhcc
        dxh_from_cc = np.dot(Wcc.T, dhcc)

        # Combine all contributions to dxh, and extract the gradient for the
        # h part to propagate backwards as dhnext.
        dxh = dxh_from_o + dxh_from_f + dxh_from_i + dxh_from_cc
        dhnext = dxh[V:, :]

        # dcnext from dc and the forget gate.
        dcnext = fgs[t] * dc

    # Gradient clipping to the range [-5, 5].
    for dparam in grads.values():
        np.clip(dparam, -5, 5, out=dparam)

    return loss, grads, hs[len(inputs)-1], cs[len(inputs)-1]


def sample(h, c, seed_ix, n, p, V):
    """Sample a sequence of integers from the model.

    Runs the LSTM in forward mode for n steps; seed_ix is the seed letter for
    the first time step, and h and c are the memory state.
    """
    Wf, bf = p['Wf'], p['bf']
    Wi, bi = p['Wi'], p['bi']
    Wcc, bcc = p['Wcc'], p['bcc']
    Wo, bo = p['Wo'], p['bo']
    Wy, by = p['Wy'], p['by']

    x = np.zeros((V, 1))
    x[seed_ix] = 1
    ixes = []

    for t in range(n):
        # Run the forward pass only.
        xh = np.vstack((x, h))
        fg = sigmoid(np.dot(Wf, xh) + bf)
        ig = sigmoid(np.dot(Wi, xh) + bi)
        og = sigmoid(np.dot(Wo, xh) + bo)
        cc = np.tanh(np.dot(Wcc, xh) + bcc)
        c = fg * c + ig * cc
        h = np.tanh(c) * og
        y = np.dot(Wy, h) + by
        p_out = np.exp(y) / np.sum(np.exp(y))

        # Sample from the distribution produced by softmax.
        ix = np.random.choice(range(V), p=p_out.ravel())
        x = np.zeros((V, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes


# ----------------------------------------------------------------------------
# Numerical gradient check (optional, --grad-check)
# ----------------------------------------------------------------------------
def gradCheck(inputs, targets, hprev, cprev, p, V, num_checks=10, delta=1e-5):
    loss, grads, _, _ = lossFun(inputs, targets, hprev, cprev, p, V)
    names = ['Wf', 'bf', 'Wi', 'bi', 'Wcc', 'bcc', 'Wo', 'bo', 'Wy', 'by']
    for name in names:
        param, dparam = p[name], grads[name]
        assert dparam.shape == param.shape
        print(name)
        for i in range(num_checks):
            ri = np.random.randint(0, param.size)
            old_val = param.flat[ri]
            param.flat[ri] = old_val + delta
            numloss0 = lossFun(inputs, targets, hprev, cprev, p, V)[0]
            param.flat[ri] = old_val - delta
            numloss1 = lossFun(inputs, targets, hprev, cprev, p, V)[0]
            param.flat[ri] = old_val  # reset
            grad_analytic = dparam.flat[ri]
            grad_numerical = (numloss0 - numloss1) / (2 * delta)
            if grad_numerical + grad_analytic == 0:
                rel_error = 0
            else:
                rel_error = (abs(grad_analytic - grad_numerical) /
                             abs(grad_numerical + grad_analytic))
            print('%s, %s => %e' % (grad_numerical, grad_analytic, rel_error))


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
    V = len(char_to_ix)
    H = hidden_size

    # n is the iteration counter; p_ptr is the input sequence pointer.
    n, p_ptr = 0, 0

    # Reset LSTM memory.
    hprev = np.zeros((H, 1))
    cprev = np.zeros((H, 1))

    # Memory variables for Adagrad.
    mem = {name: np.zeros_like(val) for name, val in p.items()}
    smooth_loss = -np.log(1.0 / V) * seq_length

    history = []
    samples = []
    csv_writer = csv.writer(log_file) if log_file is not None else None

    t0 = time.perf_counter()

    while p_ptr < max_data:
        # Prepare inputs (we're sweeping from left to right in steps seq_length
        # long)
        if p_ptr + seq_length + 1 >= len(data) or n == 0:
            # Reset LSTM memory
            hprev = np.zeros((H, 1))
            cprev = np.zeros((H, 1))
            p_ptr = 0  # go from start of data

        # In each step we unroll the LSTM for seq_length cells.
        inputs = [char_to_ix[ch] for ch in data[p_ptr:p_ptr + seq_length]]
        targets = [char_to_ix[ch] for ch in data[p_ptr + 1:p_ptr + seq_length + 1]]

        # Sample from the model now and then.
        if n % sample_every == 0:
            sample_ix = sample(hprev, cprev, inputs[0], sample_len, p, V)
            print('---- \n Starting symbol >>> ', ix_to_char[inputs[0]])
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print('Generated text after symbol \n %s \n ----' % (txt,))
            samples.append((n, ix_to_char[inputs[0]], txt))
            if sample_file is not None:
                sample_file.write('==== iteration %d (chars seen %d), seed %r\n%s\n\n'
                                  % (n, p_ptr, ix_to_char[inputs[0]], txt))
                sample_file.flush()

        # Forward seq_length characters through the net and fetch gradient.
        loss, grads, hprev, cprev = lossFun(inputs, targets, hprev, cprev, p, V)
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
    ap = argparse.ArgumentParser(description='min-char-lstm (Lab 3)')
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
        hprev = np.random.randn(args.hidden, 1)
        cprev = np.random.randn(args.hidden, 1)
        print('Running numerical gradient check (relative error should be ~1e-7):')
        gradCheck(inputs, targets, hprev, cprev, p, vocab_size)
        return 0

    tag = args.tag or os.path.splitext(os.path.basename(args.file))[0] + '_lstm'
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
        'model': 'LSTM (min-char-lstm)',
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

    title = 'LSTM — %s' % os.path.basename(args.file)
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
