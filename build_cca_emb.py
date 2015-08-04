from cca.io import clean
from cca.io import set_quiet
from cca.strop import count_unigrams
from cca.strop import decide_vocab
from cca.strop import extract_stat
from cca.strop import rewrite_corpus
from cca.canon import canon
from cca.strop import count_unigrams

def main(corpus, out_file, cutoff, vocab_size, window, emb_size, kappa):
    unigrams = count_unigrams(corpus)
    vocab, outfname = decide_vocab(unigrams, cutoff, vocab_size, None)
    XYcount, Xcount, Ycount = extract_stat(args.corpus, vocab, outfname, window)

    C = canon()
    C.set_params(emb_size, kappa)
    C.load_stat(XYcount, Xcount, Ycount)
    C.approx_cca()
    C.write_result(out_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('corpus')
    parser.add_argument('out_file')
    parser.add_argument('--cutoff', default=0)
    parser.add_argument('--vocab_size', default=None)
    parser.add_argument('--window', default=5)
    parser.add_argument('--emb_size', default=170)
    parser.add_argument('--kappa', default=20)

    args = parser.parse_args()

    main(**vars(args))