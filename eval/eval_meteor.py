# Author: Dmytro Kalpakchi
# before running the file, make sure that nlg-eval virtualenv is running

from __future__ import division

import atexit
import logging
import os
import re
import subprocess
import threading

import psutil

import inspect

import nlgeval


METEOR_JAR = 'meteor-1.5.jar'


class MultilingualMeteor(nlgeval.Meteor):
    def __init__(self, lang='other', norm=False):
        # Used to guarantee thread safety
        self.lock = threading.Lock()

        mem = '2G'
        mem_available_G = psutil.virtual_memory().available / 1E9
        if mem_available_G < 2:
            logging.warning("There is less than 2GB of available memory.\n"
                            "Will try with limiting Meteor to 1GB of memory but this might cause issues.\n"
                            "If you have problems using Meteor, "
                            "then you can try to lower the `mem` variable in meteor.py")
            mem = '1G'

        meteor_cmd = ['java', '-jar', '-Xmx{}'.format(mem), METEOR_JAR,
                      '-', '-', '-stdio', '-l', lang]
        if norm:
            meteor_cmd.append('-norm')
        env = os.environ.copy()
        env['LC_ALL'] = "C"
        self.meteor_p = subprocess.Popen(meteor_cmd,
                                         cwd=os.path.dirname(os.path.abspath(inspect.getmodule(nlgeval.Meteor).__file__)),
                                         env=env,
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
        atexit.register(self.close)


if __name__ == '__main__':
    import click

    def _strip(s):
        return s.strip()

    @click.command()
    @click.option('--lang', type=str, required=True, help='Evaluation language')
    @click.option('--norm', is_flag=True, help='Flag. If provided, normalization will be applied')
    @click.option('--references', type=click.Path(exists=True), multiple=True, required=True, help='Path of the reference file. This option can be provided multiple times for multiple reference files.')
    @click.option('--hypothesis', type=click.Path(exists=True), required=True, help='Path of the hypothesis file.')
    def compute_metrics(lang, norm, hypothesis, references):
        """
        Compute nlg-eval metrics.
        The --hypothesis and at least one --references parameters are required.
        To download the data and additional code files, use `nlg-eval --setup [data path]`.
        Note that nlg-eval also features an API, which may be easier to use.
        """
        with open(hypothesis, 'r') as f:
            hyp_list = f.readlines()
        ref_list = []
        for iidx, reference in enumerate(references):
            with open(reference, 'r') as f:
                ref_list.append(f.readlines())
        ref_list = [list(map(_strip, refs)) for refs in zip(*ref_list)]
        refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
        hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
        assert len(refs) == len(hyps)

        ret_scores = {}
        scorers = [
            (MultilingualMeteor(lang, norm), "METEOR"),
        ]
        for scorer, method in scorers:
            score, scores = scorer.compute_score(refs, hyps)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.6f" % (m, sc))
                    ret_scores[m] = sc
            else:
                print("%s: %0.6f" % (method, score))
                ret_scores[method] = score
            if isinstance(scorer, MultilingualMeteor):
                scorer.close()
        del scorers

    compute_metrics()