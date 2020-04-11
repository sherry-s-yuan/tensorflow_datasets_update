# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2019 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Targeted syntactic evaluation templates."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds

# C:\Users\15475\AppData\Local\Programs\Python\Python37\Lib\site-packages\tensorflow_datasets
_CITATION = """
@article{DBLP:journals/corr/abs-1808-09031,
  author    = {Rebecca Marvin and
               Tal Linzen},
  title     = {Targeted Syntactic Evaluation of Language Models},
  journal   = {CoRR},
  volume    = {abs/1808.09031},
  year      = {2018},
  url       = {http://arxiv.org/abs/1808.09031},
  archivePrefix = {arXiv},
  eprint    = {1808.09031},
  timestamp = {Mon, 03 Sep 2018 13:36:40 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1808-09031.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """ \
Data Description
This data is purely used to test the accuracy of a language model. \
It is a collection of tasks that evaluate language models along \
three different structure-sensitive linguistic phenomena: \
subject-verb agreement, reflexive anaphora and negative \
polarity items. Given a large number of minimally different \
pairs of English sentences, each consisting of a grammatical \
and an ungrammatical sentence (where the first sentence of each \
pair has the correct grammar), a language model should assign \
a higher probability to a grammatical sentence than an ungrammatical one. 

Origin
Rebecca Marvin and Tal Linzen. Targeted syntactic evaluation of language models. arXiv preprint
arXiv:1808.09031, 2018.

Data can be found here: https://github.com/BeckyMarvin/LM_syneval
"""


class Syneval(tfds.core.GeneratorBasedBuilder):
  """A dataset/template for targeted syntactic evaluation. \
  Each observation consisting of a grammatical and an ungrammatical \
  sentence, where the first sentence in each pair has the correct grammar."""

  VERSION = tfds.core.Version('0.1.0')
  MANUAL_DOWNLOAD_INSTRUCTIONS = """Dataset will be manually \
                                 generated in in there."""

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            # These are the features of the dataset like images, labels ...
            # describe type of tests: subject-verb agreement, reflexive anaphora etc
            "test_case": tfds.features.Text(),
            # the text of grammatically correct sentence
            "grammatical": tfds.features.Text(),
            # the text of grammatically incorrect sentence
            "ungrammatical": tfds.features.Text()
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        # supervised_keys=(),
        # Homepage of the dataset for documentation
        homepage='https://github.com/BeckyMarvin/LM_syneval',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    # download data into manual path
    download_data(dl_manager)
    data_path = os.path.join(dl_manager.manual_dir, 'syneval.csv')
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={"archive": data_path},
        ),
    ]

  def _generate_examples(self, archive):
    """Yields examples."""
    with tf.io.gfile.GFile(archive) as f:
      for line_id, line in enumerate(f):
        # The format of the line is:
        # test_case, grammatical_sentence, ungrammatical_sentence
        line = line.split(',')
        yield line_id, {"test_case": line[0], 'grammatical': line[1], \
                        "ungrammatical": line[2]}


# Everything below is responsible for getting data
class MakeAgreementTemplate():
  """Construct template for grammar: agreement"""
  def __init__(self):
    self.terminals = AgreementTerminals().terminals
    self.rules = AgreementTemplate().rules

  def switch_number(self, wrds, verb=False):
    """switch the sentiment number"""
    new_wrds = []
    for wrd in wrds:
      if wrd.split()[0] == "is":
        new_wrds.append(' '.join(['are'] + wrd.split()[1:]))
      elif verb:
        if len(wrd.split()) > 1:
          new_wrds.append(' '.join([wrd.split()[0][:-1]] + wrd.split()[1:]))
        else:
          new_wrds.append(wrd[:-1])
      elif wrd[-4:] == "self":
        new_wrds.append("themselves")
      else:
        new_wrds.append(wrd + "s")
    return new_wrds

  def get_case_name(self, preterms, match, vary, opt='sing', v_opt='sing'):
    """get the name of the case"""
    sent = opt + "_"
    for j in range(len(match)):
      for i in range(len(match[j])):
        sent += preterms[match[j][i]] + "_"
    if len(vary) > 0:
      sent += v_opt + "_"
      for j in range(len(vary)):
        sent += preterms[vary[j]] + "_"
    return sent[:-1]

  def switch_numbers(self, base_sent, variables, preterms):
    """switch sentiment numbers"""
    new_sent = base_sent[:]
    for idx in variables:
      new_sent[idx] = self.switch_number(new_sent[idx], \
                                         preterms[idx][-1] == "V")
    return new_sent

  def make_variable_sents(self, preterms, match, vary):
    """make sentiment variable"""
    all_sents = {}
    base_sent = [self.terminals[p] for p in preterms]
    prefixes = ['sing', 'plur']
    for i in range(2):
      s_grammatical = base_sent[:]
      p_grammatical = self.switch_numbers(base_sent, vary, preterms)

      s_ungrammatical = self.switch_numbers(s_grammatical, match[1], preterms)
      p_ungrammatical = self.switch_numbers(p_grammatical, match[1], preterms)

      if i == 1:
        s_ungrammatical = self.switch_numbers(s_grammatical, match[0], preterms)
        p_ungrammatical = self.switch_numbers(p_grammatical, match[0], preterms)

        s_grammatical = self.switch_numbers(s_grammatical, \
                                            match[0] + match[1], preterms)
        p_grammatical = self.switch_numbers(p_grammatical, \
                                            match[0] + match[1], preterms)
      all_sents[self.get_case_name(preterms, match, vary, opt=prefixes[i],
                                   v_opt='sing')] = [s_grammatical,
                                                     s_ungrammatical]
      if len(vary) > 0:
        all_sents[self.get_case_name(preterms, match, vary, opt=prefixes[i], \
                                     v_opt='plur')] = [p_grammatical,
                                                       p_ungrammatical]

    return all_sents


class MakeNPITemplate():
  """make NPI templates"""
  def __init__(self):
    self.terminals = NPITerminals().terminals
    self.rules = NPITemplate().rules

  def switch_tense(self, preterms):
    """switch tense"""
    new_preterms = preterms[:]
    new_preterms[new_preterms.index('PASTAUX')] = 'FUTAUX'
    if 'APMV' in preterms:
      new_preterms[new_preterms.index('APMV')] = 'AFMV'
    else:
      new_preterms[new_preterms.index('IPMV')] = 'IFMV'
    return new_preterms

  def switch_dets(self, preterms, opt=''):
    """switch predeterminant"""
    new_preterms = preterms[:]
    if opt == 'intrusive':
      new_preterms[new_preterms.index('NO')] = 'SD'
    elif opt == 'ungram':
      new_preterms[new_preterms.index('NO')] = 'MOST'
    return new_preterms

  def make_variable_sents(self, preterms, simple=False):
    """make sentiment variables"""
    all_sents = {}
    prefixes = ['past', 'future']
    p_grammatical = [self.terminals[p] for p in preterms]
    f_grammatical = [self.terminals[p] for p in self.switch_tense(preterms)]

    p_intrusive = [self.terminals[p] for p in
                   self.switch_dets(preterms, opt='intrusive'
                                    if simple else '')]
    f_intrusive = [self.terminals[p] for p in
                   self.switch_tense(self.switch_dets(preterms, opt='intrusive'
                                                      if simple else ''))]

    p_ungrammatical = [self.terminals[p] for p in
                       self.switch_dets(preterms, opt='ungram')]
    f_ungrammatical = [self.terminals[p] for p in
                       self.switch_tense(self.switch_dets(
                           preterms, opt='ungram'))]

    all_sents['past'] = [p_grammatical, p_intrusive, p_ungrammatical]
    all_sents['future'] = [f_grammatical, f_intrusive, f_ungrammatical]

    return all_sents


class MakeTestCase():
  """make test cases"""
  def __init__(self, template, test_case):
    self.template = template
    self.test_case = test_case
    self.sent_templates = self.get_rules()

  def get_rules(self):
    """get a set of rules for generating sentiment"""
    sent_templates = {}
    preterminals, templates = self.template.rules[self.test_case]
    if templates is not None:
      sents = self.template.make_variable_sents(preterminals,
                                                templates['match'],
                                                templates['vary'])
      for k in sents.keys():
        if k not in sent_templates:
          sent_templates[k] = []
        gram = list(self.expand_sent(sents[k][0]))
        ungram = list(self.expand_sent(sents[k][1]))
        for i in range(len(gram)):
          sent_templates[k].append((gram[i], ungram[i]))
    else:
      sents = self.template.make_variable_sents(preterminals,
                                                simple=self.test_case.
                                                startswith('simple'))
      for k in sents.keys():
        if k not in sent_templates:
          sent_templates[k] = []
        gram = list(self.expand_sent(sents[k][0]))
        intrusive = list(
            self.expand_sent(sents[k][1], partial="",
                             switch_ds=not self.test_case.startswith('simple')))
        ungram = list(self.expand_sent(sents[k][2]))
        for i in range(len(gram)):
          sent_templates[k].append((gram[i], intrusive[i], ungram[i]))
    return sent_templates

  def expand_sent(self, sent, partial="", switch_ds=False):
    """expand sentiment"""
    if len(sent) == 1:
      for wrd in sent[0]:
        if switch_ds:
          sp = partial.split(" ")
          no = sp[0]
          the = sp[3]
          new_partial_one = ' '.join([x for x in partial.split()[1:3]])
          new_partial_two = ' '.join([x for x in partial.split()[4:]])
          yield ' '.join([the, new_partial_one, no, new_partial_two, wrd])
        # We want to avoid repeating words/phrases multiple times in the sentences
        # but some words are allowed to repeat, such as determiners or complementizers
        # We also need to check that the phrase isn't repeated save for number
        # e.g. 'the man who the guards like likes pizza'
        # not all sentences with repeating phrases are bad, but many seem implausible
        # so we do not generate them!
        elif wrd not in partial and wrd not in self.template.terminals['D'] \
            and wrd not in self.template.terminals['C'] and not (
                wrd.split(" ")[0] + "s " + ' '.join(wrd.split(" ")[1:])
                in partial
                or wrd.split(" ")[0][:-1] + " " + ' '.join(wrd.split(" ")[1:])
                in partial) \
            and not ((wrd.startswith('is') and 'are ' + wrd[3:] in partial)
                     or (wrd.startswith('are') and 'is ' + wrd[4:] in partial)):
          yield partial + wrd
        else:
          yield "None"
    else:
      for wrd in sent[0]:
        for x in self.expand_sent(sent=sent[1:], partial=partial + wrd + " ",
                                  switch_ds=switch_ds):
          if x != "None":
            yield x


# Template
class AgreementTemplate():
  def __init__(self):
    self.rules = {
        'obj_rel_across_anim': (['D', 'MS', 'C', 'D', 'ES', 'EV', 'MV'],
                                {'match': ([1], [6]), 'vary': [4, 5]}),
        'obj_rel_within_anim': (['D', 'MS', 'C', 'D', 'ES', 'EV', 'MV'],
                                {'match': ([4], [5]), 'vary': [1, 6]}),
        'obj_rel_across_inanim': (['D', 'IS', 'IC', 'D', 'ES', 'EV', 'IV'],
                                  {'match': ([1], [6]), 'vary': [4, 5]}),
        'obj_rel_within_inanim': (['D', 'IS', 'IC', 'D', 'ES', 'EV', 'IV'],
                                  {'match': ([4], [5]), 'vary': [1, 6]}),
        'subj_rel': (['D', 'MS', 'C', 'EV', 'D', 'ES', 'MV'],
                     {'match': ([1, 3], [6]), 'vary': [5]}),
        'prep_anim': (['D', 'MS', 'P', 'D', 'ES', 'MV'],
                      {'match': ([1], [5]), 'vary': [4]}),
        'prep_inanim': (['D', 'IS', 'IP', 'D', 'ES', 'IV'],
                        {'match': ([1], [5]), 'vary': [4]}),
        'obj_rel_no_comp_across_anim': (['D', 'MS', 'D', 'ES', 'EV', 'MV'],
                                        {'match': ([1], [5]), 'vary': [3, 4]}),
        'obj_rel_no_comp_within_anim': (['D', 'MS', 'D', 'ES', 'EV', 'MV'],
                                        {'match': ([3], [4]), 'vary': [1, 5]}),
        'obj_rel_no_comp_across_inanim': (['D', 'IS', 'D', 'ES', 'EV', 'IV'],
                                          {'match': ([1], [5]), 'vary': [3, 4]}
                                          ),
        'obj_rel_no_comp_within_inanim': (['D', 'IS', 'D', 'ES', 'EV', 'IV'],
                                          {'match': ([3], [4]), 'vary': [1, 5]}
                                          ),
        'simple_agrmt': (['D', 'MS', 'MV'], {'match': ([1], [2]), 'vary': []}),
        'sent_comp': (['D', 'BS', 'BV', 'D', 'MS', 'MV'],
                      {'match': ([4], [5]), 'vary': [1]}),
        'vp_coord': (['D', 'MS', 'MV', 'AND', 'MV'],
                     {'match': ([1, 2], [4]), 'vary': []}),
        'long_vp_coord': (['D', 'MS', 'LMV', 'AND', 'LMV'],
                          {'match': ([1, 2], [4]), 'vary': []}),
        'reflexives_across': (['D', 'MS', 'C', 'D', 'ES', 'EV', 'RMV', 'ANPHR'],
                              {'match': ([1], [7]), 'vary': [4, 5]}),
        'simple_reflexives': (['D', 'MS', 'RMV', 'ANPHR'],
                              {'match': ([1], [3]), 'vary': []}),
        'reflexive_sent_comp': (['D', 'BS', 'BV', 'D', 'MS', 'RMV', 'ANPHR'],
                                {'match': ([4], [6]), 'vary': [1]})
    }

    # TO CREATE NEW CONSTRUCTIONS, PLEASE FOLLOW THIS FORMAT:
    # 'name': ([list of preterminals], {dict containing ('match', 'vary') indices formatted as below})
    # {'match':([first indices (subject)], [second indices (verb/anaphor)]), 'vary':[list of indices for words to vary in number (attractors)},


class NPITemplate():
  def __init__(self):
    self.rules = {'npi_across_anim': (['NO', 'MS', 'C', 'D', 'ES', 'EV'
                                       , 'PASTAUX', 'NPI', 'APMV'], None),
                  'npi_across_inanim': (['NO', 'IS', 'C', 'D', 'ES', 'EV',
                                         'PASTAUX', 'NPI', 'IPMV'], None),
                  'simple_npi_anim': (['NO', 'MS', 'PASTAUX',
                                       'NPI', 'APMV'], None),
                  'simple_npi_inanim': (['NO', 'IS', 'PASTAUX',
                                         'NPI', 'IPMV'], None)}
    # TO CREATE NEW CONSTRUCTIONS, PLEASE FOLLOW THIS FORMAT:
    # 'name': ([list of preterminals], None)
    # For NPIs, we aren't looking at sentences that are minimally different so the 'match/vary' schema in the AgreementTemplate doesn't work here


# Terminal
class NPITerminals():
  def __init__(self):
    self.terminals = {'D': ['the'],
                      'SD': ['the',
                             'some'],
                      'IC': ['that'],
                      'C': ['that'],
                      'MOST': ['most',
                               'many'],
                      'NO': ['no',
                             'few'],
                      'MS': ['authors',
                             'pilots',
                             'surgeons',
                             'farmers',
                             'managers',
                             'customers',
                             'officers',
                             'teachers',
                             'senators',
                             'consultants'],
                      'ES': ['guards',
                             'chefs',
                             'architects',
                             'skaters',
                             'dancers',
                             'ministers',
                             'drivers',
                             'assistants',
                             'executives',
                             'parents'],
                      'IS': ['movies',
                             'books',
                             'games',
                             'songs',
                             'pictures',
                             'paintings',
                             'novels',
                             'poems',
                             'shows'],
                      'EV': ['like',
                             'admire',
                             'hate',
                             'love'],
                      'PASTAUX': ['have'],
                      'FUTAUX': ['will'],
                      'NPI': ['ever'],
                      'IPMV': ['been seen',
                               'been appreciated'
                               'been ignored',
                               'gotten old'],
                      'IFMV': ['be seen',
                               'be appreciated',
                               'be ignored',
                               'get old'],
                      'APMV': ['been popular',
                               'been famous',
                               'had children'],
                      'AFMV': ['be popular',
                               'be famous',
                               'have children'],
                      'MV': [],
                      'LMV': []}


class AgreementTerminals():
  def __init__(self):
    self.terminals = {'D': ['the'],
                      'IC': ['that'],
                      'C': ['that'],
                      'MS': ['author',
                             'pilot',
                             'surgeon',
                             'farmer',
                             'manager',
                             'customer',
                             'officer',
                             'teacher',
                             'senator',
                             'consultant'],
                      'ES': ['guard',
                             'chef',
                             'architect',
                             'skater',
                             'dancer',
                             'minister',
                             'taxi driver',
                             'assistant',
                             'executive',
                             'parent'],
                      'IS': ['movie',
                             'book',
                             'game',
                             'song',
                             'picture',
                             'painting',
                             'novel',
                             'poem',
                             'show'],
                      'MV': ['laughs',
                             'swims',
                             'smiles',
                             'is tall',
                             'is old',
                             'is young',
                             'is short'],
                      'EV': ['likes',
                             'admires',
                             'hates',
                             'loves'],
                      'IV': ['is good',
                             'is bad',
                             'is new',
                             'is popular',
                             'is unpopular',
                             'brings joy to people',
                             'interests people'],
                      'P': ['next to',
                            'behind',
                            'in front of',
                            'near',
                            'to the side of',
                            'across from'],
                      'IP': ['from',
                             'by'],
                      'BS': ['mechanic',
                             'banker'],
                      'BV': ['said',
                             'thought',
                             'knew'],
                      'AND': ['and'],
                      'ANPHR': ['himself',
                                'herself'],
                      'LMV': ['knows many different foreign languages',
                              'likes to watch television shows',
                              'is twenty three years old',
                              'enjoys playing tennis with colleagues',
                              'writes in a journal every day'],
                      'RMV': ['hurt',
                              'injured',
                              'congratulated',
                              'embarrassed',
                              'disguised',
                              'hated',
                              'doubted']}


# Test case
class TestCase():
  """class that construct test cases"""
  def __init__(self):
    """initialize test cases"""
    self.agrmt_cases = ['obj_rel_across_anim',
                        'obj_rel_within_anim',
                        'obj_rel_across_inanim',
                        'obj_rel_within_inanim',
                        'subj_rel',
                        'prep_anim',
                        'prep_inanim',
                        'obj_rel_no_comp_across_anim',
                        'obj_rel_no_comp_within_anim',
                        'obj_rel_no_comp_across_inanim',
                        'obj_rel_no_comp_within_inanim',
                        'simple_agrmt',
                        'sent_comp',
                        'vp_coord',
                        'long_vp_coord',
                        'reflexives_across',
                        'simple_reflexives',
                        'reflexive_sent_comp']

    self.npi_cases = ['npi_across_anim',
                      'npi_across_inanim',
                      'simple_npi_anim',
                      'simple_npi_inanim']

    self.all_cases = self.agrmt_cases + self.npi_cases


# main
def append_string(case, data):
  """add lines of the file as string first"""
  string = ''
  for key in data.keys():
    test_case = case + '--' + key
    for pair in data[key]:
      grammatical, ungrammatical = pair[0], pair[1]
      # print(test_case, grammatical, ungrammatical)
      string += test_case + ',' + grammatical + ',' + ungrammatical + '\n'
  return string


def download_data(dl_manager):
  """make template"""
  agrmt_template = MakeAgreementTemplate()
  npi_template = MakeNPITemplate()

  testcase = TestCase()
  string = ''
  agrmt_test_cases = testcase.agrmt_cases
  npi_test_cases = testcase.npi_cases

  path = os.path.join(dl_manager.manual_dir, 'syneval.csv')
  for case in agrmt_test_cases:
    print("case:", case)
    sents = MakeTestCase(agrmt_template, case)
    string += append_string(case, sents.sent_templates)
  for case in npi_test_cases:
    print("case:", case)
    sents = MakeTestCase(npi_template, case)
    string += append_string(case, sents.sent_templates)
  with open(path, 'w') as f:
    f.write(string)
