
import pprint
from enum import Enum
import copy 
class RHETORICAL_FIGURES(Enum):
    ANADIPLOSIS = 0
    ANAPHORA = 1
    ANTIMETABOLE = 2
    CONDUPLICATIO = 3
    EPANALEPSIS = 4
    EPISTROPHE = 5
    EPIZEUXIS = 6
    PLOCE = 7
    POLYSYNDETON = 8
    SYMPLOCE = 9


# FIGURE_DESCRIPTION[RHETORICAL_FIGURES['ANADIPLOSIS'].name].value
class FIGURE_DESCRIPTION(Enum):
    ANADIPLOSIS = 'Repetition of the ending word or phrase from the previous clause at the beginning of the next.'
    ANAPHORA = 'Repetition of a word or phrase at the beginning of successive phrases or clauses.'
    ANTIMETABOLE = 'Repetition of words in reverse grammatical order.'
    CONDUPLICATIO = 'The repetition of a word or phrase.'
    EPANALEPSIS = 'Repetition at the end of a clause of the word or phrase that began it.'
    EPISTROPHE = 'Repetition of the same word or phrase at the end of successive clauses.'
    EPIZEUXIS = 'Repetition of a word or phrase with no others between.'
    PLOCE = 'The repetition of word in a short span of text for rhetorical emphasis.'
    POLYSYNDETON = '"Excessive" repetition of conjunctions between clauses.'
    SYMPLOCE = 'Repetition of a word or phrase at the beginning, and of another at the end, of successive clauses.'


class FigureInfo:

    def __init__(self, unique_id, orig_text, pos_template, repitition_template):
        self.unique_id = unique_id
        self.orig_text = orig_text
        self.pos_template = pos_template
        self.repitition_template = repitition_template

    def set_props(self, num_lines, num_tokens, num_rep_groups, num_tot_reps, fig_type):
        self.num_lines = num_lines
        self.num_tokens = num_tokens
        self.num_rep_groups = num_rep_groups
        self.num_tot_reps = num_tot_reps
        self.fig_type = fig_type

    def set_orig_rep_words(self, orig_rep_words):
        self.orig_rep_words = orig_rep_words

    def set_orig_tokens(self, orig_tokens):
        self.orig_tokens = orig_tokens

    def get_orig_tokens(self):
        return self.orig_tokens

    def set_gen_lines(self, gen_lines):
        self.gen_lines = gen_lines

    def get_gen_lines(self):
        return self.gen_lines

    def get_orig_rep_words(self):
        return self.orig_rep_words

    def get_unique_id(self):
        return self.unique_id

    def get_orig_text(self):
        return self.orig_text

    def get_pos_template(self):
        return self.pos_template

    def get_repitition_template(self):
        return self.repitition_template

    def get_num_lines(self):
        return self.num_lines

    def get_num_tokens(self):
        return self.num_tokens

    def get_num_rep_groups(self):
        return self.num_rep_groups

    def get_num_tot_reps(self):
        return self.num_tot_reps

    def get_fig_type(self):
        return self.fig_type

    def get_fig_desc(self):
        return FIGURE_DESCRIPTION[self.get_fig_type()].value

    def to_string(self):
        pp = pprint.PrettyPrinter(indent=4, width=160)
        obj = copy.deepcopy(self.__dict__)
        del obj['orig_tokens']
        return pprint.pformat(obj) + "\n"

    def to_string_sparse(self):
        pp = pprint.PrettyPrinter(indent=4, width=160)
        obj = {'unique_id': self.get_unique_id(), 'orig_text': self.get_orig_text(), 'gen_lines': self.get_gen_lines()}
        return pprint.pformat(obj) + "\n"

    def print(self):
        pp = pprint.PrettyPrinter(indent=4, width=160)
        obj = copy.deepcopy(self.__dict__)
        del obj['orig_tokens']
        pp.pprint(obj)

        print("\n")
