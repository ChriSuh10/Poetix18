from py_files.Limericks_new2 import Limerick_Generate_new
from gpt2.src.score import score_model
import numpy as np
import random
from functools import reduce


lg = Limerick_Generate_new()


def preprocess(lg, prompt):
    random.seed(0)
    w1s_rhyme_filtered_dict, w3s_rhyme_filtered_dict = lg.get_two_sets_filtered_henry(prompt)

    random.seed(0)
    storyline = ["", "", "", "", ""]
    storyline[0] = random.choice([k for k in w1s_rhyme_filtered_dict.keys() if k.lower() in reduce(lambda x, y: x + y, lg.load_name_list()) and
                                  len(lg.dict_meters[k.lower()]) == 1 and len(lg.dict_meters[k.lower()][0]) in [1, 2]])
    storyline[1], storyline[4] = random.sample([k for k in w1s_rhyme_filtered_dict[storyline[0]] if len(lg.dict_meters[k.lower()]) == 1 and
                                                len(lg.dict_meters[k.lower()][0]) in [1, 2]], 2)
    storyline[2] = random.choice([k for k in w3s_rhyme_filtered_dict.keys() if len(lg.dict_meters[k.lower()]) == 1 and
                                  len(lg.dict_meters[k.lower()][0]) in [1, 2]])
    storyline[3] = random.choice([k for k in w3s_rhyme_filtered_dict[storyline[2]] if len(lg.dict_meters[k.lower()]) == 1 and
                                  len(lg.dict_meters[k.lower()][0]) in [1, 2]])

    first_line = random.choice(lg.gen_first_line_new(storyline[0].lower(), strict=True))

    syllables = [9, 6, 6, 9]
    lines = ["second", "third", "fourth", "fifth"]

    random.seed(0)
    templates = []
    for i, (syll, line) in enumerate(zip(syllables, lines)):
        template = list(random.choice(lg.get_all_templates(syll, line, [storyline[i + 1]])))
        for i in range(len(template)):
            if template[i] not in lg.pos_to_words:
                template[i] = lg.words_to_pos[template[i].lower()][0]
        templates.append(template)

    return storyline, first_line, templates


def gpt2_gen_next_word(tag, previous_sentence):
    logits = score_model(model_name="345M", context_token=np.array(lg.enc.encode(" ".join(previous_sentence))).reshape(1, -1)).flatten()
    sorted_index = np.argsort(- logits)
    for i in sorted_index:
        w = lg.enc.decode([i])
        if tag in lg.words_to_pos[w]:
            return w


def madlib_gen_next_word(tag, end_word):
    return lg.get_similar_word_henry([end_word], word_set=lg.pos_to_words[tag])[0]


def generate_line(template, previous_sentence, end_word):
    line = []
    if lg.words_to_pos[end_word][:2] != "VB":
        flag = 1
    for tag in template[:-1]:
        if flag and tag[:2] == "VB":
            line.append(madlib_gen_next_word(tag, end_word))
            flag = 0
        else:
            line.append(gpt2_gen_next_word(tag, previous_sentence + line))
        print(line)
    line.append(end_word)
    print(line)
    return line


def generate_poem(lg, prompt):
    storyline, first_line, templates = preprocess(lg, prompt)

    second_line = generate_line(templates[0], first_line + [","], storyline[1])
    third_line = generate_line(templates[1], first_line + [","] + second_line + ["."], storyline[2])
    fourth_line = generate_line(templates[2], first_line + [","] + second_line + ["."] + third_line + ["."], storyline[3])
    fifth_line = generate_line(templates[3], first_line + [","] + second_line + ["."] + third_line + ["."] + fourth_line + ["."], storyline[4])

    return first_line, second_line, third_line, fourth_line, fifth_line


def write_poem(lg, prompt):
    with open("madlibs-gpt2-experiments-results.txt", "a+") as hf:
        first_line, second_line, third_line, fourth_line, fifth_line = generate_poem(lg, prompt)
        print(first_line)
        print(second_line)
        print(third_line)
        print(fourth_line)
        print(fifth_line)
        hf.write("=" * 10 + " " + prompt + " " + "=" * 10 + "\n")
        hf.write(" ".join(first_line) + "\n")
        hf.write(" ".join(second_line) + "\n")
        hf.write(" ".join(third_line) + "\n")
        hf.write(" ".join(fourth_line) + "\n")
        hf.write(" ".join(fifth_line) + "\n")


prompt = "spring"
write_poem(lg, prompt)
