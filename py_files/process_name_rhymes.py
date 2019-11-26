import pickle


with open("saved_objects/downloaded_names_rhymes.pkl", "rb") as hf:
    dic = pickle.load(hf)

lis = []
for k in dic:
    v = dic[k]
    for e in lis:
        if len(e[1].symmetric_difference(v)) <= 2:
            e[1].update(v)
            e[0].append(k)
            break
    else:
        lis.append(([k], v))
