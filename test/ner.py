from nltk.tree import Tree
from nltk.chunk import conlltags2tree
import phonlp
def get_entities(nlp, s):
    all_entities_pyramid = {}
    all_entities = {}
    ent_list=[]
    original_label=[]
    all_entities_cur = set()
    try:
        sent = nlp.annotate(s)
        original_label,ent_list = out_phoNLP(sent)
        if len(ent_list) != 0:
            for ent in ent_list:
                all_entities_cur.add(ent)
                all_entities[ent] = all_entities.get(ent, 0) + 1
    except Exception:
        print("Error")
    for e in all_entities_cur:
        all_entities_pyramid[e] = all_entities_pyramid.get(e, 0) + 1
    res={}
    for (name,label) in all_entities.keys():
        res[name]=label
    return all_entities_pyramid, all_entities,res

def out_phoNLP(phonlp_annot):
  tokens = phonlp_annot[0][0]
  ners = phonlp_annot[2][0]
  pos_tags = []
  for i in range(len(tokens)):
    pos_tags.append(phonlp_annot[1][0][i][0])

  conlltags = [(token, pos, ner) for token, pos, ner in zip(tokens, pos_tags, ners)]
  ne_tree = conlltags2tree(conlltags)
  original_text, entity = [], []
  for subtree in ne_tree:
      # skipping 'O' tags
      if type(subtree) == Tree:
          original_label = subtree.label()
          original_string = " ".join([token for token, pos in subtree.leaves()])
          entity.append(original_string)
          if len(original_string)!=0:
            original_text.append((original_string, original_label))
        #   original_text.append((original_string, original_label))
  
  return entity,original_text

