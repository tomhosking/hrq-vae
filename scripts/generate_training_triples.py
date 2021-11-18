import jsonlines, os, json
import numpy as np

from flair.models import SequenceTagger
from flair.data import Sentence

from collections import defaultdict, Counter
from tqdm import tqdm
from copy import deepcopy


import torch
# predictor = Predictor.from_archive(load_archive("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz"))

import argparse

parser = argparse.ArgumentParser(
    description="WikiAnswers to 3way script",
)
parser.add_argument(
 "--data_dir", type=str, metavar="PATH", default='./data/', help="Path to data folder"
)

parser.add_argument(
 "--dataset", type=str, metavar="PATH", default='wa-triples', help="Source dataset"
)


parser.add_argument("--debug", action="store_true", help="Debug mode")
parser.add_argument("--pos_templates", action="store_true", help="Use POS tags for templating")
parser.add_argument("--constit_templates", action="store_true", help="Use constituency parses for templating")
parser.add_argument("--use_diff_templ_for_sem", action="store_true", help="Force different templates between tgt and sem input")
parser.add_argument("--use_stop_class", action="store_true", help="Convert stopwords to a STOP class for templating")

parser.add_argument("--extended_stopwords", action="store_true", help="Use an extended stopwords list")
parser.add_argument("--no_stopwords", action="store_true", help="Don't use any stopwords")

parser.add_argument("--single_vocab", action="store_true", help="Use one vocab and ignore tags")
parser.add_argument("--resample_cluster", action="store_true", help="Generate the full sample size from each cluster, even if the cluster is smaller")
parser.add_argument("--uniform_sampling", action="store_true", help="Sample from the vocab unformly rather than weighted by occurrence")

parser.add_argument("--real_exemplars", action="store_true", help="Use exemplars from the dataset if possible")

parser.add_argument(
 "--sample_size", type=int, metavar="N", default=10, help="Number of samples per cluster"
)

parser.add_argument(
 "--rate", type=float, metavar="RATE", default=0.5, help="Template noising rate"
)

parser.add_argument(
 "--template_dropout", type=float, metavar="TEMPL_DROP", default=0.0, help="Prob of using an arbitrarily different template from the same cluster"
)

parser.add_argument(
 "--seed", type=int, metavar="SEED", default=123, help="Random seed"
)

args = parser.parse_args()

DEBUG = args.debug


np.random.seed(args.seed)

if args.pos_templates:
    tagger = SequenceTagger.load('pos')
elif args.constit_templates:

    from allennlp.predictors.predictor import Predictor
    import allennlp_models.structured_prediction
    from allennlp.models.archival import load_archive
    predictor = Predictor.from_archive(
        load_archive("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz",
        cuda_device=torch.cuda.current_device()),
    )

else:
    tagger = SequenceTagger.load('chunk')

splits = ['train','dev','test']

SAMPLES_PER_CLUSTER = args.sample_size

KEEP_FRACTION = 1 - args.rate

rate_str = int(100 * args.rate)

if args.dataset == 'wa-triples':
    dataset = 'wikianswers-triples'
else:
    dataset = args.dataset

modifiers = "__".join(dataset.split('/'))
cache_key = 'cache'

if args.debug:
    modifiers += '-debug'

if args.pos_templates:
    modifiers += '-pos'
elif args.constit_templates:
    modifiers += '-constit'
else:
    modifiers += '-chunk'

if args.use_stop_class:
    modifiers += '-stopclass'
if args.no_stopwords:
    modifiers += '-nostop'
if args.extended_stopwords:
    modifiers += '-extendstop'
if args.single_vocab:
    modifiers += '-combinedvocab'

cache_key += modifiers

if args.real_exemplars:
    modifiers += '-realexemplars'
if args.resample_cluster:
    modifiers += '-resample'
if args.uniform_sampling:
    modifiers += '-uniform'
if args.seed != 123:
    seed = args.seed
    modifiers += f'-seed{seed}'

if args.template_dropout > 0:
    dropstr = str(int(args.template_dropout * 100))
    # print(dropstr)
    modifiers += f"-drop{dropstr}"

if not args.use_diff_templ_for_sem:
    modifiers += '-unforced'


name_slug = f"{modifiers}-N{SAMPLES_PER_CLUSTER}-R{rate_str}"

os.makedirs(os.path.join(args.data_dir, f'{name_slug}/'), exist_ok=True)


stopwords = []
tags_to_preserve = []
if not args.no_stopwords:
    stopwords += ['who','what','when','where', 'why','how', 'many', 'which']
if args.extended_stopwords:
        tags_to_preserve += ['.','WP','IN','$','``',"''",'DT','PRP','SYM',':']
        stopwords += ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
# tags_to_preserve = ['.','WP','IN','$','``',"''",'DT','PRP','SYM',':']


def get_chunk_parse(sent, use_stop_class=False):
    parse = [tok.labels[0].value for tok in sent.tokens]
    toks = [tok.text for tok in sent.tokens]
    

    parse_filtered = []
    toks_filtered = []

    curr_tag = None
    span_text = []
    for i in range(len(parse)):
        tok = toks[i]
    
        if tok.lower() not in stopwords:
            tag_parts = parse[i].split('-')

            if args.pos_templates:
                tag_parts = [''] + tag_parts

            if len(tag_parts) > 1:
                this_tag = tag_parts[1]
                if this_tag == curr_tag:
                    span_text.append(tok)
                elif curr_tag is not None:
                    parse_filtered.append(curr_tag)
                    toks_filtered.append(" ".join(span_text))
                    curr_tag = this_tag
                    span_text = [tok]
                else:
                    curr_tag = this_tag
                    span_text = [tok]
                    
            else:
                if len(span_text) > 0:
                    parse_filtered.append(curr_tag)
                    toks_filtered.append(" ".join(span_text))
                curr_tag = None
                span_text = []
                parse_filtered.append(curr_tag)
                toks_filtered.append(tok)
            
        else:
            if len(span_text) > 0:
                parse_filtered.append(curr_tag)
                toks_filtered.append(" ".join(span_text))
            curr_tag = None
            parse_filtered.append('STOP' if use_stop_class else None)
            toks_filtered.append(tok)
            span_text = []

    
    if len(span_text) > 0:
        parse_filtered.append(curr_tag)
        toks_filtered.append(" ".join(span_text))


    parse_filtered = [tag if tag is not None else toks_filtered[i] for i, tag in enumerate(parse_filtered)]

    return parse_filtered, toks_filtered

NODES_TO_PRESERVE = ['NP', 'PP']

def tree_to_chunks(node):
    if node['nodeType'] in NODES_TO_PRESERVE:
        return [node['word']], [node['nodeType']]
    if 'children' in node:
        sub_tags = []
        sub_words = []
        for child in node['children']:
            sub_parse = tree_to_chunks(child)
            sub_words.extend(sub_parse[0])
            sub_tags.extend(sub_parse[1])

        for i in range(len(sub_words)):
            if sub_words[i] in stopwords:
                sub_tags[i] = sub_words[i]

        return sub_words, sub_tags
    return [node['word']], [node['nodeType']]


for split in splits:

    # Check cache
    # args.use_stop_class
    # args.single_vocab
    # args.pos_templates
    # args.no_stopwords
    # args.extended_stopwords
    if args.dataset == 'wa-triples':
        with jsonlines.open(os.path.join(args.data_dir, f'wikianswers-pp/{split}.jsonl')) as f:
            all_clusters = [x for x in f]
    else:
        with jsonlines.open(os.path.join(args.data_dir, f'{args.dataset}/{split}.jsonl')) as f:
            all_clusters = [x for x in f]

    samples = []
    rebuild_cache = True

    cache_file = os.path.join(args.data_dir, f"{dataset}-cache/{cache_key}_{split}.json")
    if os.path.exists(cache_file):
        print("Loading from cache")
        with open(cache_file) as f:
            cache = json.load(f)
        vocab_by_pos = cache["vocab_by_pos"]
        parses = cache['parses']
        tokenised = cache['tokenised']
        if "qs_by_templ" in cache:
            rebuild_cache = False
            qs_by_templ = cache['qs_by_templ']
        else:
            print('Cache is missing qs_by_templ, rebuilding')
        
    if rebuild_cache:
        print("Cache file missing, building")
        parses = []
        tokenised = []
        vocab_by_pos = defaultdict(Counter)
        qs_by_templ = defaultdict(list)
        

        for cix,cluster in enumerate(tqdm(all_clusters)):
            cluster_parses = []
            cluster_tokenised = []
            for q in (cluster['qs'] if 'qs' in cluster else cluster['paraphrases']):
                if args.constit_templates:
                    res = predictor.predict_batch_json(
                    inputs=[{'sentence': q}]
                    )
                    

                    toks, parse = tree_to_chunks(res[0]['hierplane_tree']['root'])
                else:
                    sent = Sentence(q)
                    tagger.predict(sent)

                    parse, toks = get_chunk_parse(sent, use_stop_class=args.use_stop_class)

                # print([tok.labels[0].value for tok in sent.tokens])

                # print(parse)
                # print(toks)
                # exit()
                
                for ix in range(len(parse)):
                    vocab_key = 'KEY' if args.single_vocab else parse[ix]
                    vocab_by_pos[vocab_key][toks[ix]] += 1
                cluster_parses.append(" ".join(parse))
                cluster_tokenised.append(toks)
                qs_by_templ[" ".join(parse)].append(q)
            parses.append(cluster_parses)
            tokenised.append(cluster_tokenised)
            if cix > 1000 and DEBUG:
                break
        
        vocab_by_pos = {tag: [(w,count) for w,count in vocab.items() if count > 1] for tag,vocab in vocab_by_pos.items()}
        
        vocab_by_pos = {tag: sorted(vocab, reverse=True, key=lambda x: x[1])[:5000] for tag,vocab in vocab_by_pos.items()}
        vocab_by_pos_size = {tag: sum([x[1] for x in vocab]) for tag,vocab in vocab_by_pos.items()}
        vocab_by_pos = {tag: [(x[0],x[1]/vocab_by_pos_size[tag]) for x in vocab] for tag,vocab in vocab_by_pos.items()}

        os.makedirs(os.path.join(args.data_dir, f"{dataset}-cache/"), exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump({
                "vocab_by_pos": vocab_by_pos,
                "parses": parses,
                "tokenised": tokenised,
                "qs_by_templ": qs_by_templ
            }, f)
    
    
    
    max_cluster_len = max([len(c['qs'] if 'qs' in c else c['paraphrases']) for c in all_clusters])
    max_q_len = max([len(toks) for c in tokenised for toks in c])
    max_vocab_size = max([len(voc) for voc in vocab_by_pos.values()])

    noising_randoms = np.random.rand(len(all_clusters), SAMPLES_PER_CLUSTER, max_q_len)
    replace_randoms = np.random.rand(len(all_clusters), SAMPLES_PER_CLUSTER, max_q_len)
    dropout_randoms = np.random.rand(len(all_clusters), SAMPLES_PER_CLUSTER)
    # replace_randoms = np.random.randint(0, max_vocab_size, size=(len(all_clusters), SAMPLES_PER_CLUSTER, max_q_len))
    sample_randoms = np.random.randint(0, max_cluster_len, size=(len(all_clusters), SAMPLES_PER_CLUSTER, 4))

    # num_samples = SAMPLES_PER_CLUSTER if split == 'train' else 1

    for cix,row in enumerate(tqdm(all_clusters)):
        cluster = row['qs'] if 'qs' in row else row['paraphrases']

        sample_size = SAMPLES_PER_CLUSTER if args.resample_cluster else min(SAMPLES_PER_CLUSTER, len(cluster)-1)
        for i in range(sample_size):
#             tgt_ix, sem_ix = np.random.choice(len(cluster), replace=False, size=2)
            tgt_ix, sem_ix, parse_ix, exemplar_ix = sample_randoms[cix][i]
            tgt_ix = tgt_ix % len(cluster)
            tgt_txt = cluster[tgt_ix]

            if args.template_dropout > 0 and dropout_randoms[cix, i] < args.template_dropout:
                tgt_parse = parses[cix][parse_ix % len(cluster)]
            else:
                tgt_parse = parses[cix][tgt_ix]

            sem_options = [cluster[i] for i in range(len(cluster)) if parses[cix][tgt_ix] != parses[cix][i]]
            if len(sem_options) > 0: # and args.use_diff_templ_for_sem
                sem_ix = sem_ix % len(sem_options)
                sem_text = sem_options[sem_ix]
            else:
                sem_ix = sem_ix % len(cluster)
                sem_text = cluster[sem_ix]

            toks = tokenised[cix][tgt_ix]

            exemplar_options = []
            if args.real_exemplars and tgt_parse in qs_by_templ:
                exemplar_options = deepcopy(qs_by_templ[tgt_parse])
                # remove any exemplar from this cluster
                for q in cluster:
                    if q in exemplar_options:
                        exemplar_options.remove(q)
            if len(exemplar_options) > 0:
                syn_text = exemplar_options[exemplar_ix % len(exemplar_options)]
            else:
                # Build an exemplar by noising
                syn_text = []
                
                j=0
                for tok, tag in zip(toks, tgt_parse.split(' ')):
                    if tag not in tags_to_preserve and tok.lower() not in stopwords and noising_randoms[cix,i,j] > KEEP_FRACTION:
    #                     replacement = np.random.choice(list(vocab_by_pos[tag]))
                        
                        vocab_key = 'KEY' if args.single_vocab else tag

                        if len(vocab_by_pos[vocab_key]) > 0:
                            options, probs = zip(*vocab_by_pos[vocab_key])
                            # repl_ix = replace_randoms[cix,i,j] % len(options)
                            cum_prob = 0
                            for k in range(len(probs)):
                                cum_prob += 1/len(probs) if args.uniform_sampling else probs[k]
                                if cum_prob > replace_randoms[cix,i,j]:
                                    repl_ix = k
                                    break
                            syn_text.append(options[repl_ix])
                        else:
                            syn_text.append(tok)
                    else:
                        syn_text.append(tok)
                    j += 1
                syn_text = " ".join(syn_text)


            samples.append({
                'tgt': tgt_txt,
                'sem_input': sem_text,
                'syn_input': syn_text
            })
#             print(samples[-1])
#             break
#         break
        if cix > 100 and DEBUG:
            # print(vocab_by_pos)
            # print(samples[:10])
            break
#     break
    
    with jsonlines.open(f'{args.data_dir}/{name_slug}/{split}.jsonl','w') as f:
        f.write_all(samples)