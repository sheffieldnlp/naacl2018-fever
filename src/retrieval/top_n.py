import math,sys

from .retrieval_method import RetrievalMethod
from drqa import retriever
from drqascripts.retriever.build_tfidf_lines import OnlineTfidfDocRanker

from common.util.log_helper import LogHelper

class TopNDocsTopNSents(RetrievalMethod):


    class RankArgs:
        def __init__(self):
            self.ngram = 2
            self.hash_size = int(math.pow(2,24))
            self.tokenizer = "simple"
            self.num_workers = None

    def __init__(self,db,n_docs,n_sents,model):
        super().__init__(db)
        self.n_docs = n_docs
        self.n_sents = n_sents
        self.ranker = retriever.get_class('tfidf')(tfidf_path=model)
        self.onlineranker_args = self.RankArgs()

    def get_docs_for_claim(self, claim_text):
        doc_names, doc_scores = self.ranker.closest_docs(claim_text, self.n_docs)
        return zip(doc_names, doc_scores)


    def tf_idf_sim(self, claim, lines, freqs=None):
        tfidf = OnlineTfidfDocRanker(self.onlineranker_args, [line["sentence"] for line in lines], freqs)
        line_ids, scores = tfidf.closest_docs(claim,self.n_sents)
        ret_lines = []
        for idx, line in enumerate(line_ids):
            ret_lines.append(lines[line])
            ret_lines[-1]["score"] = scores[idx]
        return ret_lines

    LogHelper.setup()
    logger = LogHelper.get_logger(__name__)


    def get_sentences_given_claim(self,page,logger,line_no):
        lines = self.db.get_doc_lines(page)
        lines = [line.split("\t")[1] if len(line.split("\t")[1]) > 1 else "" for line in
                 lines.split("\n")]
        sent=lines[line_no]
        return sent


    def get_sentences_for_claim(self,claim_text,include_text=False):
        #given a claim get a bunch of documents that might be relevant for it
        pages = self.get_docs_for_claim(claim_text)
        sorted_p = list(sorted(pages, reverse=True, key=lambda elem: elem[1]))
        pages = [p[0] for p in sorted_p[:self.n_docs]]
        p_lines = []
        for page in pages:
            logger.info("page:"+page)
            #query the db and get the list of sentences in a given wikipedia page
            lines = self.db.get_doc_lines(page)
            logger.info(lines)
            sys.exit(1)
            lines = [line.split("\t")[1] if len(line.split("\t")[1]) > 1 else "" for line in
                     lines.split("\n")]

            p_lines.extend(zip(lines, [page] * len(lines), range(len(lines))))


        lines = []
        for p_line in p_lines:
            logger.info("value of sentence in p_line is:"+p_line[0])
            sys.exit(1)
            lines.append({
                "sentence": p_line[0],
                "page": p_line[1],
                "line_on_page": p_line[2]
            })

        scores = self.tf_idf_sim(claim_text, lines)

        if include_text:
            return scores

        return [(s["page"], s["line_on_page"]) for s in scores]

