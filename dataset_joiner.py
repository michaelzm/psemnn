#all work to prepare the dataset is done inside DatasetWorker
import numpy as np
from tqdm import tqdm 
import os
import zipfile
import pickle

np.random.seed(1234567890)

class DatasetWorker(object):
    #init the worker with the dataset (in our case its the example_data)
    def __init__(self, _dataset):
        self.dataset = _dataset
        #default train test split ratio is 0.8
        self.train_test_split = 0.8
        self.split_size = 0
        
        self.extraction_of = ""
        
        self.train_tokens = []
        self.test_tokens = []
        
        self.train_labels = list()
        self.test_labels = list()
        self.train_labels_uncertainty = list()
    #define here extraction of sentiment, aspect or modifier
    def setExtractionOf(self,to_extract):
        self.extraction_of = to_extract
        
    def applyPreprocessing(self):
        preprocessor = DatasetPreprocessor()
        self.dataset = preprocessor.tokenizeDataset(self.dataset)
        
    #configure train test split ratio, default is 0.8
    def setTrainTestSplitRatio(self, _t_t_param):
        self.train_test_split = _t_t_param
        print("set train test split ratio to "+str(self_train_test_split))
    
    #split the tokens (words) into litst of train and test tokens
        
    def splitDatasetTokens(self):
        len_dataset = len(self.dataset) - 1
        self.split_size = round(len_dataset * self.train_test_split)
        for i, (k,v) in tqdm(enumerate(self.dataset.items()), desc="split dataset tokens"):
            if i < self.split_size:
                self.train_tokens.append(v["tokens"])
            else:
                self.test_tokens.append(v["tokens"])
    #param split_by
    # union -> always use the sentiment labeling of every user, even if only 1 out of n users
    # labeled the token as sentiment
    def splitDatasetLabels(self, split_by):
        for d_idx, (k,v) in tqdm(enumerate(self.dataset.items()), desc="split dataset labels"):
            curr_users = [s for s in v.keys() if s != "tokens"]
            
            if split_by == "union":
                merged_label = []
                for i in range(len(v["tokens"])):
                    merged_label.insert(i, "O")
                
                for usr in curr_users:
                    for i, e in enumerate(v[usr][self.extraction_of]):
                        if e != "O":
                            del merged_label[i]
                            merged_label.insert(i, e)
                if d_idx < self.split_size:
                    self.train_labels.append(merged_label)
                else:
                    self.test_labels.append(merged_label)
    def buildDatasetSequence(self,max_seq_length):
        self.train_tokens = [t[0:max_seq_length] for t in tqdm(self.train_tokens, desc="update train tokens")]
        self.test_tokens = [t[0:max_seq_length] for t in tqdm(self.test_tokens, desc="update test tokens")]
        self.train_labels = [t[0:max_seq_length] for t in tqdm(self.train_labels, desc="update train labels")]
        self.test_labels = [t[0:max_seq_length] for t in tqdm(self.test_labels, desc="update test labels")]
    
    #describe train test dataset  - partially hardcoded
    def describe(self):
        print("train test ratio "+str(self.train_test_split))
        print("train data sentence count: "+str(len(self.train_labels)))
        print("test data sentence count: "+str(len(self.test_labels))+"\n")
        sum_train_tokens = 0
        train_0_count = 0
        train_B_S_count = 0
        train_I_S_count = 0
        train_analysis = {"0":0, "B_S": 0, "I_S": 0}

        sum_test_tokens = 0
        test_analysis = {"0":0, "B_S": 0, "I_S": 0}

        #train analysis
        for i in range(len(self.train_labels)):
            sum_train_tokens+=len(self.train_labels[i])
            for tok in range(len(self.train_labels[i])):
                token = self.train_labels[i][tok]
                if token == "O":
                    train_analysis["0"] += 1
                elif token == "B_S":
                    train_analysis["B_S"] += 1
                else:
                    train_analysis["I_S"] += 1

        #test analysis
        for i in range(len(self.test_labels)):
            sum_test_tokens+=len(self.test_labels[i])
            for tok in range(len(self.test_labels[i])):
                token = self.test_labels[i][tok]
                if token == "O":
                    test_analysis["0"] += 1
                elif token == "B_S":
                    test_analysis["B_S"] += 1
                else:
                    test_analysis["I_S"] += 1
                    
        print("train token count: "+str(sum_train_tokens))
        print("test token count: "+str(sum_test_tokens)+"\n")
        print("train details: "+str(train_analysis))
        print("test details: "+str(test_analysis))
                    
#does all the nasty stuff like tokenizing, lower casing etc.
class DatasetPreprocessor(object):
    def __init__(self):
        self.openTasks = []
    
    #we tokenize each word inside the sentences and apply lowercase lettering
    def tokenizeDataset(self, dataset_input):
        for i,(k,v) in tqdm(enumerate(dataset_input.items()), desc="tokenize dataset"):
            tokens = v.get("tokens")
            tokens = [token.lower() for token in tokens]
            dataset_input[k]["tokens"] = tokens
        return dataset_input

#does vocabulary, token ids and embedding for vocabulary
class VocabularyWorker(object):
    def __init__(self):
        self.vocabulary = None
        self.filename_embedding = None
        self.embedding= None
        self.max_seq_length= None
        self.labelclass_to_id= None
        self.n_tags= None
        self.embedding_vectors= None
        self.vocab_size = None
    
    #first step: gett all words
    def buildVocabulary(self, dataset):
        vocabulary = set(['##unknown_token##','##padding_token##'])
        for v in tqdm(dataset.values(), desc="build vocabulary"):
            vocabulary.update(set(v["tokens"]))
        vocabulary = list(vocabulary)
        vocabulary.sort()
        self.vocabulary = vocabulary
        
    #get embedding 
    def prepareEmbedding(self):
        ## glove word embeddings as input for the neural network
        ## download glove.42B.300d.txt from http://nlp.stanford.edu/data/glove.42B.300d.zip
        filename_embedding_zip = r'glove.42B.300d.zip' # folder of downloaded glove zip file
        ## specify folder where to store the glove embeddings
        filepath_embedding = filename_embedding_zip.replace('.zip','')
        ## unzip and save glove to a folder manually or with the next lines
        if not os.path.exists(filepath_embedding):
            with zipfile.ZipFile(filename_embedding_zip,"r") as zip_ref:
                zip_ref.extractall(filepath_embedding)
        os.listdir(filepath_embedding)[0]
        self.filename_embedding = filepath_embedding + '/' + os.listdir(filepath_embedding)[0]

    def buildEmbedding(self, train_labels_input):
        self.prepareEmbedding()
        filename_rel_embedding = self.filename_embedding.replace('.txt','_rel.pkl')
        if not os.path.exists(filename_rel_embedding):
            embedding = load_embedding(self.filename_embedding, self.vocabulary)
            pickle.dump(embedding,open(filename_rel_embedding,'wb'))
        else:
            embedding = pickle.load(open(filename_rel_embedding,'rb'))
       
        #make embedding globally accessable
        self.embedding = embedding
        
        #build the embedding vectors
        vocab_size = len(self.vocabulary)
        self.vocab_size = vocab_size
        embed_size = list(embedding.values())[0]["vector"].shape[0]
        embedding_vectors = np.zeros((vocab_size, embed_size))
        
        # in embedding, there are only embeddings for known words of train and test data
        for v in tqdm(embedding.values(), desc="build embedding vectors"):
            vector = v['vector']
            token_id = v['token_id']
            embedding_vectors[token_id] = vector
        
        self.embedding_vectors = embedding_vectors
        all_labelclasses = set()
        for row in tqdm(train_labels_input, desc="build labelclasses"):
            all_labelclasses.update(row)
        all_labelclasses=list(all_labelclasses)

        all_labelclasses.sort()
        self.labelclass_to_id = dict(zip(all_labelclasses,list(range(len(all_labelclasses)))))

        self.n_tags = len(list(self.labelclass_to_id.keys()))

    def convert_tokens_labels_list_to_ids_list(self, tokens_list, labels_list,max_seq_length):
        
        token_ids_list, label_ids_list = [], []
        n_tags = len(list(self.labelclass_to_id.keys()))
        for index in tqdm(range(len(tokens_list)), desc="Converting tokens & labels to ids "):
            tokens = tokens_list[index]
            labels = labels_list[index]   
            token_ids = []
            for token in tokens:
                if token not in self.embedding.keys():
                    token = '##unknown_token##'
                token_ids.append(self.embedding[token]['token_id'])
            label_ids = [self.labelclass_to_id[label] for label in labels]

            # Zero-pad up to the sequence length.
            while len(token_ids) < max_seq_length:
                token_ids.append(0)
                label_ids.append(0)
            token_ids_list.append(token_ids)
            label_ids_list.append(label_ids)
        
        return (
            np.array(token_ids_list),
            np.array(label_ids_list)
        )