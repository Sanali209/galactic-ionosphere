import unittest

from tqdm import tqdm

from SLM.NLPSimple.NLPPipline import NLPPipline, NLPTextReplace, NLPTextTokenize
from SLM.files_db.components.File_record_wraper import FileRecord
from SLM.groupcontext import group


class TestSLMFSDB(unittest.TestCase):


    def test_create_bag_of_words(self):
        bag_of_words_save_path = r"D:\data\bags_of_words.json"
        reords = FileRecord.find({})
        names_str = ""
        for record in tqdm(reords):
            names_str += record.name + " "

        NLP_pipline = NLPPipline()
        NLP_pipline.text = names_str
        with group():
            NLP_pipline.operations.append(NLPTextReplace('.', ' '))
            NLP_pipline.operations.append(NLPTextReplace('_', ' '))
            NLP_pipline.operations.append(NLPTextReplace('-', ' '))
            NLP_pipline.operations.append(NLPTextReplace('[', ' '))
            NLP_pipline.operations.append(NLPTextReplace(']', ' '))
            NLP_pipline.operations.append(NLPTextReplace('(', ' '))
            NLP_pipline.operations.append(NLPTextReplace(')', ' '))
            NLP_pipline.operations.append(NLPTextReplace('+', ' '))
            NLP_pipline.operations.append(NLPTextReplace('=', ' '))
            NLP_pipline.operations.append(NLPTextReplace('\\', ' '))

            NLP_pipline.operations.append(NLPTextTokenize())
            NLP_pipline.operations.append(NLPTokensToLoverCase())
            NLP_pipline.operations.append(NLPTokensStripSpacesOperation())
            NLP_pipline.operations.append(NLPTokensDeleteStopWords())
            NLP_pipline.operations.append(NLPTokensDeleteIntegers())
            NLP_pipline.operations.append(NLPTokensDeleteHexadecimal())
            NLP_pipline.operations.append(NLPTokensDeleteShort(3))
            NLP_pipline.operations.append(NLPTokensDeleteRandString())
            NLP_pipline.operations.append(NLPTokensBagOfWords())
            NLP_pipline.operations.append(NLPTokensBagOfWordsDeleteWithFrequency(1))
            NLP_pipline.operations.append(NLPTokensSaveBagOfWords(bag_of_words_save_path))

        NLP_pipline.run()

    def test_create_bag_of_words_fromDescription(self):
        bag_of_words_save_path = r"D:\data\bags_of_words_description.json"
        reords = FileRecord.find({})
        names_str = ""
        for record in reords:
            names_str += record.description + " "

        NLP_pipline = NLPPipline()
        NLP_pipline.text = names_str
        with group():
            NLP_pipline.operations.append(NLPTextReplace('.', ' '))
            NLP_pipline.operations.append(NLPTextReplace('_', ' '))
            NLP_pipline.operations.append(NLPTextReplace('-', ' '))
            NLP_pipline.operations.append(NLPTextReplace('[', ' '))
            NLP_pipline.operations.append(NLPTextReplace(']', ' '))
            NLP_pipline.operations.append(NLPTextReplace('(', ' '))
            NLP_pipline.operations.append(NLPTextReplace(')', ' '))
            NLP_pipline.operations.append(NLPTextReplace('+', ' '))
            NLP_pipline.operations.append(NLPTextReplace('=', ' '))
            NLP_pipline.operations.append(NLPTextReplace('\\', ' '))

            NLP_pipline.operations.append(NLPTextTokenize())
            NLP_pipline.operations.append(NLPTokensToLoverCase())
            NLP_pipline.operations.append(NLPTokensStripSpacesOperation())
            NLP_pipline.operations.append(NLPTokensDeleteStopWords())
            NLP_pipline.operations.append(NLPTokensDeleteIntegers())
            NLP_pipline.operations.append(NLPTokensDeleteHexadecimal())
            NLP_pipline.operations.append(NLPTokensDeleteShort(3))
            NLP_pipline.operations.append(NLPTokensDeleteRandString())
            NLP_pipline.operations.append(NLPTokensBagOfWords())
            NLP_pipline.operations.append(NLPTokensBagOfWordsDeleteWithFrequency(1))
            NLP_pipline.operations.append(NLPTokensSaveBagOfWords(bag_of_words_save_path))

        NLP_pipline.run()

    def test_get_ai_description(self):
        path = r"E:\rawimagedb\repository\nsfv repo\drawn"
        query = {'local_path': {"$regex": '^' + re.escape(path)}}
        records = FileRecord.find(query)
        for record in tqdm(records):
            record: FileRecord
            ext = os.path.splitext(record.name)[1]
            if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue
            if record.description is not None and record.description != "" and not isinstance(record.description, dict):
                continue
            try:
                description = record.get_ai_expertise("image-text", "mc_llava_13b_4b")
                record.description = description['data']
            except Exception as e:
                continue
            print(record.description)