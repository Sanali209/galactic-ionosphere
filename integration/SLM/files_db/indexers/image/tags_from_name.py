from SLM.NLPSimple.NLPPipline import NLPPipline, NLPTextReplace, NLPTextTokenize, NLPTokensToLoverCase, \
    NLPTokensDeleteStopWords, NLPTokensDeleteShort, NLPTokensDeleteDuplicates, NLPTokensStripSpacesOperation, \
    NLPTokensDeleteIntegers, NLPTokensDeleteHexadecimal, NLPTokensDeleteRandString, load_bag_of_words, \
    NLPTokensSetBagOfWords, NLPTokensDeleteNotInBagOfWords
from SLM.files_db.indexers.image.content_md5 import files_db_indexer
from SLM.files_db.components.fs_tag import TagRecord

from SLM.chains.chains_main import DictFieldMergeChainFunction, DictFormatterChainFunction
from SLM.groupcontext import group
from SLM.indexerpyiplain.idexpyiplain import ItemIndexer
from SLM.metadata.MDManager.mdmanager import MDManager


# todo :tags black list
# todo: tags sinonims


class ImageTagsFromName(files_db_indexer):
    bag_of_words_load_path = r"D:\data\bags_of_words.json"
    bag_of_words_data = load_bag_of_words(bag_of_words_load_path)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fieldName = "Tags_from_name"

    def index(self, parent_indexer: ItemIndexer, item, need_index):
        from SLM.files_db.components.File_record_wraper import FileRecord
        file_item = FileRecord(item["_id"])

        NLP_pipline = NLPPipline()
        NLP_pipline.text = file_item.full_path
        NLP_pipline.bagOfWords = ImageTagsFromName.bag_of_words_data[0]
        NLP_pipline.countedBagOfWords= ImageTagsFromName.bag_of_words_data[1]
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
            NLP_pipline.operations.append(NLPTokensDeleteDuplicates())
            NLP_pipline.operations.append(NLPTokensDeleteRandString())
            NLP_pipline.operations.append(NLPTokensDeleteNotInBagOfWords())

        NLP_pipline.run()
        taglist = item.get("tags", [])

        for tag in NLP_pipline.tokens:
            tag = "from_name/" + tag
            tag_record = TagRecord.get_or_create(tag)
            taglist.append(tag_record.fullName)
        taglist = list(set(taglist))
        item["tags"] = taglist

        parent_indexer.shared_data["item_indexed"] = True
        self.mark_as_indexed(item, parent_indexer)



