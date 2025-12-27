import os
import uuid

from html2image import Html2Image
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

from SLM.LangChain.LangChainHelper import LLM_hugingface_model_inference
from SLM.appGlue.DesignPaterns import allocator
from SLM.files_db.components.collectionItem import CollectionRecord
from SLM.mongoext.MongoClientEXT_f import MongoClientExt

from SLM.mongoext.wraper import MongoRecordWrapper, FieldPropInfo

import requests
from bs4 import BeautifulSoup


class uriDecoderManager:
    decoders = {}

    @staticmethod
    def get_scrinshoth_path():
        # todo save to glbal data cache
        return os.path.join(os.path.dirname(__file__), 'screen_shot')

    @staticmethod
    def get_scrinshoth_size():
        return 512, 512

    @classmethod
    def register_decoder(cls, decoder):
        cls.decoders[decoder.__class__.__name__] = decoder


class uriDecoder:

    def create_thumbnail(self, url):
        # todo fix this strong dependency for edge browser and html2image
        hti = Html2Image(output_path=uriDecoderManager.get_scrinshoth_path(), browser='edge')
        str_uuid = str(uuid.uuid4())
        hti.screenshot(url=url, save_as=f'{str_uuid}.jpg',
                       size=uriDecoderManager.get_scrinshoth_size())
        return None

    def get_page_text(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        return text

    def get_summary(self, page_text):
        template_string = (
            """
            INSTRUCTIONS:
            you are given a text of web page. You need to write a short summary of the text.
            end the answer with 
            ---------------------------------------
            ANSWER FORMAT:
            text surrounded by tags <summary> and </summary>
            ------------------------------------------------------------
            Question:
            page_text:<page_text> {file_name} </page_text>
            answer:<summary>on this page"""
        )
        # todo implement summarization by lang_chain
        text = page_text
        if len(text) > 32000:
            text_shorten = text[:32000]
        else:
            text_shorten = text
        prompt = PromptTemplate.from_template(template_string)

        model = LLM_hugingface_model_inference(model_name='mistralai/Mistral-7B-Instruct-v0.2')

        llm_chain = LLMChain(prompt=prompt, llm=model)

        answer = llm_chain.run(text_shorten)

        extracted_summary = answer.split('<summary>')[2]
        extracted_summary = extracted_summary.split('</summary>')[0]
        return extracted_summary


class WebLinkRecord(CollectionRecord):
    itemType: str = FieldPropInfo('item_type', str, 'WebLinkRecord')
    uri: str = FieldPropInfo('name', str, None)

    @classmethod
    def new_record(cls, uri, **kwargs):
        return super(WebLinkRecord, cls).new_record(**kwargs)


def init(config):
    CollectionRecord.itemTypeMap['WebLinkRecord'] = WebLinkRecord
    service = allocator.Allocator.get_instance(MongoClientExt)
    service.register_collection("WebLinkRecord", WebLinkRecord)


allocator.Allocator.add_initializer(init)
