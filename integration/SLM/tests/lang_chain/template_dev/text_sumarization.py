import os
import unittest

from g4f import Provider

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import WebBaseLoader

os.environ['DATA_CACHE_MANAGER_PATH'] = r'D:\data\ImageDataManager'
text = """Описание проекта
html2image logo
PyPI PyPI PyPI GitHub GitHub

PyPI Package	GitHub Repository
A lightweight Python package acting as wrapper around the headless mode of existing web browsers, allowing image generation from HTML/CSS strings, files and URLs.

 
This package has been tested on Windows, Ubuntu (desktop and server) and MacOS. If you encounter any problems or difficulties while using it, feel free to open an issue on the GitHub page of this project. Feedback is also welcome!

Principle
Most web browsers have a Headless Mode, which is a way to run them without displaying any graphical interface. Headless mode is mainly used for automated testing but also comes in handy if you want to take screenshots of web pages that are exact replicas of what you would see on your screen if you were using the browser yourself.

However, for the sake of taking screenshots, headless mode is not very convenient to use. HTML2Image aims to hide the inconveniences of the browsers' headless modes while adding useful features, such as allowing the creation of images from simple strings.

For more information about headless modes :

(Chrome) https://developers.google.com/web/updates/2017/04/headless-chrome
(Firefox) https://developer.mozilla.org/en-US/docs/Mozilla/Firefox/Headless_mode
Installation
HTML2Image is published on PyPI and can be installed through pip:

pip install --upgrade html2image
In addition to this package, at least one of the following browsers must be installed on your machine :

Google Chrome (Windows, MacOS)
Chromium Browser (Linux)
Microsoft Edge """

from typing import Optional, List, Mapping, Any
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import g4f


class EducationalLLM(LLM):

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        out = g4f.ChatCompletion.create(
            model=Provider.Bing.get_models()[3],provider=Provider.Bing,
            messages=[{"role": "user", "content": prompt}],
        )  #
        if stop:
            stop_indexes = (out.find(s) for s in stop if s in out)
            min_stop = min(stop_indexes, default=-1)
            if min_stop > -1:
                out = out[:min_stop]
        return out


class TestSLMFSDB(unittest.TestCase):
    def test_gui(self):
        from g4f.gui import run_gui
        run_gui()

    def test_summarization_2(self):

        # Define prompt
        prompt_template = """Write a concise summary of the following:
        "{text}"
        CONCISE SUMMARY:"""
        prompt = PromptTemplate.from_template(prompt_template)
        llm = EducationalLLM()
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        # Create an instance of LLMChain
        summarization_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

        loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
        docs = loader.load()

        summary = summarization_chain.invoke(docs)
        print(summary)


    def test_youtube_summarization(self):
        from langchain_community.document_loaders import YoutubeLoader
        from langchain.chains.summarize import load_summarize_chain

        # Define prompt
        prompt_template = """Write text article from given text :
                ARTICLE WRITING TEMPLATE:
                # article header
                
                ## shot intro
                describe about what be text
                
                ## table of content
                write table of content of important parts
                
                ## main text
                rewrite the text in your own words with more short and clear keep important parts
                
                ## conclusion
                write your conclusion
                
                ARTICLE WRITING RULES:
                - use .md files syntax for formating text
                INPUT TEXT:
                "{text}"
                ARTICLE TEXT:"""

        prompt = PromptTemplate.from_template(prompt_template)

        loader = YoutubeLoader.from_youtube_url(
            "https://www.youtube.com/watch?v=Py7fGlU7DGY", add_video_info=True,
            language=["en", "ru"],
            translation="en",
        )
        document = loader.load()
        llm = EducationalLLM()
        chain = LLMChain(prompt=prompt, llm=llm)
        summarization_chain = StuffDocumentsChain(llm_chain=chain, document_variable_name="text")
        resalt  = summarization_chain.invoke(document)
        print(resalt)





