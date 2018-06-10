import functools
import os
import tempfile

from six import string_types

from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import XMLCorpusReader, XMLCorpusView
import re
import numpy as np
import pickle

def _parse_args(fun):
    """
    Wraps function arguments:
    if fileids not specified then function set NKJPCorpusReader paths.
    """
    @functools.wraps(fun)
    def decorator(self, fileids=None, **kwargs):
        if not fileids:
            fileids = self._paths
        return fun(self, fileids, **kwargs)

    return decorator


class NKJPCorpusReader(XMLCorpusReader):
    WORDS_MODE = 0
    NE_MODE = 1

    def __init__(self, root, fileids='.*'):
        """
        Corpus reader designed to work with National Corpus of Polish.
        See http://nkjp.pl/ for more details about NKJP.
        use example:
        import nltk
        import nkjp
        from nkjp import NKJPCorpusReader
        x = NKJPCorpusReader(root='/home/USER/nltk_data/corpora/nkjp/', fileids='') # obtain the whole corpus
        x.header()
        x.raw()
        x.words()
        x.tagged_words(tags=['subst', 'comp'])  #Link to find more tags: nkjp.pl/poliqarp/help/ense2.html
        x.sents()
        x = NKJPCorpusReader(root='/home/USER/nltk_data/corpora/nkjp/', fileids='Wilk*') # obtain particular file(s)
        x.header(fileids=['WilkDom', '/home/USER/nltk_data/corpora/nkjp/WilkWilczy'])
        x.tagged_words(fileids=['WilkDom', '/home/USER/nltk_data/corpora/nkjp/WilkWilczy'], tags=['subst', 'comp'])
        """
        if isinstance(fileids, string_types):
            XMLCorpusReader.__init__(self, root, fileids + '.*/header.xml')
        else:
            XMLCorpusReader.__init__(self, root, [fileid + '/header.xml' for fileid in fileids])
        self._paths = self.get_paths()

    def get_paths(self):
        return [os.path.join(str(self._root), f.split("header.xml")[0]) for f in self._fileids]


    def fileids(self):
        """
        Returns a list of file identifiers for the fileids that make up
        this corpus.
        """
        return [f.split("header.xml")[0] for f in self._fileids]


    def _view(self, filename, **kwargs):
        """
        Returns a view specialised for use with particular corpus file.
        """
        mode = kwargs.pop('mode', NKJPCorpusReader.WORDS_MODE)
        if mode is NKJPCorpusReader.WORDS_MODE:
            return NKJPCorpus_Words_View(filename)
        elif mode is NKJPCorpusReader.NE_MODE:
            return NKJPCorpus_Named_View(filename)

        else:
            raise NameError('No such mode!')

    def add_root(self, fileid):
        """
        Add root if necessary to specified fileid.
        """
        if self.root in fileid:
            return fileid
        return self.root + fileid


    @_parse_args
    def words(self, fileids=None, **kwargs):
        """
        Returns words in specified fileids.
        """
        for fileid in fileids:
            if not os.path.exists(os.path.join(self.add_root(fileid), 'ann_words.xml')):
                return []
        return concat([self._view(self.add_root(fileid),
                                  mode=NKJPCorpusReader.WORDS_MODE, **kwargs).handle_query()
                       for fileid in fileids])


    @_parse_args
    def named_entities(self, fileids=None, **kwargs):
        """
        Call with specified tags as a list, e.g. tags=['subst', 'comp'].
        Returns tagged words in specified fileids.
        """
        for fileid in fileids:
            if not os.path.exists(os.path.join(self.add_root(fileid), 'ann_named.xml')):
                return []
        return concat([self._view(self.add_root(fileid),
                                  mode=NKJPCorpusReader.NE_MODE, **kwargs).handle_query()
                       for fileid in fileids])


class XML_Tool():
    """
    Helper class creating xml file to one without references to nkjp: namespace.
    That's needed because the XMLCorpusView assumes that one can find short substrings
    of XML that are valid XML, which is not true if a namespace is declared at top level
    """
    def __init__(self, root, filename):
        self.read_file = os.path.join(root, filename)
        self.write_file = tempfile.NamedTemporaryFile(delete=False, mode='w+',
                                                      encoding = 'utf-8')

    def build_preprocessed_file(self):
        try:
            fr = open(self.read_file, 'r', encoding='utf-8')
            fw = self.write_file
            line = ' '
            while len(line):
                line = fr.readline()
                x = re.split(r'nkjp:[^ ]* ', line)  #in all files
                ret = ' '.join(x)
                x = re.split('<nkjp:paren>', ret)   #in ann_segmentation.xml
                ret = ' '.join(x)
                x = re.split('</nkjp:paren>', ret)  #in ann_segmentation.xml
                ret = ' '.join(x)
                x = re.split('<choice>', ret)   #in ann_segmentation.xml
                ret = ' '.join(x)
                x = re.split('</choice>', ret)  #in ann_segmentation.xml
                ret = ' '.join(x)
                fw.write(ret)
            fr.close()
            fw.close()
            return self.write_file.name
        except Exception:
            self.remove_preprocessed_file()
            raise Exception


    def remove_preprocessed_file(self):
        os.remove(self.write_file.name)
        pass

class NKJPCorpus_Words_View(XMLCorpusView):
    """
    A stream backed corpus view specialized for use with
    ann_morphosyntax.xml files in NKJP corpus.
    """

    def __init__(self, filename, **kwargs):
        self.tagspec = '.*/seg/fs'
        self.xml_tool = XML_Tool(filename, 'ann_words.xml')
        XMLCorpusView.__init__(self, self.xml_tool.build_preprocessed_file(), self.tagspec)

    def handle_query(self):
        try:
            self._open()
            words = []
            while True:
                segm = XMLCorpusView.read_block(self, self._stream)
                if len(segm) == 0:
                    break
                for part in segm:
                    if part is not None:
                        words.append(part)
            self.close()
            self.xml_tool.remove_preprocessed_file()
            return words
        except Exception:
            self.xml_tool.remove_preprocessed_file()
            raise Exception


    def handle_elt(self, elt, context):
        word = ''
        tag = ''
        flag = False
        is_not_interp = True

        for child in elt:

            #get word
            if 'name' in child.keys() and child.attrib['name'] == 'orth':
                for symbol in child:
                    if symbol.tag == 'string':
                        word = symbol.text
            elif 'name' in child.keys() and child.attrib['name'] == 'ctag':
                for symbol_tag in child:
                    if 'value' in symbol_tag.keys() and symbol_tag.attrib['value'] != 'Interp':
                        tag = symbol_tag.attrib['value']
                    elif 'value' in symbol_tag.keys() and symbol_tag.attrib['value'] == 'Interp':
                        is_not_interp = False
        if is_not_interp:
            return (word, tag)
        
class NKJPCorpus_Named_View(XMLCorpusView):
    """
    A stream backed corpus view specialized for use with
    ann_morphosyntax.xml files in NKJP corpus.
    """

    def __init__(self, filename, **kwargs):
        self.tagspec = '.*/seg/fs'
        self.xml_tool = XML_Tool(filename, 'ann_named.xml')
        XMLCorpusView.__init__(self, self.xml_tool.build_preprocessed_file(), self.tagspec)

    def handle_query(self):
        try:
            self._open()
            words = []
            while True:
                segm = XMLCorpusView.read_block(self, self._stream)
                if len(segm) == 0:
                    break
                for part in segm:
                    if part is not None:
                        words.append(part)
            self.close()
            self.xml_tool.remove_preprocessed_file()
            return words
        except Exception:
            self.xml_tool.remove_preprocessed_file()
            raise Exception


    def handle_elt(self, elt, context):
        word = ''
        tag = ''
        flag = False
        is_not_interp = True

        for child in elt:

            #get word
            if 'name' in child.keys() and child.attrib['name'] == 'orth':
                for symbol in child:
                    if symbol.tag == 'string':
                        word = symbol.text
            elif 'name' in child.keys() and child.attrib['name'] == 'type':
                for symbol_tag in child:
                    if 'value' in symbol_tag.keys():
                        tag = symbol_tag.attrib['value']
        if is_not_interp:
            return (word, tag)
    
if __name__ == "__main__":
    x = NKJPCorpusReader(root='NKJP-PodkorpusMilionowy-1.2') # obtain the whole corpus
    fileids = x.fileids()
    all_word_data = []
    for fileid in fileids:
        x = NKJPCorpusReader(root='NKJP-PodkorpusMilionowy-1.2', fileids=fileid.split("/")[0])
        names = x.named_entities()
        words = x.words()
        word_data = []
        named_data = []
        for named, name in names:
            for val in re.split('\s+', named):
                named_data.append((val, name))
        for word, tag in words:
            label = next((name for ind, (named, name) in enumerate(named_data) if named == word), 'I')
            word_data.append((word, tag, label))
        all_word_data.append(word_data)
        print("Words in " + fileid + " " + str(len(word_data)))
    print(str(len(all_word_data)))
    with open('word_data_file.obj', 'wb') as words_object:
        pickle.dump(all_word_data, words_object)
            
