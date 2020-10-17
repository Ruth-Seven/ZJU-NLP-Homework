import logging
import os.path
import sys

import re
import zhconv
import jieba
from gensim.corpora import WikiCorpus


if __name__ == '__main__':

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s_%(levelname)s:%(message)s')
    logger.setLevel(level=logging.INFO)
    logger.info("\n\n------------------------\nRunning %s" % "".join(sys.argv))

    if len(sys.argv) < 3:
        # print( globals() ['__doc__'])
        # print(globals()['__doc__'] % locals())
        # print(globals())
        # print( locals())
        logger.error("I neede more parameter!")
        sys.exit(1)

    inp, outp = sys.argv[1:3]
    space = " "
    i = 0

    with open(outp, "w") as outfile:
       wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
       for text in wiki.get_texts():
           # print(text)
           text = zhconv.convert(text, 'zh-hans')
           words_line = " ".join(jieba.cut(text.split('\n')[0].replace(' ', ''))) + '\n'
           # print(words_line)
           #去除非中文字符
           cn_reg = r'[\u4e00-\u9fa5]+'  #
            # if(re.search(cn_reg, words_line)):
           #选出所有中文字词并用空格连接起来
           print(re.search(cn_reg, words_line).groups());
           words_line_clear = " ".join(\
               [item.groups() for item in re.findall(cn_reg, words_line)] )

           print(words_line_clear)
           break
           i = i + 1
           if(i % 10000 == 0):
                logger.info('Saved %d articles.' % i)

    logger.info("Finished Saved %d articles." % i)




