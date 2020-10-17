{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'gensim.models.Word2Vec'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-7-f8a53faccf7e>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mos\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 4\u001b[1;33m \u001b[1;32mfrom\u001b[0m \u001b[0mgensim\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mmodels\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mWord2Vec\u001b[0m  \u001b[1;32mimport\u001b[0m \u001b[0mLineSentence\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      5\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0mgensim\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtest\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mutils\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mdatapath\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'gensim.models.Word2Vec'"
     ]
    }
   ],
   "source": [
    "import sys\n",
    "import os \n",
    "\n",
    "from gensim.models.Word2Vec  import LineSentence\n",
    "from gensim.test.utils import datapath "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "ename": "AttributeError",
     "evalue": "type object 'Word2Vec' has no attribute 'LineSentence'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-9-e2088c41fdad>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0msentence\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mWord2Vec\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mLineSentence\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdatapath\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'zhwiki.text'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[0mnet\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mWord2Vec\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mLineSentence\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0moutput_path\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0msize\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m400\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mwindow\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m5\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmin_count\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m5\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mworkers\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m6\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[0mnet_output\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m'wiki.zh.model'\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mnet\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msave\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mnet_output\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mAttributeError\u001b[0m: type object 'Word2Vec' has no attribute 'LineSentence'"
     ]
    }
   ],
   "source": [
    "sentence = Word2Vec.LineSentence(datapath('zhwiki.text'))\n",
    "net = Word2Vec(sentence, size=400, window=5, min_count=5, workers=6)\n",
    "net_output = 'wiki.zh.model'\n",
    "net.save(net_output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "net = Word2Vec.load(net_output)\n",
    "# 华为词对应的numpy向量\n",
    "net.wv['华为']\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "net.wv.most_similar(\"华为\", topn=5)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
