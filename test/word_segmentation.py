import py_vncorenlp
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='C:/Work/NLP/PRIMER')


def wordsegment(sent):
    return ' '.join(rdrsegmenter.word_segment(sent))

# sent="Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."
# print(wordsegment(""))