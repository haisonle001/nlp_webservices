# coding: utf8
from primer_main import *
from word_segmentation import *
from ner import *
from flask import Flask, jsonify,request
import py_vncorenlp
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='C:/Work/NLP/PRIMER')

app = Flask(__name__) 
app.config['JSON_AS_ASCII'] = False
args=initialize_args()    
print("Loading PRIMER model ...")
nlp = phonlp.load(save_dir='C:/Work/NLP/nlp_webservices')
model = PRIMERSummarizerLN.load_from_checkpoint(args.resume_ckpt, args=args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)


@app.route('/predict',methods=["POST","GET"])
def predict():
    # predictt(args,model)
    # with open("C:/Work/NLP/web/test/predicted_folder/prediction.txt","r",encoding='utf8') as file:
    #     txt=file.readlines()
    if request.method == 'POST':
        docs=request.form.get("text")
        txt=initialize_model(args,model,docs,rdrsegmenter)
        return jsonify({'text': txt[0]})

@app.route('/wordsegmentation',methods=["POST"])
def wordsegmentation():
    if request.method == 'POST':
        sent=request.form.get("text")
        # sent="Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."
        sent=' '.join(rdrsegmenter.word_segment(sent))
        return jsonify({'text': sent})

@app.route('/nermodel',methods=["POST"])
def nermodel():
    if request.method == 'POST':
        sent=request.form.get("text")
        # sent="Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."
        entity_pyramid, entity,res = get_entities(nlp, sent)
        return jsonify(res)


if __name__ == '__main__':
    app.run()   

