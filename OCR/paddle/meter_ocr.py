from paddleocr import PaddleOCR

ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)
result = ocr.predict('m14.png')

for res in result:
    for text, score in zip(res['rec_texts'], res['rec_scores']):
        print(text, score)
