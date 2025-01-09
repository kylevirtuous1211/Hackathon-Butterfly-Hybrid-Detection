How to train the model
1. download `both` directory at https://drive.google.com/drive/folders/1IqKJkVNm7QnQ81fPxPkm63J8TlTOPBo6?usp=sharing
* structure your directories like this:
```
Hackathon-Butterfly-Hybrid (clone 下來的)
│
├── input_data (創一個資料夾，把雲端下載的`both`放進去)
│   ├──both (下載放這裡面)
├── DINO_notebook (跑這裡面的notebook就好了)
├── DINO_train (這個不用動)
├── submission (裡面有兩個上傳範例)
│   ├── submission_detr_svm/
│   ├── submission_DINO_svm/
│   ├── model.py 
```


2. Open `DINO_notebook.ipynb` 然後開始跑應該不會有 bug 如果資料夾按照我說的做
* 跑完notebook生出來的.pkl檔改名成`clf.pkl`(因為創建的時候檔名會是`DINO_svm.pkl`之類的)
3. 上傳codabench小教學
* submission 的時候把我給你的`model.py`, 你生的`clf.pkl` `requirements.txt` 打包成.zip檔
