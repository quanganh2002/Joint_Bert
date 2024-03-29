# JointBERT: Joint intent detection and slot filling

## Kiến trúc mô hình 
![image](https://github.com/quanganh2002/Joint_Bert/assets/138965151/ca0de96c-c8e9-44c9-ad7c-3d4c42b2cad1)

## Dữ liệu
![image](https://github.com/quanganh2002/Joint_Bert/assets/138965151/0947e3a3-e924-469b-b5d1-83e590bfd88b)
Dữ liệu gồm các câu lệnh người dùng với 10 nhãn intent và 23 nhãn slot theo cấu trúc
- seq.in: chứa các câu cần phân loại intent và slot
- label: chứa các nhãn intent cho từng câu trong seq.in
- seq.out: chứa các nhãn slot cho từng từ của mỗi câu trong seq.in
- intent_label.txt: là tập từ điển cho các nhãn intent
- slot_label.txt: là tập từ điển cho các nhãn slot

![image](https://github.com/quanganh2002/Joint_Bert/assets/138965151/d08aa159-ef4c-400e-a360-329cb01494a9)
![image](https://github.com/quanganh2002/Joint_Bert/assets/138965151/71b1fc0d-3bdd-4f71-9c27-f0a8b4dd1976)

## Cài đặt
- Phiên bản Python >= 3.6
- Phiên bản PyTorch >= 1.4.0
- Cài đặt thư viện:


```
pip3 install torch
pip3 install transformers
pip3 install seqeval
pip3 install pytorch-crf
pip3 install tensorflow
pip3 install sentencepiece
pip3 install tensorboard
pip3 install numpy
pip3 install tqdm
pip3 install pydrive
pip3 install scikit-learn
```

## Huấn luyện
Chạy lệnh này để thực hiện quá trình training:
```
python3 main.py --token_level syllable-level \
                --model_type phobert \
                --model_dir MODEL_DIR_jointBERT-CRF_PhoBERTencoder \
                --data_dir NLU \
                --seed 100 \
                --do_train \
                --do_eval \
                --save_steps 140 \
                --logging_steps 140 \
                --num_train_epochs 50 \
                --tuning_metric mean_intent_slot \
                --use_crf \
                --gpu_id 0 \
                --embedding_type soft \
                --intent_loss_coef 0.6 \
                --learning_rate 3e-5
```

## Đánh giá
- Thực hiện lệnh dưới để đánh giá mô hình đã được lưu:
```
python3 predict.py  --input_file 'MODEL_DIR_jointBERT-CRF_PhoBERTencoder/private_test_data/seq.in' \
                    --output_file 'MODEL_DIR_jointBERT-CRF_PhoBERTencoder/private_test_data/submission_results.csv' \
                    --model_dir 'MODEL_DIR_jointBERT-CRF_PhoBERTencoder'

%cd MODEL_DIR_jointBERT-CRF_PhoBERTencoder/private_test_data
python3 evaluate.py
```
```
python3 predict.py  --input_file 'MODEL_DIR_jointBERT-CRF_PhoBERTencoder/public_test_data/seq.in' \
                    --output_file 'MODEL_DIR_jointBERT-CRF_PhoBERTencoder/public_test_data/submission_results.csv' \
                    --model_dir 'MODEL_DIR_jointBERT-CRF_PhoBERTencoder'

%cd MODEL_DIR_jointBERT-CRF_PhoBERTencoder/private_test_data
python3 evaluate.py
```
- Dữ liệu của mô hình đã được huấn luyện được lưu với tên:  MODEL_DIR_JointBERT-CRF_PhoBERTencoder
- Trong đó có 2 thư mục private_test_data và public_test_data lưu kết quả đánh giá model với file submission_results.csv chứa kết quả của bài toán và file scores.txt chứa kết quả được đánh giá bằng sentence accuracy.
