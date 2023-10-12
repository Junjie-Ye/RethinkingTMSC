#!/bin/bash
export PYTHONPATH=../

for i in 'Twitter15' 'Twitter17' 
do
    echo ${i}

    for k in 'Bert', 'ResNet' 'ResBert' 'ResBertTFN' 'Res2Bert' 'Bert2Res' 'Res22Bert' 'ResBertAtt'
    do
        echo ${k} 
        for j in 2e-5
        do
            echo ${j}
            for q in 0 42 199 2022 11122
            do 
                echo ${q}
                for p in 8.0
                do 
                    echo ${p}
                    for f in test.tsv
                    do
                        PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1 python Code/training/run_data_analysis.py --data_dir \
                        data/${i} --task_name ${i} --output_dir Code/output/${i}/${k}/${q}_output_test/ --learning_rate ${j} \
                        --seed ${q} --test_file ${f} --bert_model bert-base-uncased --encoder resnet --do_train --do_eval --train_batch_size 32 --mm_model ${k} --num_train_epochs ${p}
                    done
                done
            done
        done
    done

    for k in 'Vit' 'VitBert' 'VitBertTFN' 'Vit2Bert' 'Bert2Vit' 'Vit22Bert' 'VitBertAtt'
    do
        echo ${k} 
        for j in 2e-5
        do
            echo ${j}
            for q in 0 42 199 2022 11122
            do 
                echo ${q}
                for p in 8.0
                do 
                    echo ${p}
                    for f in test.tsv
                    do
                        PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0 python Code/training/run_data_analysis.py --data_dir \
                        data/${i} --task_name ${i} --output_dir Code/output/${i}/${k}/${q}_output_test/ --learning_rate ${j} \
                        --seed ${q} --test_file ${f} --bert_model bert-base-uncased --encoder vit --do_train --do_eval --train_batch_size 32 --mm_model ${k} --num_train_epochs ${p} 
                done
            done
        done
    done
    
    for k in 'FasterRCNN' 'FasterBert' 'FasterBertTFN' 'Faster2Bert' 'Bert2Faster' 'Faster22Bert' 'FasterBertAtt'
    do
        echo ${k} 
        for j in 2e-5
        do
            echo ${j}
            for q in 0 42 199 2022 11122
            do 
                echo ${q}
                for p in 8.0
                do 
                    echo ${p}
                    for f in test.tsv
                    do
                        PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1 python Code/training/run_data_analysis.py --data_dir \
                        data/${i} --task_name ${i} --output_dir Code/output/${i}/${k}/${q}_output_test/ --learning_rate ${j} \
                        --seed ${q} --test_file ${f} --bert_model bert-base-uncased --encoder faster --do_train --do_eval --train_batch_size 32 --mm_model ${k} --num_train_epochs ${p} 
                    done
                done
            done
        done
    done    
done