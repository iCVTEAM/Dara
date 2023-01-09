#!/bin/sh

#download dataset
#https://drive.google.com/file/d/15Dw4d-GMIPBIrLcK1xB8RpJlANPWUujf/view?usp=sharing

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=15Dw4d-GMIPBIrLcK1xB8RpJlANPWUujf' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=15Dw4d-GMIPBIrLcK1xB8RpJlANPWUujf" -O dara_pretrain_model.tar && rm -rf /tmp/cookies.txt

tar -zxvf dara_pretrain_model.tar
