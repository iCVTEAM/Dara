#!/bin/sh

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ZNIp77WRkDTRVGMRgb_qHUsFagAvU_V2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ZNIp77WRkDTRVGMRgb_qHUsFagAvU_V2" -O dara_pretrain_model.tar && rm -rf /tmp/cookies.txt
