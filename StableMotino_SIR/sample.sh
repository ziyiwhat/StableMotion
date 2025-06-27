python sample.py --num 1
echo "12 step:" >> ./metrics.txt
python metric.py --results ./results --gts ./DIR-D/testing/gt >> ./metrics.txt